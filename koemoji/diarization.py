"""話者分離（CPU）。

VAD で発話区間を切り出し、sherpa-onnx の話者埋め込み（声紋）を抽出して
クラスタリングし、「誰がいつ話したか」を推定する。Whisper の単語
タイムスタンプと突き合わせて「話者1: ...」形式の会話ログを作れる。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .core import Segment, Word
from .models import DEFAULT_EMBEDDING, ensure_diarization_model

ProgressCallback = Callable[[float, str], None]

SAMPLE_RATE = 16000
WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.5
MAX_GAP_MERGE = 0.75
LONG_REGION_SECONDS = 15.0

_extractor_cache: Dict[Tuple[str, int], object] = {}
_extractor_lock = threading.Lock()


@dataclass
class SpeakerSpan:
    start: float
    end: float
    speaker: int


def _load_extractor(
    embedding: str,
    model_dir: Optional[Path],
    cpu_threads: Optional[int],
    on_progress: Optional[ProgressCallback] = None,
):
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "話者分離には sherpa-onnx が必要です: pip install sherpa-onnx"
        ) from exc

    def _model_progress(done: int, total: int) -> None:
        if on_progress is None:
            return
        frac = done / total if total else 0.0
        on_progress(
            frac,
            f"話者分離モデルをダウンロード中 {done / 1048576:.0f}MB / {total / 1048576:.0f}MB（初回のみ）",
        )

    model_path = ensure_diarization_model(embedding, model_dir, progress_cb=_model_progress)
    threads = cpu_threads or os.cpu_count() or 4
    cache_key = (str(model_path), threads)

    with _extractor_lock:
        cached = _extractor_cache.get(cache_key)
    if cached is not None:
        return cached

    if on_progress is not None:
        on_progress(0.0, "話者分離モデルをCPUに読み込み中...")

    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path),
        num_threads=threads,
        provider="cpu",
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    with _extractor_lock:
        _extractor_cache[cache_key] = extractor
    return extractor


def _speech_regions(samples) -> List[Tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=500, speech_pad_ms=200)
    raw = get_speech_timestamps(samples, options, sampling_rate=SAMPLE_RATE)

    regions: List[Tuple[float, float]] = []
    for item in raw:
        start = item["start"] / SAMPLE_RATE
        end = item["end"] / SAMPLE_RATE
        if end <= start:
            continue
        if regions and start - regions[-1][1] <= MAX_GAP_MERGE:
            regions[-1] = (regions[-1][0], max(regions[-1][1], end))
        else:
            regions.append((start, end))
    return regions


def _windows_for_region(start: float, end: float) -> List[Tuple[float, float]]:
    duration = end - start
    if duration <= WINDOW_SECONDS:
        return [(start, end)]
    windows: List[Tuple[float, float]] = []
    position = start
    while position + WINDOW_SECONDS < end - 0.2:
        windows.append((position, position + WINDOW_SECONDS))
        position += HOP_SECONDS
    windows.append((max(end - WINDOW_SECONDS, start), end))
    return windows


def _embed(extractor, samples, start: float, end: float):
    import numpy as np

    chunk = samples[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
    if len(chunk) == 0:
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(SAMPLE_RATE, chunk)
    stream.input_finished()
    vector = np.asarray(extractor.compute(stream), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return None
    return vector / norm


def _region_embedding(extractor, samples, start: float, end: float):
    import numpy as np

    vectors = []
    position = start
    while position < end:
        piece_end = min(position + WINDOW_SECONDS, end)
        vector = _embed(extractor, samples, position, piece_end)
        if vector is not None:
            vectors.append(vector)
        position = piece_end
    if not vectors:
        return None
    mean = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 0:
        return None
    return mean / norm


def _agglomerative_cluster(
    embeddings: Sequence,
    threshold: float = 0.5,
    num_clusters: int = -1,
) -> List[int]:
    """平均リンケージ凝集クラスタリング（コサイン距離）。

    threshold はコサイン距離（1 - 類似度）。小さいほど同じ話者にまとまりやすい。
    num_clusters が正の値なら、距離に関係なくその数まで統合する。
    """
    import heapq

    import numpy as np

    count = len(embeddings)
    if count == 0:
        return []
    if count == 1:
        return [0]

    matrix = np.stack([np.asarray(e, dtype=np.float64) for e in embeddings])
    distances = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)

    sizes = np.ones(count, dtype=np.float64)
    active = np.ones(count, dtype=bool)
    labels = np.arange(count)
    heap: List[Tuple[float, int, int]] = []
    for i in range(count):
        for j in range(i + 1, count):
            heapq.heappush(heap, (float(distances[i, j]), i, j))

    target = num_clusters if num_clusters > 0 else 1
    remaining = count

    while remaining > target and heap:
        distance, i, j = heapq.heappop(heap)
        if not (active[i] and active[j]):
            continue
        if abs(distance - float(distances[i, j])) > 1e-9:
            continue
        if num_clusters <= 0 and distance > threshold:
            break

        weight_i, weight_j = sizes[i], sizes[j]
        merged_row = (weight_i * distances[i] + weight_j * distances[j]) / (weight_i + weight_j)
        distances[i] = merged_row
        distances[:, i] = merged_row
        distances[i, i] = 0.0
        sizes[i] = weight_i + weight_j
        active[j] = False
        labels[labels == labels[j]] = labels[i]

        for k in np.nonzero(active)[0]:
            k = int(k)
            if k != i:
                a, b = (i, k) if i < k else (k, i)
                heapq.heappush(heap, (float(distances[a, b]), a, b))
        remaining -= 1

    remap: Dict[int, int] = {}
    result: List[int] = []
    for value in labels:
        key = int(value)
        if key not in remap:
            remap[key] = len(remap)
        result.append(remap[key])
    return result


def _build_spans(
    labeled: Sequence[Tuple[float, float, int]],
) -> List[SpeakerSpan]:
    spans: List[SpeakerSpan] = []
    for start, end, speaker in labeled:
        if spans and spans[-1].speaker == speaker and start - spans[-1].end <= MAX_GAP_MERGE:
            spans[-1].end = max(spans[-1].end, end)
        else:
            spans.append(SpeakerSpan(start, end, speaker))
    return spans


def diarize(
    audio_path: Path,
    embedding: str = DEFAULT_EMBEDDING,
    num_speakers: int = -1,
    threshold: float = 0.5,
    model_dir: Optional[Path] = None,
    cpu_threads: Optional[int] = None,
    on_progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> List[SpeakerSpan]:
    """音声ファイルから話者区間のリストを返す。"""
    import numpy as np
    from faster_whisper.audio import decode_audio

    extractor = _load_extractor(embedding, model_dir, cpu_threads, on_progress)

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    if on_progress is not None:
        on_progress(0.0, "音声を読み込み中（話者分離）...")
    samples = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    if _cancelled():
        return []

    if on_progress is not None:
        on_progress(0.05, "発話区間を検出中...")
    regions = _speech_regions(samples)
    if not regions:
        if on_progress is not None:
            on_progress(1.0, "発話区間が見つかりませんでした")
        return []

    # 長すぎる区間は分割（区間内の話者変化を拾いやすくする）
    split_regions: List[Tuple[float, float]] = []
    for start, end in regions:
        if end - start <= LONG_REGION_SECONDS:
            split_regions.append((start, end))
            continue
        position = start
        while position < end:
            piece_end = min(position + LONG_REGION_SECONDS, end)
            split_regions.append((position, piece_end))
            position = piece_end

    region_embeddings = []
    for index, (start, end) in enumerate(split_regions):
        if _cancelled():
            return []
        vector = _region_embedding(extractor, samples, start, end)
        region_embeddings.append(vector)
        if on_progress is not None and len(split_regions) > 0:
            frac = 0.1 + 0.55 * (index + 1) / len(split_regions)
            on_progress(frac, f"声紋を抽出中... {frac * 100:.0f}%")

    valid = [(region, vector) for region, vector in zip(split_regions, region_embeddings) if vector is not None]
    if not valid:
        return []
    regions_valid = [region for region, _ in valid]
    vectors = [vector for _, vector in valid]

    if on_progress is not None:
        on_progress(0.7, "話者をクラスタリング中...")
    labels = _agglomerative_cluster(vectors, threshold=threshold, num_clusters=num_speakers)

    centroids: Dict[int, object] = {}
    for label, vector in zip(labels, vectors):
        if label not in centroids:
            centroids[label] = vector.copy()
        else:
            centroids[label] = centroids[label] + vector

    # 長い区間は 3 秒窓ごとに最も近い話者へ割り当てる
    if on_progress is not None:
        on_progress(0.8, "発話ごとに話者を割り当て中...")
    labeled: List[Tuple[float, float, int]] = []
    for (start, end), label in zip(regions_valid, labels):
        windows = _windows_for_region(start, end)
        if len(windows) == 1:
            labeled.append((start, end, label))
            continue
        for window_start, window_end in windows:
            if _cancelled():
                return []
            vector = _embed(extractor, samples, window_start, window_end)
            if vector is None:
                labeled.append((window_start, window_end, label))
                continue
            best_label = label
            best_similarity = None
            for candidate, centroid in centroids.items():
                centroid_norm = float(np.linalg.norm(centroid))
                if centroid_norm <= 0:
                    continue
                similarity = float(vector @ (centroid / centroid_norm))
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity
                    best_label = candidate
            labeled.append((window_start, window_end, best_label))

    spans = _build_spans(labeled)
    if on_progress is not None:
        on_progress(1.0, "話者分離が完了しました")
    return spans


def _speaker_for_span(
    start: float,
    end: float,
    spans: Sequence[SpeakerSpan],
) -> Optional[int]:
    if not spans:
        return None
    if end < start:
        start, end = end, start

    best_speaker: Optional[int] = None
    best_overlap = 0.0
    for span in spans:
        overlap = min(end, span.end) - max(start, span.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = span.speaker
    if best_speaker is not None:
        return best_speaker

    midpoint = (start + end) / 2.0
    for span in spans:
        if span.start <= midpoint <= span.end:
            return span.speaker

    nearest_speaker: Optional[int] = None
    nearest_distance: Optional[float] = None
    for span in spans:
        distance = min(abs(midpoint - span.start), abs(midpoint - span.end))
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = span.speaker
    return nearest_speaker


class _SpeakerLabels:
    """話者IDを登場順に 0,1,2... へ振り直す。"""

    def __init__(self) -> None:
        self._mapping: Dict[int, int] = {}

    def label(self, raw: Optional[int]) -> Optional[int]:
        if raw is None:
            return None
        if raw not in self._mapping:
            self._mapping[raw] = len(self._mapping)
        return self._mapping[raw]

    def __len__(self) -> int:
        return len(self._mapping)


def merge_words_with_speakers(
    words: Sequence[Word],
    spans: Sequence[SpeakerSpan],
    gap_split: float = 1.5,
) -> List[Segment]:
    """単語タイムスタンプと話者区間を突き合わせ、発話ごとのセグメントに再構成する。

    話者が変わった箇所・無音が gap_split 秒以上空いた箇所で区切る。
    """
    if not words or not spans:
        return []

    labels = _SpeakerLabels()
    utterances: List[Segment] = []
    current: Optional[Segment] = None
    previous_end: Optional[float] = None

    for word in words:
        raw = _speaker_for_span(word.start, word.end, spans)
        speaker = labels.label(raw)
        gap = word.start - previous_end if previous_end is not None else 0.0
        if current is None or speaker != current.speaker or gap > gap_split:
            if current is not None and current.text.strip():
                utterances.append(current)
            current = Segment(start=word.start, end=word.end, text=word.text, speaker=speaker)
        else:
            current.text += word.text
            current.end = max(current.end, word.end)
        previous_end = word.end

    if current is not None and current.text.strip():
        utterances.append(current)

    for utterance in utterances:
        utterance.text = utterance.text.strip()
    return [utterance for utterance in utterances if utterance.text]


def attribute_segments(
    segments: Sequence[Segment],
    spans: Sequence[SpeakerSpan],
) -> List[Segment]:
    """単語タイムスタンプが無い場合のフォールバック（セグメント単位で話者を付与）。"""
    if not segments or not spans:
        return []

    labels = _SpeakerLabels()
    attributed: List[Segment] = []
    for segment in segments:
        raw = _speaker_for_span(segment.start, segment.end, spans)
        attributed.append(
            Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker=labels.label(raw),
            )
        )
    return attributed
