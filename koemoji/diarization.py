"""話者分離（CPU）。

VAD で発話区間を切り出し、sherpa-onnx の話者埋め込み（声紋）を抽出して
クラスタリングし、「誰がいつ話したか」を推定する。Whisper の単語
タイムスタンプと突き合わせて「話者1: ...」形式の会話ログを作れる。

クラスタリングは 3 秒窓ごとの声紋に対して行い、話者数が未指定のときは
凝集クラスタリングの分離度（シルエット係数）から自動推定する。
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
MAX_AUTO_SPEAKERS = 8
MIN_SPEAKER_DISTANCE = 0.35
MIN_SILHOUETTE = 0.15
MIN_CLUSTER_FRACTION = 0.03
MAX_FIT_WINDOWS = 600

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


def _linkage(
    embeddings: Sequence,
    target: Optional[int] = None,
    threshold: Optional[float] = None,
    snapshots_at: int = 0,
    distances=None,
) -> Tuple[List[int], List[float], Dict[int, List[int]]]:
    """平均リンケージ凝集クラスタリング（コサイン距離）。

    target を指定するとそのクラスタ数で停止する。threshold を指定すると
    統合距離が threshold を超えた時点で停止する。どちらも None なら
    1 クラスタまで統合し、統合距離の履歴（昇順）を返す。
    snapshots_at > 0 のとき、クラスタ数がその値以下の各段階のラベルを保存する。
    """
    import heapq

    import numpy as np

    count = len(embeddings)
    if count == 0:
        return [], [], {}
    if count == 1:
        labels = [0]
        return labels, [], {1: labels}

    if distances is None:
        matrix = np.stack([np.asarray(e, dtype=np.float64) for e in embeddings])
        distances = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
        np.fill_diagonal(distances, 0.0)
    else:
        distances = np.array(distances, dtype=np.float64, copy=True)

    sizes = np.ones(count, dtype=np.float64)
    active = np.ones(count, dtype=bool)
    labels = np.arange(count)
    heap: List[Tuple[float, int, int]] = []
    for i in range(count):
        for j in range(i + 1, count):
            heapq.heappush(heap, (float(distances[i, j]), i, j))

    heights: List[float] = []
    snapshots: Dict[int, List[int]] = {}
    remaining = count

    while remaining > 1 and heap:
        distance, i, j = heapq.heappop(heap)
        if not (active[i] and active[j]):
            continue
        if abs(distance - float(distances[i, j])) > 1e-9:
            continue
        if target is not None and remaining <= target:
            break
        if target is None and threshold is not None and distance > threshold:
            break

        heights.append(distance)
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

        if snapshots_at and remaining <= snapshots_at:
            snapshots[remaining] = _remap(labels)

    return _remap(labels), heights, snapshots


def _remap(labels) -> List[int]:
    mapping: Dict[int, int] = {}
    result: List[int] = []
    for value in labels:
        key = int(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        result.append(mapping[key])
    return result


def _agglomerative_cluster(
    embeddings: Sequence,
    threshold: float = 0.5,
    num_clusters: int = -1,
) -> List[int]:
    """平均リンケージ凝集クラスタリング（コサイン距離）。

    threshold はコサイン距離（1 - 類似度）。小さいほど同じ話者にまとまりやすい。
    num_clusters が正の値なら、距離に関係なくその数まで統合する。
    """
    if num_clusters > 0:
        labels, _, _ = _linkage(embeddings, target=num_clusters)
    else:
        labels, _, _ = _linkage(embeddings, threshold=threshold)
    return labels


def _silhouette_score(labels: Sequence[int], distances) -> float:
    """クラスタリングの分離度（シルエット係数、-1〜1）。"""
    import numpy as np

    labels_array = np.asarray([int(label) for label in labels])
    unique = np.unique(labels_array)
    if len(unique) < 2:
        return -1.0
    count = len(labels_array)
    scores: List[float] = []
    for index in range(count):
        own = labels_array == labels_array[index]
        own[index] = False
        own_count = int(own.sum())
        if own_count == 0:
            scores.append(0.0)
            continue
        inside = float(distances[index, own].mean())
        outside = float("inf")
        for label in unique:
            if label == labels_array[index]:
                continue
            other = labels_array == label
            if other.any():
                outside = min(outside, float(distances[index, other].mean()))
        if outside == float("inf"):
            scores.append(0.0)
        else:
            scores.append((outside - inside) / max(inside, outside, 1e-12))
    return float(np.mean(scores))


def _cluster_embeddings(
    embeddings: Sequence,
    num_speakers: int = -1,
    threshold: Optional[float] = None,
) -> List[int]:
    """声紋をクラスタリングしてラベルを返す。

    num_speakers が正なら人数を固定。threshold 指定時は距離しきい値で分割。
    どちらも未指定なら分離度が最大になる人数を自動推定する。
    """
    count = len(embeddings)
    if count == 0:
        return []
    if count == 1:
        return [0]
    if num_speakers > 0:
        return _agglomerative_cluster(embeddings, num_clusters=num_speakers)
    if threshold is not None:
        return _agglomerative_cluster(embeddings, threshold=threshold)

    import numpy as np

    matrix = np.stack([np.asarray(e, dtype=np.float64) for e in embeddings])
    distances = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    _, heights, snapshots = _linkage(
        embeddings, snapshots_at=MAX_AUTO_SPEAKERS, distances=distances
    )

    min_size = max(2, int(np.ceil(count * MIN_CLUSTER_FRACTION)))
    best_labels: Optional[List[int]] = None
    best_score = -1.0
    for k in range(2, min(MAX_AUTO_SPEAKERS, count) + 1):
        if float(heights[count - k]) < MIN_SPEAKER_DISTANCE:
            break
        labels = snapshots.get(k)
        if labels is None:
            continue
        labels_array = np.asarray(labels)
        sizes = np.bincount(labels_array)
        core = [label for label in range(len(sizes)) if sizes[label] >= min_size]
        if len(core) < 2:
            continue
        centroids = np.stack(
            [matrix[labels_array == label].mean(axis=0) for label in core]
        )
        centroids = centroids / np.maximum(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12
        )
        effective = _remap((matrix @ centroids.T).argmax(axis=1).tolist())
        score = _silhouette_score(effective, distances)
        if score > best_score + 0.02:
            best_score = score
            best_labels = effective
    if best_labels is None or best_score < MIN_SILHOUETTE:
        return [0] * count
    return best_labels


def _build_spans(
    labeled: Sequence[Tuple[float, float, int]],
) -> List[SpeakerSpan]:
    spans: List[SpeakerSpan] = []
    for start, end, speaker in labeled:
        if spans and spans[-1].speaker == speaker and start - spans[-1].end <= MAX_GAP_MERGE:
            spans[-1].end = max(spans[-1].end, end)
        else:
            spans.append(SpeakerSpan(start, end, speaker))

    # 3秒窓は 1.5 秒ずつ重なるため、話者の変わり目は重なりの中央で切る。
    for previous, current in zip(spans, spans[1:]):
        if previous.end > current.start:
            boundary = (previous.end + current.start) / 2.0
            previous.end = boundary
            current.start = boundary
    return [span for span in spans if span.end - span.start > 0.05]


def _smooth_labels(
    windows: Sequence[Tuple[float, float, object]],
    labels: Sequence[int],
) -> List[int]:
    """孤立した 1 窓だけのラベルを前後に合わせる（時間方向の平滑化）。"""
    smoothed = list(labels)
    for i in range(1, len(labels) - 1):
        if labels[i] == labels[i - 1] or labels[i] == labels[i + 1]:
            continue
        if labels[i - 1] != labels[i + 1]:
            continue
        if windows[i + 1][0] - windows[i][0] > WINDOW_SECONDS:
            continue
        smoothed[i] = labels[i - 1]
    return smoothed


def diarize(
    audio_path: Path,
    embedding: str = DEFAULT_EMBEDDING,
    num_speakers: int = -1,
    threshold: Optional[float] = None,
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

    windows: List[Tuple[float, float, object]] = []
    for index, (start, end) in enumerate(regions):
        if _cancelled():
            return []
        for window_start, window_end in _windows_for_region(start, end):
            vector = _embed(extractor, samples, window_start, window_end)
            if vector is not None:
                windows.append((window_start, window_end, vector))
        if on_progress is not None and regions:
            frac = 0.1 + 0.55 * (index + 1) / len(regions)
            on_progress(frac, f"声紋を抽出中... {frac * 100:.0f}%")

    if not windows:
        return []

    if on_progress is not None:
        on_progress(0.7, "話者をクラスタリング中...")

    vectors = [window[2] for window in windows]
    if len(vectors) > MAX_FIT_WINDOWS:
        stride = (len(vectors) + MAX_FIT_WINDOWS - 1) // MAX_FIT_WINDOWS
        fit_vectors = vectors[::stride]
    else:
        fit_vectors = vectors

    fit_labels = _cluster_embeddings(fit_vectors, num_speakers, threshold)
    cluster_count = len(set(fit_labels))

    centroids: Dict[int, object] = {}
    for label, vector in zip(fit_labels, fit_vectors):
        if label not in centroids:
            centroids[label] = np.array(vector, dtype=np.float64)
        else:
            centroids[label] = centroids[label] + vector
    centroid_keys = sorted(centroids)
    centroid_matrix = np.stack(
        [centroids[key] / max(float(np.linalg.norm(centroids[key])), 1e-12) for key in centroid_keys]
    )

    if cluster_count <= 1:
        labels = [0] * len(vectors)
    else:
        matrix = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])
        similarities = matrix @ centroid_matrix.T
        labels = [centroid_keys[int(index)] for index in similarities.argmax(axis=1)]
        labels = _remap(labels)
        labels = _smooth_labels(windows, labels)

    if on_progress is not None:
        on_progress(0.85, "発話ごとに話者を割り当て中...")
    labeled = [(window[0], window[1], label) for window, label in zip(windows, labels)]
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
