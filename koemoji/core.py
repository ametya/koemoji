"""文字起こしエンジン本体（faster-whisper / CTranslate2 / CPU int8 固定）。"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .models import DEFAULT_EMBEDDING, DEFAULT_MODEL, default_model_dir, ensure_model, get_spec

# frac (0.0-1.0), message
ProgressCallback = Callable[[float, str], None]


class TranscribeCancelled(Exception):
    """ユーザー操作によるキャンセル。"""


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[int] = None


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class TranscribeOptions:
    model: str = DEFAULT_MODEL
    language: str = "ja"
    vad: bool = True
    beam_size: int = 5
    compute_type: str = "int8"
    initial_prompt: Optional[str] = None
    model_dir: Optional[Path] = None
    cpu_threads: Optional[int] = None
    diarize: bool = False
    num_speakers: int = -1
    diarize_threshold: Optional[float] = None
    diarize_embedding: str = DEFAULT_EMBEDDING


@dataclass
class TranscribeResult:
    source: Path
    segments: List[Segment]
    language: str
    language_probability: float
    duration: float
    elapsed: float
    num_speakers: int = 0

    @property
    def has_speakers(self) -> bool:
        return any(seg.speaker is not None for seg in self.segments)

    @property
    def text(self) -> str:
        """全文テキスト。話者分離済みなら会話ログ形式で返す。"""
        if self.has_speakers:
            return to_log(self.segments).strip()
        sep = "" if self.language.startswith("ja") else " "
        return sep.join(seg.text for seg in self.segments).strip()


_model_cache: Dict[Tuple[str, int], object] = {}
_model_lock = threading.Lock()


def load_model(
    model_name: str = DEFAULT_MODEL,
    model_dir: Optional[Path] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cpu_threads: Optional[int] = None,
    compute_type: str = "int8",
):
    """モデルを（必要ならダウンロードして）読み込む。プロセス内でキャッシュする。"""
    from faster_whisper import WhisperModel

    spec = get_spec(model_name)
    threads = cpu_threads or os.cpu_count() or 4
    cache_key = (spec.repo_id, threads, compute_type)

    with _model_lock:
        cached = _model_cache.get(cache_key)
    if cached is not None:
        return cached

    def _dl(done: int, total: int) -> None:
        if progress_cb is None:
            return
        frac = done / total if total else 0.0
        progress_cb(
            frac,
            f"モデル「{model_name}」をダウンロード中 {done / 1048576:.0f}MB / {total / 1048576:.0f}MB（初回のみ）",
        )

    local_path = ensure_model(model_name, model_dir, progress_cb=_dl)

    if progress_cb is not None:
        progress_cb(0.0, f"モデル「{model_name}」をCPUに読み込み中...")

    model = WhisperModel(
        local_path,
        device="cpu",
        compute_type=compute_type,
        cpu_threads=threads,
        num_workers=1,
    )

    with _model_lock:
        _model_cache[cache_key] = model
    return model


def transcribe_file(
    audio_path: Path,
    options: Optional[TranscribeOptions] = None,
    on_progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> TranscribeResult:
    """音声・動画ファイルを日本語（既定）で文字起こしする。"""
    options = options or TranscribeOptions()
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"ファイルが見つかりません: {audio_path}")

    spec = get_spec(options.model)
    if options.diarize and not spec.supports_word_timestamps:
        raise ValueError(
            f"モデル「{options.model}」は単語タイムスタンプに未対応のため、話者分離と併用できません。"
            "話者分離には turbo / small / medium / large-v3 を使ってください。"
        )

    if on_progress is not None:
        on_progress(0.0, "モデルを準備中...")
    model = load_model(
        options.model,
        model_dir=options.model_dir,
        progress_cb=on_progress,
        cpu_threads=options.cpu_threads,
        compute_type=options.compute_type,
    )

    if cancel_event is not None and cancel_event.is_set():
        raise TranscribeCancelled()

    language = None if options.language in (None, "", "auto") else options.language
    if on_progress is not None:
        on_progress(0.0, "音声を解析中...")

    started = time.perf_counter()
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=options.beam_size,
        vad_filter=options.vad,
        vad_parameters={"min_silence_duration_ms": 500} if options.vad else None,
        initial_prompt=options.initial_prompt,
        condition_on_previous_text=False,
        word_timestamps=options.diarize,
    )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    segments: List[Segment] = []
    words: List[Word] = []

    for seg in segments_iter:
        if cancel_event is not None and cancel_event.is_set():
            raise TranscribeCancelled()
        text = (seg.text or "").strip()
        if text:
            segments.append(Segment(float(seg.start), float(seg.end), text))
        if options.diarize:
            for word in getattr(seg, "words", None) or []:
                token = word.word or ""
                if token.strip():
                    words.append(Word(float(word.start), float(word.end), token))
        if on_progress is not None and duration > 0:
            frac = min(seg.end / duration, 1.0)
            on_progress(frac, f"文字起こし中... {frac * 100:.0f}%")

    num_speakers = 0
    if options.diarize:
        if cancel_event is not None and cancel_event.is_set():
            raise TranscribeCancelled()

        from .diarization import attribute_segments, diarize, merge_words_with_speakers

        if on_progress is not None:
            on_progress(0.0, "話者分離を開始します...")
        spans = diarize(
            audio_path,
            embedding=options.diarize_embedding,
            num_speakers=options.num_speakers,
            threshold=options.diarize_threshold,
            model_dir=options.model_dir,
            cpu_threads=options.cpu_threads,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise TranscribeCancelled()

        merged = merge_words_with_speakers(words, spans) if words else attribute_segments(segments, spans)
        if merged:
            segments = merged
            num_speakers = len({seg.speaker for seg in segments if seg.speaker is not None})

    elapsed = time.perf_counter() - started
    if on_progress is not None:
        on_progress(1.0, "完了")

    return TranscribeResult(
        source=audio_path,
        segments=segments,
        language=info.language,
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
        duration=duration,
        elapsed=elapsed,
        num_speakers=num_speakers,
    )


# ---------------------------------------------------------------------------
# 出力フォーマット
# ---------------------------------------------------------------------------


def _timestamp(seconds: float, decimal_sep: str) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_sep}{millis:03d}"


def _clock(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _speaker_prefix(segment: Segment) -> str:
    if segment.speaker is None:
        return ""
    return f"話者{segment.speaker + 1}: "


def to_log(segments: List[Segment]) -> str:
    """会話ログ形式。例: [00:01:23] 話者1: おはようございます。"""
    lines = [
        f"[{_clock(seg.start)}] {_speaker_prefix(seg)}{seg.text}"
        for seg in segments
    ]
    return "\n".join(lines) + "\n"


def to_txt(segments: List[Segment]) -> str:
    if any(seg.speaker is not None for seg in segments):
        return to_log(segments)
    return "\n".join(seg.text for seg in segments) + "\n"


def to_srt(segments: List[Segment]) -> str:
    lines: List[str] = []
    for index, seg in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(f"{_timestamp(seg.start, ',')} --> {_timestamp(seg.end, ',')}")
        lines.append(f"{_speaker_prefix(seg)}{seg.text}")
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: List[Segment]) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_timestamp(seg.start, '.')} --> {_timestamp(seg.end, '.')}")
        lines.append(f"{_speaker_prefix(seg)}{seg.text}")
        lines.append("")
    return "\n".join(lines)


FORMATS: Dict[str, Callable[[List[Segment]], str]] = {
    "txt": to_txt,
    "srt": to_srt,
    "vtt": to_vtt,
    "log": to_log,
}
