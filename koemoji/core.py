"""文字起こしエンジン本体（faster-whisper / CTranslate2 / CPU int8 固定）。"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .models import DEFAULT_EMBEDDING, DEFAULT_MODEL, default_model_dir, ensure_model, get_spec

if TYPE_CHECKING:
    from .glossary import Glossary

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
    glossary: Optional["Glossary"] = None
    hotwords: Optional[str] = None
    temperature: Optional[float] = None
    drop_hallucinations: bool = True
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    resegment: bool = True
    max_line_chars: int = 32


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
    initial_prompt = options.initial_prompt
    if options.glossary is not None:
        initial_prompt = options.glossary.build_prompt(initial_prompt)
    if on_progress is not None:
        on_progress(0.0, "音声を解析中...")

    started = time.perf_counter()
    # temperature=None をそのまま渡すと faster-whisper 内部で temperatures=[None] になり
    # デコード時の比較で落ちるので、指定されたときだけ渡す（未指定は既定のフォールバック列）。
    transcribe_kwargs: Dict[str, Any] = dict(
        language=language,
        beam_size=options.beam_size,
        vad_filter=options.vad,
        vad_parameters={"min_silence_duration_ms": 500} if options.vad else None,
        initial_prompt=initial_prompt,
        hotwords=options.hotwords,
        no_speech_threshold=options.no_speech_threshold,
        log_prob_threshold=options.log_prob_threshold,
        condition_on_previous_text=False,
        word_timestamps=options.diarize,
    )
    if options.temperature is not None:
        transcribe_kwargs["temperature"] = options.temperature
    segments_iter, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    segments: List[Segment] = []
    seg_probs: List[Tuple[float, float]] = []
    words: List[Word] = []

    for seg in segments_iter:
        if cancel_event is not None and cancel_event.is_set():
            raise TranscribeCancelled()
        text = (seg.text or "").strip()
        if text:
            segments.append(Segment(float(seg.start), float(seg.end), text))
            seg_probs.append(
                (
                    float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
                    float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                )
            )
        if options.diarize:
            for word in getattr(seg, "words", None) or []:
                token = word.word or ""
                if token.strip():
                    words.append(Word(float(word.start), float(word.end), token))
        if on_progress is not None and duration > 0:
            frac = min(seg.end / duration, 1.0)
            on_progress(frac, f"文字起こし中... {frac * 100:.0f}%")

    # 幻覚（無音・BGM 区間でモデルが作文する）と同一文の繰り返しを取り除く。
    # 落とした区間の単語タイムスタンプも消さないと、話者分離の経路で復活する。
    if options.drop_hallucinations and segments:
        segments, seg_probs, dropped = _drop_hallucinations(
            segments,
            seg_probs,
            options.no_speech_threshold,
            options.log_prob_threshold,
        )
        if dropped:
            words = [w for w in words if not any(s < w.end and w.start < e for s, e in dropped)]
        segments = _collapse_repeats(segments)

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

    if options.resegment and segments and options.max_line_chars > 0:
        segments = resegment(
            segments,
            max_chars=options.max_line_chars,
            language=info.language,
        )

    # 用語集の適用は1回だけ行う（連結してからかけた方が、境界をまたぐ用語も直せる）。
    if options.glossary is not None and not options.glossary.is_empty:
        for segment in segments:
            segment.text = options.glossary.apply(segment.text)

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


# --- 後処理 -----------------------------------------------------------------

_SENTENCE_END = ("。", "．", "！", "？", "!", "?", "…")
# 長いセグメントを切るときに探す境界（句読点 → 読点 → 空白の順に優先）
_BREAK_CHARS = "。．！？!?…、，,　 "
# 句読点が無い音声（Whisper の ja 出力は句読点が付かないことが多い）で
# 語の途中を切らないための代替境界。助詞の直後は文節境界なので必ず語中にならない。
# ただし漢字の読みの一部（「上(あ)がる」の「が」など）と紛れることがあるため、
# 直後が小書き仮名（っ ゃ ゅ ょ …）のときは境界として採用しない。
_PARTICLE_CHARS = "はがをにへとのもやねよか"
# 「ので」「のに」のような複合的な接続表現。直前の助詞ではなく直後の文字までを1単位にする
_PARTICLE_PAIRS = {"の": "でにはがをも", "か": "ら", "で": ""}
_SMALL_KANA = "ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョ"


def _drop_hallucinations(
    segments: List[Segment],
    probs: List[Tuple[float, float]],
    no_speech_threshold: float,
    log_prob_threshold: float,
) -> Tuple[List[Segment], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """無音確率が高く確信度が低いセグメントを落とす。

    `probs` は各セグメントの (no_speech_prob, avg_logprob)。戻り値は
    (残したセグメント, 残した prob, 落とした区間の (start, end) 一覧)。
    """
    kept: List[Segment] = []
    kept_probs: List[Tuple[float, float]] = []
    dropped: List[Tuple[float, float]] = []
    for seg, prob in zip(segments, probs):
        no_speech_prob, avg_logprob = prob
        if no_speech_prob > no_speech_threshold and avg_logprob < log_prob_threshold:
            dropped.append((seg.start, seg.end))
            continue
        kept.append(seg)
        kept_probs.append(prob)
    return kept, kept_probs, dropped


def _collapse_repeats(segments: List[Segment]) -> List[Segment]:
    """同一テキストが連続する区間を1本にまとめる（幻覚のループ対策）。"""
    out: List[Segment] = []
    for seg in segments:
        if out and seg.text == out[-1].text and len(seg.text) >= 2:
            out[-1].end = max(out[-1].end, seg.end)
            continue
        out.append(seg)
    return out


def _find_cut(text: str, floor: int, limit: int) -> int:
    """text を limit 文字以内で切れる位置（切った直後のインデックス）を返す。無ければ 0。

    優先順は「句読点の直後 → 助詞の直後」。助詞の直後は文節境界なので、
    句読点の無い音声でも語の途中で切らずに済む。
    """
    end = min(len(text), limit)
    if end <= floor:
        return 0
    particle = 0
    for i in range(end - 1, floor - 1, -1):
        ch = text[i]
        if ch in _BREAK_CHARS:
            return i + 1
        if particle or ch not in _PARTICLE_CHARS:
            continue
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if nxt in _SMALL_KANA:
            continue  # 「上がって」の「が」のような読みの一部
        pair = _PARTICLE_PAIRS.get(ch)
        if pair and nxt in pair:
            if i + 2 <= end:
                particle = i + 2  # 「ので」まで含めて切る
            continue
        particle = i + 1
    return particle


def _split_long_segment(seg: Segment, max_chars: int) -> List[Segment]:
    """1セグメントを max_chars 前後で分割し、文字数比でタイムスタンプを按分する。"""
    text = seg.text
    if len(text) <= max_chars:
        return [seg]

    pieces: List[str] = []
    rest = text
    floor = max(max_chars // 2, 4)
    while len(rest) > max_chars:
        cut = _find_cut(rest, floor, max_chars)
        if cut <= 0:
            # 近傍に境界が無い → 少し先まで探す（行が長くなる方が語中で切るよりまし）
            cut = _find_cut(rest, floor, max_chars * 2)
        if cut <= 0:
            cut = max_chars
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)

    total = len(text) or 1
    span = max(seg.end - seg.start, 0.0)
    out: List[Segment] = []
    cursor = 0
    for piece in pieces:
        start = seg.start + span * (cursor / total)
        cursor += len(piece)
        end = seg.start + span * (cursor / total)
        body = piece.strip()
        if body:
            out.append(Segment(start, end, body, seg.speaker))
    return out


def resegment(
    segments: List[Segment],
    max_chars: int = 32,
    language: Optional[str] = "ja",
    gap_split: float = 0.6,
) -> List[Segment]:
    """Whisper のセグメントを読みやすい行に組み替える。

    Whisper は息継ぎで切る／切らないの振幅が大きく、1行が100文字を超えたり
    数文字で切れたりする。ここでは「同じ話者・無音が gap_split 秒未満・文末でない・
    連結しても max_chars 以内」のときだけ連結し、長すぎる行は句読点で割る。
    """
    if not segments or max_chars <= 0:
        return segments

    joiner = "" if (language or "").startswith("ja") or (language or "") == "zh" else " "

    pieces: List[Segment] = []
    for seg in segments:
        pieces.extend(_split_long_segment(seg, max_chars))

    merged: List[Segment] = []
    for seg in pieces:
        if merged:
            last = merged[-1]
            gap = seg.start - last.end
            joinable = (
                last.speaker == seg.speaker
                and gap <= gap_split
                and len(last.text) + len(seg.text) <= max_chars
                and not last.text.endswith(_SENTENCE_END)
            )
            if joinable:
                last.text = last.text + joiner + seg.text
                last.end = max(last.end, seg.end)
                continue
        merged.append(Segment(seg.start, seg.end, seg.text, seg.speaker))
    return merged


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
