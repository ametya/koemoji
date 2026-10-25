"""モデルを必要としない軽量テスト（フォーマット・CLI・モデルカタログ）。"""

from pathlib import Path

from koemoji.cli import build_parser
from koemoji.core import (
    Segment,
    FORMATS,
    TranscribeOptions,
    transcribe_file,
    to_srt,
    to_txt,
    to_vtt,
    _collapse_repeats,
    _drop_hallucinations,
    _timestamp,
    resegment,
)
from koemoji.models import DEFAULT_MODEL, MODEL_CATALOG, get_spec


SEGMENTS = [
    Segment(0.0, 1.234, "こんにちは、世界。"),
    Segment(1.5, 65.4, "これは二行目のテキストです。"),
]


def test_timestamp_format():
    assert _timestamp(0.0, ",") == "00:00:00,000"
    assert _timestamp(3661.5, ",") == "01:01:01,500"
    assert _timestamp(1.2345, ".") == "00:00:01.234"


def test_to_txt():
    out = to_txt(SEGMENTS)
    assert out == "こんにちは、世界。\nこれは二行目のテキストです。\n"


def test_to_srt():
    out = to_srt(SEGMENTS)
    assert "1\n00:00:00,000 --> 00:00:01,234\nこんにちは、世界。\n" in out
    assert "2\n00:00:01,500 --> 00:01:05,400\n" in out
    assert out.endswith("\n\n") is False


def test_to_vtt():
    out = to_vtt(SEGMENTS)
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.234" in out


def test_formats_registry():
    assert set(FORMATS) == {"txt", "srt", "vtt", "log"}


def test_model_catalog():
    assert DEFAULT_MODEL in MODEL_CATALOG
    assert DEFAULT_MODEL == "turbo"
    for name, spec in MODEL_CATALOG.items():
        assert spec.name == name
        assert "/" in spec.repo_id
        assert spec.label
    assert MODEL_CATALOG["kotoba"].supports_word_timestamps is False
    assert MODEL_CATALOG["turbo"].supports_word_timestamps is True


def test_get_spec_error():
    try:
        get_spec("unknown-model")
    except KeyError as exc:
        assert "unknown-model" in str(exc)
    else:
        raise AssertionError("KeyError が発生しませんでした")


def test_parser_transcribe():
    parser = build_parser()
    args = parser.parse_args(["transcribe", "a.mp3", "-m", "small", "--formats", "txt,srt"])
    assert args.command == "transcribe"
    assert args.inputs == ["a.mp3"]
    assert args.model == "small"
    assert args.formats == "txt,srt"
    assert args.language == "ja"
    assert args.compute_type == "int8"


def test_diarize_rejects_model_without_word_timestamps(tmp_path: Path):
    dummy = tmp_path / "a.wav"
    dummy.write_bytes(b"")
    options = TranscribeOptions(model="kotoba", diarize=True, model_dir=tmp_path)
    try:
        transcribe_file(dummy, options)
    except ValueError as exc:
        assert "話者分離" in str(exc)
    else:
        raise AssertionError("ValueError が発生しませんでした")


def test_parser_models():
    parser = build_parser()
    args = parser.parse_args(["models", "--download", "small"])
    assert args.command == "models"
    assert args.download == ["small"]


def test_outdir_creation(tmp_path: Path):
    assert tmp_path.is_dir()


# --- 幻覚除去・行の再構成 ------------------------------------------------

def test_drop_hallucinations_removes_silent_segment():
    segments = [
        Segment(0.0, 1.0, "ありがとうございます。"),
        Segment(1.0, 2.0, "こんにちは"),
    ]
    probs = [(0.9, -1.5), (0.1, -0.3)]
    kept, kept_probs, dropped = _drop_hallucinations(segments, probs, 0.6, -1.0)
    assert [seg.text for seg in kept] == ["こんにちは"]
    assert kept_probs == [(0.1, -0.3)]
    assert dropped == [(0.0, 1.0)]

def test_drop_hallucinations_keeps_low_speech_prob_but_confident():
    # 無音確率は高いが確信度も高い → 落とさない（BGM 上の実発話を取りこぼさない）
    segments = [Segment(0.0, 1.0, "はい")]
    kept, _, dropped = _drop_hallucinations(segments, [(0.8, -0.2)], 0.6, -1.0)
    assert len(kept) == 1
    assert dropped == []

def test_collapse_repeats_merges_identical_neighbours():
    segments = [
        Segment(0.0, 1.0, "ご視聴ありがとうございました"),
        Segment(1.0, 2.0, "ご視聴ありがとうございました"),
        Segment(2.0, 3.0, "本題に入ります"),
    ]
    out = _collapse_repeats(segments)
    assert [seg.text for seg in out] == ["ご視聴ありがとうございました", "本題に入ります"]
    assert out[0].end == 2.0

def test_collapse_repeats_ignores_single_char():
    segments = [Segment(0.0, 1.0, "あ"), Segment(1.0, 2.0, "あ")]
    assert len(_collapse_repeats(segments)) == 2

def test_resegment_merges_short_lines():
    segments = [Segment(0.0, 1.0, "今日は"), Segment(1.1, 2.0, "いい天気ですね")]
    out = resegment(segments, max_chars=32, language="ja")
    assert len(out) == 1
    assert out[0].text == "今日はいい天気ですね"
    assert (out[0].start, out[0].end) == (0.0, 2.0)

def test_resegment_does_not_merge_across_sentence_end():
    segments = [Segment(0.0, 1.0, "なるほど。"), Segment(1.1, 2.0, "そうですね")]
    out = resegment(segments, max_chars=32, language="ja")
    assert [seg.text for seg in out] == ["なるほど。", "そうですね"]

def test_resegment_does_not_merge_across_speaker():
    a = Segment(0.0, 1.0, "はい", speaker=0)
    b = Segment(1.1, 2.0, "そうですか", speaker=1)
    out = resegment([a, b], max_chars=32, language="ja")
    assert [seg.speaker for seg in out] == [0, 1]

def test_resegment_splits_long_segment_by_punctuation():
    text = "あ" * 20 + "。" + "い" * 19
    out = resegment([Segment(0.0, 10.0, text)], max_chars=32, language="ja")
    assert len(out) == 2
    assert out[0].text.endswith("。")
    assert out[0].end == 10.0 * 21 / 40
    assert "".join(seg.text for seg in out) == text

def test_resegment_uses_space_for_english():
    segments = [Segment(0.0, 1.0, "hello"), Segment(1.1, 2.0, "world")]
    out = resegment(segments, max_chars=32, language="en")
    assert out[0].text == "hello world"

def test_resegment_disabled_returns_input():
    segments = [Segment(0.0, 1.0, "a"), Segment(1.0, 2.0, "b")]
    assert resegment(segments, max_chars=0) == segments
