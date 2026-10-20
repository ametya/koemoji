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
    _timestamp,
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
