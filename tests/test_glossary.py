from pathlib import Path
from types import SimpleNamespace

import pytest

import koemoji.core as core
from koemoji.cli import build_parser
from koemoji.core import TranscribeOptions, transcribe_file
from koemoji.glossary import MAX_PROMPT_CHARS, Glossary, load_glossary


def test_load_glossary(tmp_path: Path):
    path = tmp_path / "glossary.txt"
    path.write_text(
        "# コメント\n"
        "\n"
        "株式会社サンプル\n"
        "  プロジェクトあおぞら  \n"
        "サンプル社=>株式会社サンプル\n"
        "誤り -> 正しい\n"
        "タブ\t置換\n",
        encoding="utf-8-sig",
    )

    glossary = load_glossary(path)

    assert glossary.terms == ("株式会社サンプル", "プロジェクトあおぞら")
    assert glossary.replacements == (
        ("サンプル社", "株式会社サンプル"),
        ("誤り", "正しい"),
        ("タブ", "置換"),
    )
    assert glossary.is_empty is False


def test_empty_glossary(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("# 用語なし\n", encoding="utf-8")

    glossary = load_glossary(path)

    assert glossary.is_empty is True
    assert glossary.build_prompt(None) is None
    assert glossary.build_prompt("ヒント") == "ヒント"


def test_build_prompt_deduplicates_targets():
    glossary = Glossary(
        terms=("株式会社サンプル",),
        replacements=(("サンプル社", "株式会社サンプル"),),
    )

    prompt = glossary.build_prompt("会議の文字起こし")

    assert prompt.startswith("会議の文字起こし。用語: ")
    assert prompt.count("株式会社サンプル") == 1


def test_build_prompt_keeps_existing_period():
    glossary = Glossary(terms=("株式会社サンプル",))

    prompt = glossary.build_prompt("会議の文字起こし。")

    assert prompt == "会議の文字起こし。用語: 株式会社サンプル"


def test_build_prompt_respects_limit():
    glossary = Glossary(terms=tuple(f"用語{i:03d}" for i in range(100)))

    prompt = glossary.build_prompt("ヒント")

    assert prompt is not None
    assert len(prompt) <= MAX_PROMPT_CHARS
    assert prompt.startswith("ヒント。用語: ")
    assert "用語000" in prompt


def test_apply_longest_match_first_without_cascade():
    glossary = Glossary(
        replacements=(
            ("A", "B"),
            ("AB", "C"),
            ("B", "D"),
        )
    )

    assert glossary.apply("AB A B") == "C B D"


def test_apply_is_literal():
    glossary = Glossary(replacements=(("1+1", "2"),))

    assert glossary.apply("1+1 と 1+1") == "2 と 2"
    assert glossary.apply("11") == "11"


def test_cp932_glossary(tmp_path: Path):
    path = tmp_path / "sjis.txt"
    path.write_bytes("株式会社サンプル\n誤り=>正しい\n".encode("cp932"))

    glossary = load_glossary(path)

    assert glossary.terms == ("株式会社サンプル",)
    assert glossary.replacements == (("誤り", "正しい"),)


def test_invalid_replacement(tmp_path: Path):
    path = tmp_path / "invalid.txt"
    path.write_text("=>正しい\n", encoding="utf-8")

    with pytest.raises(ValueError, match="形式"):
        load_glossary(path)


def test_duplicate_replacement_target(tmp_path: Path):
    path = tmp_path / "duplicate.txt"
    path.write_text("誤り=>正しい1\n誤り=>正しい2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="重複"):
        load_glossary(path)


def test_missing_glossary(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_glossary(tmp_path / "missing.txt")


def test_parser_glossary():
    parser = build_parser()
    args = parser.parse_args(["transcribe", "a.mp3", "--glossary", "terms.txt"])

    assert args.glossary == "terms.txt"


def test_transcribe_applies_glossary(tmp_path: Path, monkeypatch):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    captured = {}

    class FakeModel:
        def transcribe(self, *args, **kwargs):
            captured.update(kwargs)
            segment = SimpleNamespace(start=0.0, end=1.0, text="誤りです。", words=None)
            info = SimpleNamespace(duration=1.0, language="ja", language_probability=1.0)
            return iter([segment]), info

    monkeypatch.setattr(core, "load_model", lambda *args, **kwargs: FakeModel())
    glossary = Glossary(terms=("株式会社サンプル",), replacements=(("誤り", "正しい"),))

    result = transcribe_file(
        audio,
        TranscribeOptions(glossary=glossary, initial_prompt="会議の文字起こし"),
    )

    assert result.text == "正しいです。"
    assert captured["initial_prompt"] == "会議の文字起こし。用語: 株式会社サンプル、正しい"
