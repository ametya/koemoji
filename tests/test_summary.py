"""議事録整理（ルールベース）のテスト。"""

from __future__ import annotations

from pathlib import Path

from koemoji.summary import (
    SummaryOptions,
    _merge_llm_json,
    _repair_names,
    _split_clauses,
    extract_dates,
    extract_facts,
    extract_people,
    infer_speakers,
    parse_transcript,
    summarize_text,
)

SAMPLE = """\
[00:00:01] 話者1: お電話ありがとうございます。モイネットシステムです。
[00:00:06] 話者2: 宮永薬局の平岡と申します。来年1月の導入に向けて、デモ機に患者データを入れたいんですけど、大丈夫ですか。
[00:00:20] 話者1: データのファイルを写していただくのは全く問題ないです。親機に入れれば子機からも参照できます。
[00:00:34] 話者2: 4台とも見れるという認識でよろしいですか。
[00:00:40] 話者1: はい、問題ないです。ただ通常と異なるため、担当者に確認して折り返します。
[00:00:52] 話者2: お願いします。12月までにデータを移す必要があります。
"""


def test_parse_transcript_reads_speakers_and_timestamps() -> None:
    utterances = parse_transcript(SAMPLE)
    assert utterances
    assert utterances[0].speaker == 0
    assert utterances[0].start == 1.0
    assert any(u.speaker == 1 and "宮永薬局" in u.text for u in utterances)


def test_infer_speakers_marks_requester_and_responder() -> None:
    utterances = parse_transcript(SAMPLE)
    requester, responder, labels = infer_speakers(utterances)
    assert (requester, responder) == (1, 0)
    assert "依頼側" in labels[1]
    assert "応対側" in labels[0]


def test_split_clauses_does_not_break_modal_phrase() -> None:
    clauses = _split_clauses("そのデータを今一台あるデモ機親機というものに移していいかという質問だった")
    assert any("ものに移して" in clause for clause in clauses)
    assert all("ものに" not in clause or "移して" in clause for clause in clauses)


def test_extract_people_and_dates() -> None:
    text = "".join(u.text for u in parse_transcript(SAMPLE))
    people = extract_people(text)
    assert any("宮永薬局" in name for name in people)
    assert any("平岡" in name for name in people)
    assert any("モイネット" in name for name in people)
    dates = extract_dates(text)
    assert any("来年1月" in value for value in dates)
    assert any("12月" in value for value in dates)


def test_extract_facts_finds_request_answer_pending_action() -> None:
    facts = extract_facts(SAMPLE)
    assert "宮永薬局" in facts.title
    assert facts.requests
    assert any("デモ機" in fact.text for fact in facts.requests)
    assert any("問題ない" in fact.text for fact in facts.answers)
    assert any("折り返し" in fact.text for fact in facts.pending)
    assert facts.actions
    assert any("折り返し" in action.what for action in facts.actions)
    assert "宮永薬局" in facts.speaker_labels[1]
    assert "モイネット" in facts.speaker_labels[0]


def test_summarize_text_without_llm_renders_markdown() -> None:
    result = summarize_text(
        SAMPLE,
        SummaryOptions(use_llm="never"),
        source=Path("sample.log"),
    )
    assert result.method == "rule"
    markdown = result.to_markdown()
    assert markdown.startswith("# ")
    assert "## 概要" in markdown
    assert "## 要望・依頼" in markdown
    assert "## 次のアクション" in markdown
    assert "sample.log" in markdown
    assert "文字起こしの誤認識" in markdown


def test_summarize_text_does_not_invent_dates() -> None:
    result = summarize_text(SAMPLE, SummaryOptions(use_llm="never"))
    joined = "\n".join([result.overview, *result.requests, *result.answers, *result.dates])
    assert "2027年" not in joined
    assert "3月" not in joined


def test_merge_llm_json_prefers_polished_items() -> None:
    facts = extract_facts(SAMPLE)
    data = {
        "title": "デモ機へのデータ移行",
        "overview": "宮永薬局がデモ機へのデータ移行を確認した。",
        "requests": [
            "デモ機に患者データを入れられるか確認したい。",
            "子機からも参照できるか確認したい。",
        ],
        "answers": [
            "データのファイルを写すのは問題ない。",
            "親機に入れれば子機からも参照できる。",
        ],
        "decisions": [],
        "pending": ["通常と異なるため、担当者に確認して折り返す。"],
        "actions": [],
        "dates": [],
        "people": [],
    }
    merged, warnings = _merge_llm_json(facts, data, "qwen3b")
    assert warnings == []
    assert merged.method == "rule+llm"
    assert merged.requests[0] == "デモ機に患者データを入れられるか確認したい。"
    assert all("デモ機に患者データを入れたい" not in item for item in merged.requests)
    assert any("折り返" in item for item in merged.pending)
    assert any("宮永薬局" in fact for fact in [merged.title])


def test_merge_llm_json_rejects_invented_names_and_dates() -> None:
    facts = extract_facts(SAMPLE)
    data = {
        "title": "データ移行の確認",
        "overview": "宮永薬局がデモ機へのデータ移行を確認した。",
        "requests": [
            "宮永薬局がモアイネットシステムに、デモ機への移行を確認したい。",
            "2027年3月までにデータを移したい。",
        ],
        "answers": ["平島様に確認して折り返す。"],
        "decisions": [],
        "pending": [],
        "actions": [{"who": "モイネットシステム", "what": "平島様へ連络する", "when": ""}],
        "dates": ["2027年3月"],
        "people": [],
    }
    merged, _warnings = _merge_llm_json(facts, data, "qwen3b")
    joined = "\n".join([*merged.requests, *merged.answers, *merged.pending, *merged.dates])
    assert "2027年3月" not in joined
    # 誤字は既知の名前に戻すか、裏付けが無ければ落とす
    assert "平島様" not in joined
    assert "モアイネットシステム" not in joined
    # アクションはルールベースを正とする
    assert any("折り返し" in action.what for action in merged.actions)
    assert all("平島様" not in action.what for action in merged.actions)


def test_repair_names_fixes_one_character_typos() -> None:
    assert (
        _repair_names("平島様へ折り返す", ["平岡様", "宮永薬局", "モイネットシステム"])
        == "平岡様へ折り返す"
    )
    assert (
        _repair_names("モアイネットシステムに確認", ["平岡様", "モイネットシステム"])
        == "モイネットシステムに確認"
    )

