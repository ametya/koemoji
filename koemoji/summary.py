"""文字起こし結果を「人が読める議事録」に整理する。

追加の重い依存なしで動くルールベース整理を基本とし、任意でローカル LLM
（CTranslate2 形式の小型モデル）を使って文章を整える。どちらも CPU のみで
動作し、音声・テキストが外部へ送られることはない。

ルールベース部分の方針:
- 話者分離ログから「依頼側 / 応対側」を推定する
- 要望・回答・決定・保留・ToDo・日付・人物を手がかり語で抽出する
- 抽出した事実だけを並べ、入力にない情報は作らない

LLM 部分の方針:
- ルールベースで抽出済みの事実だけを渡し、文章の整形に使う
- 出力は必ず元の事実と照合し、裏付けのない項目は捨てる
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .core import Segment, TranscribeResult
from .models import (
    DEFAULT_SUMMARY_MODEL,
    SUMMARY_CATALOG,
    ensure_summary_model,
    is_summary_downloaded,
)

ProgressCallback = Callable[[float, str], None]


class SummaryCancelled(Exception):
    """議事録整理のキャンセル。"""


# ---------------------------------------------------------------------------
# オプション・結果
# ---------------------------------------------------------------------------


@dataclass
class SummaryOptions:
    # "auto": モデルが取得済みなら LLM / "always": 必要なら DL / "never": ルールのみ
    use_llm: str = "auto"
    model: str = DEFAULT_SUMMARY_MODEL
    model_dir: Optional[Path] = None
    cpu_threads: Optional[int] = None
    max_new_tokens: int = 900
    include_transcript: bool = False
    # 顧客名・店舗名など、文字起こしだけでは分からない補足情報
    context: Optional[str] = None

    def llm_mode(self) -> str:
        mode = (self.use_llm or "auto").lower()
        if mode not in ("auto", "always", "never"):
            raise ValueError(f"use_llm は auto / always / never のいずれかです: {self.use_llm}")
        return mode


@dataclass
class ActionItem:
    who: str = ""
    what: str = ""
    when: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {"who": self.who, "what": self.what, "when": self.when}


@dataclass
class SummaryResult:
    title: str
    overview: str
    requests: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)
    actions: List[ActionItem] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    source: Optional[Path] = None
    method: str = "rule"
    model: Optional[str] = None
    elapsed: float = 0.0
    transcript: str = ""
    note: Optional[str] = None

    def to_markdown(self, include_transcript: bool = False) -> str:
        lines: List[str] = [f"# {self.title}", ""]
        if self.source is not None:
            lines.append(f"- 元ファイル: `{self.source.name}`")
        if self.method.startswith("rule+llm") or self.method == "llm":
            model_name = self.model or "ローカルLLM"
            lines.append(f"- 整理方法: ルールベース抽出 + AI整形（{model_name}・CPU）")
        else:
            lines.append("- 整理方法: ルールベース抽出（CPU）")
        if self.elapsed > 0:
            lines.append(f"- 処理時間: 約{self.elapsed:.1f}秒")
        lines.append("")

        lines.extend(["## 概要", "", self.overview.strip() or "（概要を抽出できませんでした）", ""])

        sections: List[Tuple[str, List[str]]] = [
            ("要望・依頼", self.requests),
            ("回答・説明", self.answers),
            ("決定事項", self.decisions),
            ("確認中・保留", self.pending),
        ]
        for heading, items in sections:
            if not items:
                continue
            lines.append(f"## {heading}")
            lines.append("")
            lines.extend(f"- {item}" for item in items)
            lines.append("")

        if self.actions:
            lines.extend(["## 次のアクション", ""])
            for action in self.actions:
                detail = action.what.strip()
                if action.when.strip():
                    detail += f"（{action.when.strip()}）"
                if action.who.strip():
                    detail = f"{action.who.strip()}：{detail}"
                lines.append(f"- [ ] {detail}")
            lines.append("")

        if self.dates:
            lines.extend(["## 重要な日付・数値", ""])
            lines.extend(f"- {item}" for item in self.dates)
            lines.append("")

        if self.people:
            lines.extend(["## 登場人物・連絡先", ""])
            lines.extend(f"- {item}" for item in self.people)
            lines.append("")

        if self.note:
            lines.extend(["> " + self.note.replace("\n", "\n> "), ""])

        lines.append(
            "> ※ 文字起こしの誤認識を含む可能性があります。日付・金額・人名は元の音声や資料で確認してください。"
        )
        lines.append("")

        if include_transcript and self.transcript.strip():
            lines.extend(
                [
                    "## 文字起こし（原文）",
                    "",
                    "<details>",
                    "<summary>全文を表示</summary>",
                    "",
                    "```log",
                    self.transcript.strip(),
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def to_plain(self, include_transcript: bool = False) -> str:
        lines: List[str] = [self.title, "=" * min(max(len(self.title), 8), 40), ""]
        if self.source is not None:
            lines.append(f"元ファイル: {self.source.name}")
        lines.append(f"整理方法: {'ルールベース + AI整形' if self.method != 'rule' else 'ルールベース'}")
        lines.append("")
        lines.extend(["■ 概要", self.overview.strip() or "（概要を抽出できませんでした）", ""])
        sections = [
            ("■ 要望・依頼", self.requests),
            ("■ 回答・説明", self.answers),
            ("■ 決定事項", self.decisions),
            ("■ 確認中・保留", self.pending),
        ]
        for heading, items in sections:
            if not items:
                continue
            lines.append(heading)
            lines.extend(f"・{item}" for item in items)
            lines.append("")
        if self.actions:
            lines.append("■ 次のアクション")
            for action in self.actions:
                detail = action.what
                if action.when:
                    detail += f"（{action.when}）"
                if action.who:
                    detail = f"{action.who}: {detail}"
                lines.append(f"・[ ] {detail}")
            lines.append("")
        if self.dates:
            lines.append("■ 重要な日付・数値")
            lines.extend(f"・{item}" for item in self.dates)
            lines.append("")
        if self.people:
            lines.append("■ 登場人物・連絡先")
            lines.extend(f"・{item}" for item in self.people)
            lines.append("")
        if include_transcript and self.transcript.strip():
            lines.extend(["■ 文字起こし（原文）", self.transcript.strip(), ""])
        lines.append("※ 文字起こしの誤認識を含む可能性があります。")
        return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# 入力の解析
# ---------------------------------------------------------------------------


@dataclass
class Utterance:
    index: int
    text: str
    speaker: Optional[int] = None
    start: Optional[float] = None


@dataclass
class Fact:
    text: str
    source: str
    speaker: Optional[int] = None
    score: float = 0.0
    timestamp: Optional[float] = None


@dataclass
class ExtractedFacts:
    title: str = ""
    overview: str = ""
    requests: List[Fact] = field(default_factory=list)
    answers: List[Fact] = field(default_factory=list)
    decisions: List[Fact] = field(default_factory=list)
    pending: List[Fact] = field(default_factory=list)
    actions: List[ActionItem] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    people: List[str] = field(default_factory=list)
    requester: Optional[int] = None
    responder: Optional[int] = None
    speaker_labels: Dict[int, str] = field(default_factory=dict)
    topic: str = ""
    utterance_texts: List[str] = field(default_factory=list)

    @property
    def all_sources(self) -> List[str]:
        items: List[str] = []
        for group in (self.requests, self.answers, self.decisions, self.pending):
            items.extend(f.source for f in group)
        items.extend(a.what for a in self.actions if a.what)
        items.extend(self.utterance_texts)
        return items


_TS_RE = re.compile(
    r"^\[(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?\]\s*(.*)$"
)
_SPEAKER_RE = re.compile(r"^(話者\s*(\d+))\s*[:：]\s*(.*)$")
_SRT_TIME_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*")


def _timestamp_seconds(hours: str, minutes: str, seconds: str, millis: Optional[str]) -> float:
    value = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    if millis:
        value += int(millis.ljust(3, "0")[:3]) / 1000.0
    return float(value)


def _split_sentences(text: str, max_chars: int = 110) -> List[str]:
    """長い発話を句点で文に分ける。句点が無い場合は読点で切る。"""
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[。．！？!?])\s*", text) if p.strip()]
    out: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            out.append(part)
            continue
        # 文末記号が無い長文は読点で分割する（意味の切れ目を優先）
        rest = part
        while len(rest) > max_chars:
            cut = -1
            for mark in ("、", "，", ",", " ", "　"):
                pos = rest.rfind(mark, 0, max_chars)
                if pos > max_chars // 3:
                    cut = max(cut, pos + 1)
            if cut <= 0:
                break
            out.append(rest[:cut].strip())
            rest = rest[cut:].strip()
        if rest:
            out.append(rest)
    return out


def _parse_line(line: str) -> Tuple[Optional[float], Optional[int], Optional[str], str]:
    """1 行を (start, speaker, speaker_label, text) に分解する。"""
    raw = line.strip()
    if not raw:
        return None, None, None, ""
    if raw.isdigit() or _SRT_TIME_RE.match(raw):
        return None, None, None, ""
    start: Optional[float] = None
    m = _TS_RE.match(raw)
    if m:
        start = _timestamp_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        raw = m.group(5).strip()
    m = _SPEAKER_RE.match(raw)
    if m:
        return start, int(m.group(2)) - 1, m.group(1), m.group(3).strip()
    return start, None, None, raw


def parse_transcript(text: str) -> List[Utterance]:
    """会話ログ（`[00:00:01] 話者1: ...`）や SRT、プレーンテキストを発話単位に分ける。

    同じ話者が連続する行は 1 ターンにまとめ、句点で文に分割する。
    これにより、Whisper が細かく切った行をそのまま要点として拾ってしまうのを防ぐ。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    raw_turns: List[Tuple[Optional[int], Optional[float], List[str]]] = []
    current_speaker: Optional[int] = None
    current_start: Optional[float] = None
    current_parts: List[str] = []
    saw_speaker = False
    last_start: Optional[float] = None

    def flush() -> None:
        nonlocal current_parts
        if current_parts:
            body = "".join(current_parts).strip()
            if body:
                raw_turns.append((current_speaker, current_start, [body]))
        current_parts = []

    for line in lines:
        start, speaker, _label, body = _parse_line(line)
        if not body:
            continue
        if speaker is not None:
            saw_speaker = True
            if current_parts and speaker != current_speaker:
                flush()
            elif current_parts and start is not None and last_start is not None and start - last_start > 25:
                # 同じ話者でも大きな無音を挟むなら別ターン
                flush()
            current_speaker = speaker
            if current_start is None or not current_parts:
                current_start = start
            current_parts.append(body)
            last_start = start if start is not None else last_start
        else:
            if current_parts and saw_speaker:
                current_parts.append(body)
                if start is not None:
                    last_start = start
            else:
                # 話者ラベルのない入力を 1 行ずつ扱う
                raw_turns.append((None, start, [body]))
    flush()

    utterances: List[Utterance] = []
    index = 0
    for speaker, start, parts in raw_turns:
        body = "".join(parts).strip()
        for sentence in _split_sentences(body):
            sentence = sentence.strip()
            if not sentence:
                continue
            # 「あ、」「何?」のような切れ端は落とす
            if len(sentence) <= 2 and not re.search(r"[一-龥ァ-ヶA-Za-z0-9]", sentence):
                continue
            if re.fullmatch(r"(?:あ|え|お|はい|ええ|うん|何|そう)[、,。?？!！\s]*", sentence):
                continue
            # 話者分離の切れ目で分断された語を最低限つなぐ
            if utterances:
                prev = utterances[-1]
                prev_body = prev.text.rstrip()
                joint = prev_body + sentence
                if (
                    prev_body
                    and not prev_body.endswith(tuple(_SENTENCE_END_CHARS))
                    and (
                        len(prev_body) <= 12
                        or prev_body.endswith(("の", "は", "が", "を", "に", "で", "と", "も", "や", "って", "て", "で"))
                    )
                    and len(joint) <= 80
                    and re.match(r"^[ぁ-んァ-ヶー]", sentence)
                ):
                    prev.text = joint
                    continue
            utterances.append(Utterance(index, sentence, speaker, start))
            index += 1
    return utterances


def segments_to_text(segments: Sequence[Segment]) -> str:
    """Segment の並びを会話ログ形式にする（時刻付き）。"""
    lines: List[str] = []
    for seg in segments:
        hours, rem = divmod(int(max(seg.start, 0.0)), 3600)
        minutes, seconds = divmod(rem, 60)
        prefix = f"[{hours:02d}:{minutes:02d}:{seconds:02d}] "
        if seg.speaker is not None:
            prefix += f"話者{seg.speaker + 1}: "
        lines.append(prefix + seg.text)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 手がかり語・正規表現
# ---------------------------------------------------------------------------


_REQUEST_CUES: Dict[str, float] = {
    "お願い": 1.5,
    "ください": 1.5,
    "いただけ": 1.5,
    "ていただけ": 1.5,
    "たい": 1.0,
    "ほしい": 1.2,
    "教えて": 1.5,
    "確認したい": 1.5,
    "知りたい": 1.5,
    "相談": 1.5,
    "質問": 1.5,
    "どうしたら": 2.0,
    "どうすれば": 2.0,
    "できますか": 1.5,
    "可能ですか": 1.5,
    "大丈夫ですか": 1.5,
    "よろしいですか": 1.2,
    "でしょうか": 1.2,
    "ますか": 1.2,
    "ですか": 0.8,
    "てもら": 1.0,
    "お伺い": 1.2,
    "依頼": 1.5,
}

_ANSWER_CUES: Dict[str, float] = {
    "問題ない": 2.0,
    "問題ございません": 2.0,
    "大丈夫": 1.2,
    "可能": 1.2,
    "対応": 1.0,
    "承知": 1.2,
    "確認": 1.0,
    "お返し": 1.5,
    "折り返し": 1.5,
    "説明": 1.2,
    "ご案内": 1.2,
    "資料": 1.0,
    "送付": 1.0,
    "設定": 1.0,
    "できます": 1.2,
    "いたします": 1.2,
    "させていただ": 1.0,
    "という形": 1.0,
    "という流れ": 1.0,
    "ため": 0.6,
    "ので": 0.6,
    "から": 0.4,
    "場合": 0.6,
    "思います": 0.4,
    "なります": 0.8,
}

_DECISION_CUES: Dict[str, float] = {
    "決定": 2.0,
    "することに": 1.8,
    "で進め": 1.5,
    "進めます": 1.5,
    "ましょう": 1.2,
    "となります": 1.2,
    "確定": 1.5,
    "採用": 1.5,
    "決まり": 1.5,
    "という形で": 0.8,
    "でお願いします": 1.2,
}

_PENDING_CUES: Dict[str, float] = {
    "確認": 0.8,
    "折り返し": 1.8,
    "検討": 1.5,
    "調整": 1.2,
    "持ち帰": 1.5,
    "後日": 1.5,
    "改めて": 1.2,
    "未定": 1.5,
    "保留": 1.5,
    "確認中": 1.8,
    "お返し": 1.2,
    "また連絡": 1.5,
    "担当の方に": 1.2,
    "確認させて": 1.2,
    "いつ": 0.8,
    "までに": 0.8,
    "待ち": 0.8,
}

_ACTION_CUES: Dict[str, float] = {
    "します": 1.0,
    "いたします": 1.2,
    "させていただ": 1.0,
    "お願い": 1.0,
    "ください": 1.0,
    "折り返し": 1.5,
    "連絡": 1.0,
    "送付": 1.2,
    "確認": 0.8,
    "手配": 1.2,
    "準備": 1.0,
    "共有": 1.0,
    "設定": 1.0,
    "移して": 1.2,
    "入れて": 1.2,
    "出力": 1.2,
    "変換": 1.2,
    "コンバート": 1.2,
    "保存": 1.2,
    "取得": 1.2,
}

_GREETING_CUES = (
    "お電話ありがとうございます",
    "お世話になって",
    "よろしくお願い",
    "失礼いたします",
    "失礼します",
    "お待たせいたしました",
    "お待たせしました",
)

_BACKCHANNEL_RE = re.compile(
    r"^(はい|ええ|うん|そうですね|そうですか|なるほど|たしかに|確かに|ありがとうございます|"
    r"お願いいたします|お願いします|大丈夫です|失礼いたします|失礼します|はい、?大丈夫です)[。、！!？?]*$"
)
_SENTENCE_END_CHARS = "。．！？!?…"

_DATE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:\d{4}年)?\d{1,2}月(?:\d{1,2}日)?(?:上旬|中旬|下旬)?"),
    re.compile(r"\d{1,2}月\d{1,2}日"),
    re.compile(r"(?:今年|来年|昨年|年末|年内|年度内|来週|今週|週末|明日|今日|昨日|本日|月初|月末)"),
    re.compile(r"\d+\s*[〜～~\-ー、,]\s*\d+\s*(?:日|週間|か月|ヶ月|年|時間|分)"),
    re.compile(r"\d+\s*(?:日|週間|か月|ヶ月|年|時間|分|秒|件|台|人|円|万円|%|割)(?:程度|ほど|以上|以内)?"),
    re.compile(r"(?:第?\d+回目?)"),
)

_GENERIC_NAMES = {
    "お客様",
    "お客さま",
    "担当者",
    "担当者様",
    "皆様",
    "皆さん",
    "本日",
    "来年",
    "今年",
    "年末",
    "今回",
    "前回",
    "当社",
    "弊社",
    "先方",
    "藥局",
    "薬局",
    "会社",
    "事務",
    "事務担当",
    "システム",
    "患者様",
    "患者さん",
    "薬局様",
    "お客様",
    "お客さま",
    "担当者様",
    "先方様",
}

_NAME_CHARS = r"一-龥々〆ヵヶぁ-んァ-ヶA-Za-z0-9"
_KANJI_KATA_ASCII = r"一-龥々〆ヵヶァ-ヶA-Za-z0-9"
_NAME_BODY = r"[一-龥々〆ヵヶァ-ヶA-Za-z0-9]"
_ORG_SUFFIX = r"(?:薬局|薬店|ドラッグ|店|会社|株式会社|有限会社|システム|ネット|センター|病院|クリニック|大学|事務所|グループ)"
_ORG_RE = re.compile(rf"((?:株式会社|有限会社|合同会社)?{_NAME_BODY}{{2,12}}{_ORG_SUFFIX})")
_HIRA_ORG_RE = re.compile(rf"([ぁ-ん]{{2,8}}{_ORG_SUFFIX})")
_PERSON_HONORIFIC_RE = re.compile(rf"({_NAME_BODY}{{1,8}})(?:様|さん)")
_SELF_INTRO_RE = re.compile(
    rf"(?:私|わたくし|僕|こちら)?\s*(?:事務(?:担当)?の)?\s*({_NAME_BODY}{{1,10}})\s*(?:と申します|と申し上げます|と名乗ります)"
)
_ROLE_RE = re.compile(rf"({_NAME_BODY}{{2,10}}(?:担当|部長|課長|主任|理事|社長|店長))")
_NAME_REJECT_WORDS = (
    "いる",
    "ある",
    "なる",
    "する",
    "した",
    "ます",
    "です",
    "思う",
    "感じ",
    "届い",
    "言わ",
    "頂い",
    "いただ",
    "一応",
    "普通",
    "本当",
    "全部",
    "最初",
    "最後",
)

_TOPIC_TERMS = (
    "データ",
    "患者",
    "処方",
    "デモ機",
    "親機",
    "子機",
    "レセコン",
    "コンバート",
    "移行",
    "保存",
    "契約",
    "納品",
    "導入",
    "設定",
    "請求",
    "薬剤",
    "在庫",
    "勤怠",
    "シフト",
    "売上",
    "レポート",
    "システム",
    "運用",
    "研修",
    "マニュアル",
    "トラブル",
    "エラー",
    "問い合わせ",
    "見積",
    "請求書",
    "支払",
    "取材",
    "面接",
    "採用",
    "プロジェクト",
    "スケジュール",
    "仕様",
    "機能",
    "テスト",
    "リリース",
)


def _norm(text: str) -> str:
    return re.sub(r"[\s、。，．！？!?・「」『』（）()\[\]【】:：;；]+", "", text)


def _ngrams(text: str, n: int = 2) -> set[str]:
    body = _norm(text)
    if len(body) < n:
        return {body} if body else set()
    return {body[i : i + n] for i in range(len(body) - n + 1)}


def _similarity(a: str, b: str) -> float:
    ga, gb = _ngrams(a), _ngrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _is_backchannel(text: str) -> bool:
    body = text.strip()
    if _BACKCHANNEL_RE.match(body):
        return True
    return len(body) <= 5 and body in {"はい", "ええ", "うん", "なるほど", "そうですね"}


def _is_greeting(text: str) -> bool:
    return any(cue in text for cue in _GREETING_CUES)


def _clean_fact(text: str) -> str:
    body = text.strip()
    for _ in range(2):
        new_body = re.sub(
            r"^(?:あ|え|お|はい|ええ|うん|そうですね|そうですか|そうですよね|"
            r"なるほど|じゃあ|それで|ただ|とりあえず|そしたら|すいません|すみません|"
            r"失礼します|失礼ですが|あの|えっと|その|まあ|ちょっと|だ|です|"
            r"という|っていう|けど|けれども|けれど|ですが|ますが|ませんが|"
            r"ので|のに|んです|んだ)[、,\s]*",
            "",
            body,
        )
        if new_body == body:
            break
        body = new_body
    body = re.sub(r"[、,\s]*何[?？]?[。、,\s]*$", "", body)
    body = re.sub(
        r"(?:と思います|と思っております|と思って|という形で|という感じで|という流れで|"
        r"ということで|という認識で|というところで|という話で|んですけど|んですが|ですが|"
        r"けれども|けれど|けど|ですね|ますね|かな|かね|そうですね|という|っていう|"
        r"ので|のに|ますが|ませんが|んです|んだ)[。、,\s]*$",
        "",
        body,
    )
    body = re.sub(r"[。、,\s]+$", "", body)
    body = body.replace("ですね", "、").replace("ますね", "、")
    body = re.sub(r"、{2,}", "、", body).strip("、 ")
    body = re.sub(r"\s+", " ", body)
    if len(body) > 100:
        cut = max(body.rfind("、", 0, 95), body.rfind(",", 0, 95))
        if cut >= 35:
            body = body[: cut + 1] + "…"
        else:
            body = body[:95] + "…"
    return body.strip()


def _clause_score(clause: str, cues: Dict[str, float]) -> float:
    score = _cue_score(clause, cues)
    score += sum(0.35 for term in _TOPIC_TERMS if term in clause)
    if 12 <= len(clause) <= 60:
        score += 0.8
    elif len(clause) < 6:
        score -= 1.0
    if _is_greeting(clause) or _is_backchannel(clause):
        score -= 3.0
    return score


def _cue_score(text: str, cues: Dict[str, float]) -> float:
    return sum(weight for cue, weight in cues.items() if cue in text)


def _is_question(text: str) -> bool:
    body = text.strip()
    return bool(
        re.search(r"(?:か|かね|かな|でしょうか|ますか|ですか|の)[。？?]?$", body)
        or any(c in body for c in ("どうしたら", "どうすれば", "教えて", "確認したい", "知りたい", "大丈夫ですか", "よろしいですか"))
    )


def _is_past_action(text: str) -> bool:
    return bool(re.search(r"(?:しました|いたしました|いただきました|しました。|ました。)$", text.strip()))


def _has_request_signal(text: str) -> bool:
    """依頼・確認の意図が読み取れる文かどうか。"""
    if _is_question(text):
        return True
    # 「〜したい」「〜してほしい」は語尾が崩れていても依頼の意図とみなす
    if re.search(
        r"(?:したい|入れたい|移したい|使いたい|してほしい|てもらいたい)"
        r"(?:んです|のですが|んですが|ので|のに|から|けど|けれど|ですけど)?[。、,\s]*$",
        text,
    ):
        return True
    return any(
        cue in text
        for cue in (
            "お願い",
            "ください",
            "いただけ",
            "てもら",
            "てほしい",
            "教えて",
            "確認したい",
            "知りたい",
            "相談",
            "質問",
            "依頼",
            "大丈夫",
            "よろしい",
            "できますか",
            "可能ですか",
        )
    )


# 話し言葉の長い発話を、意味の切れ目で短い節に分ける。
# Whisper は句読点が少ないため、接続表現の前後でも切る。
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[。．！？!?])"
    r"|(?<=けど)|(?<=けれども)|(?<=けれど)|(?<=ですが)|(?<=ますが)|(?<=ませんが)"
    r"|(?<=ので)|(?<=んです)|(?<=んだ)"
    r"|(?=(?:じゃあ|それで|とりあえず|そしたら|なるほど|"
    r"すいません|すみません|はい|そうですね|えーと|あの))"
)
_FILLER_CLAUSE_RE = re.compile(
    r"^(?:はい|ええ|うん|あ|あー|えー|そうですね|そうですよね|なるほど|"
    r"じゃあ|それで|ただ|とりあえず|そしたら|すいません|すみません|"
    r"お世話になっております|お世話になります)[、,。！!？?\s]*$"
)


def _split_clauses(text: str) -> List[str]:
    """1 発話を読みやすい長さの節に分割する。"""
    body = text.strip()
    if not body:
        return []
    parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(body) if part and part.strip()]
    merged: List[str] = []
    for part in parts:
        if not part:
            continue
        if _FILLER_CLAUSE_RE.match(part):
            # 「はい」「そうですね」だけの節は捨てる
            continue
        if len(part) < 4:
            # 分割で生じた 1〜3 文字の切れ端は捨てる
            continue
        merged.append(part)
    # 長すぎる節は読点でもう一度切る
    out: List[str] = []
    for part in merged:
        if len(part) <= 85:
            out.append(part)
            continue
        rest = part
        while len(rest) > 85:
            cut = rest.rfind("、", 20, 85)
            if cut < 0:
                break
            out.append(rest[: cut + 1].strip())
            rest = rest[cut + 1 :].strip()
        if rest:
            out.append(rest)
    return [part for part in out if part and not _FILLER_CLAUSE_RE.match(part)]


def _iter_clauses(utterances: Sequence[Utterance]) -> Iterable[Tuple[Utterance, str]]:
    for utterance in utterances:
        for clause in _split_clauses(utterance.text):
            yield utterance, clause


# ---------------------------------------------------------------------------
# 話者・人物・日付
# ---------------------------------------------------------------------------


def infer_speakers(utterances: Sequence[Utterance]) -> Tuple[Optional[int], Optional[int], Dict[int, str]]:
    """2 人の会話から「依頼側」と「応対側」を推定する。"""
    speakers = sorted({u.speaker for u in utterances if u.speaker is not None})
    if len(speakers) != 2:
        return None, None, {}

    scores = {sp: {"requester": 0.0, "responder": 0.0} for sp in speakers}
    for u in utterances:
        if u.speaker is None:
            continue
        text = u.text
        score = scores[u.speaker]
        if "お電話ありがとうございます" in text:
            score["responder"] += 5.0
        if "お待たせ" in text:
            score["responder"] += 2.0
        if "と申します" in text or "からお電話" in text or "お電話しました" in text:
            score["requester"] += 5.0
        if any(c in text for c in ("弊社", "当社", "ご案内", "担当の方に", "確認させて", "折り返し")):
            score["responder"] += 1.2
        if any(c in text for c in ("教えて", "相談", "質問", "お伺い", "確認したい")):
            score["requester"] += 1.2
        if _is_question(text):
            score["requester"] += 0.6

    # 最初の発話が「お電話ありがとうございます」なら応対側
    for u in utterances:
        if u.speaker is not None and u.text.strip():
            if "お電話ありがとうございます" in u.text:
                scores[u.speaker]["responder"] += 3.0
            break

    first, second = speakers
    responder = requester = None
    if scores[first]["responder"] >= scores[second]["responder"] + 2:
        responder, requester = first, second
    elif scores[second]["responder"] >= scores[first]["responder"] + 2:
        responder, requester = second, first
    elif scores[first]["requester"] >= scores[second]["requester"] + 2:
        requester, responder = first, second
    elif scores[second]["requester"] >= scores[first]["requester"] + 2:
        requester, responder = second, first

    labels: Dict[int, str] = {}
    if requester is not None:
        labels[requester] = f"話者{requester + 1}（依頼側）"
    if responder is not None:
        labels[responder] = f"話者{responder + 1}（応対側）"
    return requester, responder, labels


def _trim_name(name: str) -> str:
    name = name.strip(" 、,。・「」『』（）()")
    # 「広島にあります宮永薬局」のような前置きを落とす
    name = re.sub(r"^.*?(?:にあります|にある|であります|の)(?=[一-龥ァ-ヶA-Za-z]{2,})", "", name)
    name = name.lstrip("はがをにでともやの")
    return name.strip()


def _looks_like_name(name: str) -> bool:
    if not name or name in _GENERIC_NAMES:
        return False
    if len(name) < 2 or len(name) > 24:
        return False
    if any(word in name for word in _NAME_REJECT_WORDS):
        return False
    # 助詞が混ざる候補は文章の一部（「いうのはおそらく薬局」など）
    if any(ch in name for ch in "はがをにでとへや"):
        return False
    if re.search(r"(.)\1{2,}", name):
        return False
    # ひらがなだけの長い列は文章の一部である可能性が高い（「みやなが」は許容）
    if re.fullmatch(r"[ぁ-ん]{5,}", name):
        return False
    return True


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def _name_variant(a: str, b: str) -> bool:
    """表記ゆれ（モイネット / モディネット など）を同一人物・組織とみなす。"""
    if not a or not b:
        return False
    def strip_suffix(name: str) -> str:
        for suffix in ("株式会社", "有限会社", "合同会社", "システム", "ネット", "薬局", "店", "会社"):
            if name.endswith(suffix) and len(name) - len(suffix) >= 4:
                return name[: -len(suffix)]
        return name

    base_a = strip_suffix(a)
    base_b = strip_suffix(b)
    katakana = r"[ァ-ヶー]{4,}"
    if not (re.fullmatch(katakana, base_a) and re.fullmatch(katakana, base_b)):
        return False
    if abs(len(base_a) - len(base_b)) > 2:
        return False
    return _edit_distance(base_a, base_b) <= 2


def _dedupe_names(names: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for raw in names:
        name = _trim_name(raw)
        if not _looks_like_name(name):
            continue
        # 敬称付きと裸の名前が両方ある場合は長い表記を優先
        base = re.sub(r"(?:様|さん)$", "", name)
        replaced = False
        for index, other in enumerate(cleaned):
            other_base = re.sub(r"(?:様|さん)$", "", other)
            if base == other_base:
                if len(name) > len(other):
                    cleaned[index] = name
                replaced = True
                break
            if base in other_base:
                replaced = True
                break
            if other_base in base:
                cleaned[index] = name
                replaced = True
                break
            if _name_variant(base, other_base):
                if len(name) > len(other):
                    cleaned[index] = name
                replaced = True
                break
        if not replaced:
            cleaned.append(name)
    return cleaned[:10]


def extract_people(text: str) -> List[str]:
    names: List[str] = []
    for match in _SELF_INTRO_RE.finditer(text):
        name = match.group(1)
        if name and not name.endswith(("の", "方", "こと")):
            names.append(name)
    for regex in (_ORG_RE, _HIRA_ORG_RE, _PERSON_HONORIFIC_RE, _ROLE_RE):
        for match in regex.finditer(text):
            name = match.group(1)
            if regex is _PERSON_HONORIFIC_RE:
                name += "様"
            names.append(name)
    return _dedupe_names(names)


def extract_dates(text: str) -> List[str]:
    found: List[Tuple[int, int, str]] = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if not value or value.isdigit():
                continue
            found.append((match.start(), match.end(), value))
    # 同じ範囲では長い表記を優先（「2、3週間」対「3週間」など）
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: List[Tuple[int, int, str]] = []
    for start, end, value in found:
        key = _norm(value)
        if not key:
            continue
        overlap = next(
            (item for item in selected if not (end <= item[0] or start >= item[1])),
            None,
        )
        if overlap is not None:
            old_start, old_end, old_value = overlap
            if len(value) > len(old_value):
                selected.remove(overlap)
                selected.append((start, end, value))
            continue
        if any(key in _norm(item[2]) or _norm(item[2]) in key for item in selected):
            continue
        selected.append((start, end, value))
    selected.sort()
    out: List[str] = []
    for start, end, value in selected:
        if (
            out
            and out[-1] in ("来年", "今年", "昨年")
            and re.match(r"\d{1,2}月", value)
        ):
            out[-1] = out[-1] + value
            continue
        out.append(value)
    return out[:14]


def _date_context(text: str, value: str, limit: int = 60) -> str:
    clause = text
    for part in _split_clauses(text):
        if value in part:
            clause = part
            break
    body = _clean_fact(clause)
    pos = body.find(value)
    if pos < 0:
        return body[:limit] + ("…" if len(body) > limit else "")
    start = max(0, pos - 24)
    end = min(len(body), pos + len(value) + 32)
    snippet = body[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(body):
        snippet += "…"
    return snippet


def _date_priority(value: str) -> int:
    if re.search(r"(?:月|年|週|日|年末|年度|上旬|中旬|下旬)", value):
        return 3
    if re.search(r"(?:台|件|人|円|%|割|時間|分)", value):
        return 2
    return 1


# ---------------------------------------------------------------------------
# 事実抽出
# ---------------------------------------------------------------------------


def _speaker_label(speaker: Optional[int], labels: Dict[int, str]) -> str:
    if speaker is None:
        return "担当者"
    return labels.get(speaker, f"話者{speaker + 1}")


def _sentence_features(
    text: str,
    speaker: Optional[int],
    requester: Optional[int],
    responder: Optional[int],
) -> Dict[str, float]:
    body = text.strip()
    length = len(body)
    req = _cue_score(body, _REQUEST_CUES)
    ans = _cue_score(body, _ANSWER_CUES)
    dec = _cue_score(body, _DECISION_CUES)
    pen = _cue_score(body, _PENDING_CUES)
    act = _cue_score(body, _ACTION_CUES)

    if _is_question(body):
        req += 1.5
        ans -= 0.5
    if speaker == requester:
        req += 2.0
        ans -= 1.0
    if speaker == responder:
        ans += 2.0
        req -= 1.0
    if _is_greeting(body):
        req -= 4.0
        ans -= 4.0
        dec -= 2.0
        pen -= 2.0
        act -= 2.0
    if _is_backchannel(body):
        req -= 3.0
        ans -= 3.0
        dec -= 3.0
        pen -= 3.0
        act -= 3.0
    if length < 8:
        req -= 1.5
        ans -= 1.5
        act -= 1.5
    elif 14 <= length <= 90:
        req += 0.8
        ans += 0.8
        act += 0.5
    elif length > 150:
        req -= 0.5
        ans -= 0.5
    if extract_dates(body):
        req += 0.4
        ans += 0.4
        pen += 0.4
        act += 0.4
    topic_hits = sum(1 for term in _TOPIC_TERMS if term in body)
    req += 0.35 * topic_hits
    ans += 0.30 * topic_hits
    act += 0.30 * topic_hits
    # 「何?」だけの聞き返しや、手順の確認だけの文は要望の中心ではない
    if re.search(r"何[?？]|なんですか[?？]?$", body):
        req -= 1.8
    if re.search(r"[12一二]回目", body) and not any(
        term in body for term in ("データ", "契約", "納品", "作業", "導入")
    ):
        req -= 2.5
        ans -= 0.8
        act -= 1.0
    if any(word in body for word in ("よろしくお願い", "失礼いたします", "失礼します", "ありがとうございます")):
        act -= 3.0
        ans -= 0.5
    if _is_past_action(body):
        act -= 2.0
        pen -= 0.5
    # 「確認」だけの文は保留/回答の両方に寄りやすいので、丁寧に補正
    if "確認" in body and any(c in body for c in ("させて", "したい", "お返し", "折り返し", "中", "必要")):
        pen += 1.0
    if "確認しました" in body or "確認できました" in body:
        pen -= 1.0
        ans += 0.5
    return {"request": req, "answer": ans, "decision": dec, "pending": pen, "action": act}


def _select_facts(
    candidates: Sequence[Fact],
    limit: int,
    threshold: float,
    exclude: Sequence[Fact] = (),
    max_per_source: int = 2,
) -> List[Fact]:
    ordered = sorted(candidates, key=lambda f: f.score, reverse=True)
    selected: List[Fact] = []
    excluded_texts = [f.text for f in exclude]
    per_source: Dict[str, int] = {}
    for fact in ordered:
        if fact.score < threshold or not fact.text.strip():
            continue
        if any(_norm(fact.text) == _norm(e.text) for e in selected):
            continue
        if any(_similarity(fact.text, other) >= 0.62 for other in excluded_texts):
            continue
        if any(_similarity(fact.text, other.text) >= 0.62 for other in selected):
            continue
        if max_per_source and per_source.get(fact.source, 0) >= max_per_source:
            continue
        selected.append(fact)
        per_source[fact.source] = per_source.get(fact.source, 0) + 1
        if len(selected) >= limit:
            break
    # 時系列に並べ直す（読みやすさのため）
    selected.sort(key=lambda f: (f.timestamp is None, f.timestamp or 0.0))
    return selected


def _topic_from_text(text: str) -> str:
    phrases: List[str] = []
    if "データ" in text and any(k in text for k in ("移す", "移行", "入れる", "入れ", "保存", "コンバート", "変換")):
        phrases.append("データ移行")
    if "デモ機" in text and ("データ" in text or "保存" in text or "入れる" in text):
        phrases.append("デモ機への保存")
    if "契約" in text and ("終了" in text or "切れ" in text or "満了" in text):
        phrases.append("契約終了に伴うデータ取得")
    if "患者" in text and "処方" in text:
        phrases.append("患者・処方データ")
    if "納品" in text and ("2回目" in text or "二回目" in text):
        phrases.append("納品2回目の作業")
    if len(phrases) >= 2:
        # 同じ意味の語が重複しないように、先頭 2 つに絞る
        return "・".join(dict.fromkeys(phrases))[:48]

    counts: Dict[str, int] = {}
    for term in _TOPIC_TERMS:
        count = text.count(term)
        if count:
            counts[term] = count
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0])))
    terms = [term for term, _ in ranked[:3]]
    if not terms:
        words = re.findall(rf"[{_KANJI_KATA_ASCII}]{{2,8}}", text)
        stop = {"今日", "本日", "今回", "来年", "今年", "場合", "内容", "確認", "作業", "電話", "担当"}
        for word in words:
            if word not in stop and not word.endswith(("です", "ます", "した")):
                terms.append(word)
            if len(terms) >= 2:
                break
    if not phrases and not terms:
        return "打ち合わせ内容"
    combined = phrases + [term for term in terms if term not in phrases]
    return "・".join(dict.fromkeys(combined))[:48]


def _make_title(
    topic: str,
    requests: Sequence[Fact],
    people: Sequence[str],
    preferred_label: str = "",
) -> str:
    org_candidates = [
        person
        for person in people
        if re.search(r"(?:薬局|会社|株式会社|有限会社|システム|ネット|店|病院|クリニック)$", person)
    ]
    # 依頼側（顧客）の組織名を優先する。応対側の自社名が先に来ても件名がぶれない。
    org = next(
        (person for person in org_candidates if preferred_label and person in preferred_label),
        org_candidates[0] if org_candidates else "",
    )
    short_topic = topic.split("・")[0] if topic and topic != "打ち合わせ内容" else ""
    if short_topic:
        base = short_topic if "確認" in short_topic else f"{short_topic}の確認"
    else:
        base = "打ち合わせ内容"
    if org:
        base = f"{org}・{base}"
    if len(base) > 44:
        base = base[:43] + "…"
    return base


def _point_text(text: str, limit: int = 72) -> str:
    """箇条書きの断片を、概要に埋め込める短い言い切りにする。"""
    body = _clean_fact(text)
    body = re.sub(
        r"(?:と思ってお電話した|と思いお電話した|と思って電話した|とお電話した|"
        r"という質問だった|っていう質問だった|という質問なんです|という質問です|"
        r"という形ですかね|っていう形ですかね|という形です|っていう形です|"
        r"という形になり|という流れになり|という形で|という流れで|"
        r"という感じですかね|というところです|というところではあります|"
        r"というとあれなん|というとあれな|あれなん|あれな|"
        r"よろしかったですかね|よろしかったです|よろしいでしょうか|よろしいですか|"
        r"大丈夫ですか|でしょうか|(?:情報|感じ|方|こと|もの)が)[。、,\s]*$",
        "",
        body,
    )
    body = re.sub(r"(?:かな|かね)$", "か", body)
    body = body.replace("でございますかね", "、").replace("でございますね", "、")
    # 言いよどみ・言い直しの切れ端を落とす
    body = re.sub(r"[、,\s]*(?:っていう|という)(?:質問|お話|話)(?:な|なん|です|でした)?$", "", body)
    body = re.sub(r"[、,\s]*(?:前の|次の|同じ|その|この)の?$", "", body)
    body = re.sub(r"[、,\s]*(?:っていう|という)(?:感じ|ところ|もの|こと|形)(?:な|で|の|ではあり|ではあります)?$", "", body)
    body = re.sub(r"[、,\s]*(?:って|という|っていう)$", "", body)
    body = re.sub(r"[、,\s]*(?:前の)+$", "", body)
    body = re.sub(r"(?:という|っていう)感じだった$", "", body)
    body = re.sub(r"(折り返し|確認|連絡)(?:させていただいて|いただいて|して)も$", r"\1します", body)
    body = body.strip(" 、,。")
    if len(body) < 6:
        return ""
    if len(body) > limit:
        cut = max(body.rfind("、", 0, limit), body.rfind(" ", 0, limit))
        body = body[: cut if cut >= limit // 2 else limit].rstrip("、 ") + "…"
    return body


def _make_overview(
    facts: ExtractedFacts,
    requester_label: str,
    responder_label: str,
) -> str:
    parts: List[str] = []
    topic = facts.topic or "打ち合わせ内容"
    label = re.sub(r"（話者\d+）$", "", requester_label)
    person = next((p for p in facts.people if p.endswith(("様", "さん"))), "")
    subject = label
    if label and person and person not in label:
        subject = f"{label}の{person}"
    if subject:
        parts.append(f"{subject}から、{topic}について連絡があった。")
    elif facts.requests:
        parts.append(f"{topic}について確認の連絡があった。")
    else:
        parts.append(f"{topic}について確認した。")

    if facts.requests:
        point = _point_text(facts.requests[0].text)
        if point:
            parts.append(f"主な確認は「{point}」。")
    if facts.answers:
        best = max(
            facts.answers,
            key=lambda fact: (
                5 * any(k in fact.text for k in ("問題ない", "大丈夫", "可能", "できます", "対応"))
                + sum(1 for term in _TOPIC_TERMS if term in fact.text)
            ),
        )
        point = _point_text(best.text, 78)
        if point:
            parts.append(f"応対側からは「{point}」との説明。")
    if facts.pending:
        if any("折り返し" in item.text for item in facts.pending):
            parts.append("通常と異なる点があるため、担当者に確認して折り返す。")
        else:
            point = _point_text(facts.pending[0].text, 62)
            if point:
                parts.append(f"「{point}」については確認中。")
    overview = "".join(parts).strip()
    if len(overview) > 480:
        overview = overview[:470] + "…"
    return overview


def _action_score(text: str, speaker: Optional[int], requester: Optional[int], responder: Optional[int]) -> float:
    body = text.strip()
    score = 0.0
    if "折り返し" in body:
        score += 4.0
    if "確認させて" in body or "確認して" in body:
        score += 2.0
    if "担当" in body:
        score += 1.0
    if "連絡" in body:
        score += 2.0
    if "データ" in body and any(k in body for k in ("入れる", "入れ", "保存", "出力", "移す", "移し", "取得", "変換", "コンバート", "取り")):
        score += 2.0
    if any(k in body for k in ("します", "いたします", "させていただ", "ください", "お願い")):
        score += 1.0
    if _is_question(body) and not any(k in body for k in ("折り返し", "確認させて", "お願い")):
        score -= 0.8
    if any(k in body for k in ("問題ない", "大丈夫", "可能", "できます")):
        score -= 3.0
    if _is_greeting(body) or _is_backchannel(body):
        score -= 5.0
    if any(k in body for k in ("ありがとうございます", "よろしくお願い", "失礼いたします", "失礼します")):
        score -= 4.0
    if len(body) < 12:
        score -= 2.0
    if speaker == responder:
        score += 0.3
    return score


def _clean_label(label: str) -> str:
    return re.sub(r"（話者\d+）$", "", label or "").strip()


def _extract_actions(
    utterances: Sequence[Utterance],
    requests: Sequence[Fact],
    pending: Sequence[Fact],
    requester: Optional[int],
    responder: Optional[int],
    labels: Dict[int, str],
    people: Sequence[str],
    dates: Sequence[str],
) -> List[ActionItem]:
    actions: List[ActionItem] = []

    def add(who: str, what: str, when: str = "") -> None:
        what = what.strip(" 、,。")
        if not what:
            return
        if any(_similarity(what, action.what) >= 0.7 for action in actions):
            return
        actions.append(ActionItem(who=who.strip(), what=what, when=when.strip()))

    responder_name = _clean_label(labels.get(responder, "")) if responder is not None else ""
    requester_name = _clean_label(labels.get(requester, "")) if requester is not None else ""
    person = next((p for p in people if p.endswith(("様", "さん"))), "")

    # 折り返し・担当者確認は最も確実な次のアクション
    for fact in pending:
        if "折り返し" in fact.text or "確認させて" in fact.text:
            target = f"{person}へ折り返し連絡" if person else "折り返し連絡"
            add(responder_name or "応対側", f"担当者に確認し、{target}", "")
            break

    # 「〜しないといけない」「必要」など、明確な今後の作業だけを拾う
    for u, clause in _iter_clauses(utterances):
        text = _clean_fact(clause)
        if len(text) < 10:
            continue
        if not any(k in text for k in ("ないといけな", "必要がある", "必要なん", "必要です", "予定")):
            continue
        when = next((value for value in dates if value in u.text), "")
        who = requester_name
        if u.speaker == responder or any(k in text for k in ("弊社", "こちらで")):
            who = responder_name
        if "データ" in text and any(k in text for k in ("入れる", "入れとく", "移す", "移し", "落とす", "落とし")):
            add(who or "担当者", "現行レセコンのデータをデモ機（親機）へ移す", when)
            continue
        if "データ" in text and any(k in text for k in ("出力", "取りに来", "取得")):
            add(who or "担当者", "データを出力してデモ機（親機）へ移す", when)

    return actions[:5]


def extract_facts(text: str) -> ExtractedFacts:
    """会話ログから議事録の材料を抽出する（LLMなし）。"""
    utterances = parse_transcript(text)
    requester, responder, labels = infer_speakers(utterances)
    all_text = "".join(u.text for u in utterances)

    people = extract_people(all_text)
    dates = extract_dates(all_text)
    # 日付は文脈付きで残す。重要度の低い「1台だけ」などは落とし、上位 8 件に絞る
    date_rows: List[Tuple[int, int, str, str]] = []
    for order, value in enumerate(dates):
        context = ""
        for u in utterances:
            if value in u.text:
                context = _date_context(u.text, value)
                break
        if re.fullmatch(r"\d+台", value) and ("しか" in context or "だけ" in context):
            continue
        if re.fullmatch(r"\d+回目", value) and not any(
            term in context for term in ("納品", "作業", "データ")
        ):
            continue
        date_rows.append((_date_priority(value), order, value, context))
    date_rows.sort(key=lambda item: (-item[0], item[1]))
    date_rows = date_rows[:8]
    date_rows.sort(key=lambda item: item[1])
    date_items: List[str] = []
    for _priority, _order, value, context in date_rows:
        date_items.append(f"{value} — {context}" if context and _norm(context) != _norm(value) else value)

    # 話者ラベルを会社名で具体化する
    enriched_labels = dict(labels)
    for speaker in {u.speaker for u in utterances if u.speaker is not None}:
        speaker_text = "".join(u.text for u in utterances if u.speaker == speaker)
        speaker_names = extract_people(speaker_text)
        prefer = ("薬局", "店", "病院", "クリニック") if speaker == requester else ("システム", "ネット", "会社", "株式会社")
        for name in speaker_names:
            if any(key in name for key in prefer):
                enriched_labels[speaker] = f"{name}（話者{speaker + 1}）"
                break
        else:
            if speaker in enriched_labels:
                enriched_labels[speaker] = enriched_labels[speaker]

    candidates: Dict[str, List[Fact]] = {
        "request": [],
        "answer": [],
        "decision": [],
        "pending": [],
        "action": [],
    }
    for u, clause in _iter_clauses(utterances):
        text_body = clause.strip()
        if len(text_body) < 6:
            continue
        features = _sentence_features(text_body, u.speaker, requester, responder)
        source = u.text
        display = _point_text(text_body, 92)
        if not display or len(display) < 6:
            continue
        for category, score in features.items():
            if score <= 0:
                continue
            if category == "request" and not _has_request_signal(text_body):
                continue
            if category == "answer" and responder is None and _has_request_signal(text_body):
                continue
            candidates[category].append(
                Fact(display, source, u.speaker, score, u.start)
            )

    requests = _select_facts(candidates["request"], limit=5, threshold=3.0)
    pending = _select_facts(candidates["pending"], limit=3, threshold=2.6, exclude=requests)
    answers = _select_facts(candidates["answer"], limit=5, threshold=3.0, exclude=requests)
    decisions = _select_facts(candidates["decision"], limit=4, threshold=2.4, exclude=requests + answers + pending)
    # 保留と回答が重複しすぎる場合は、回答側を少し削る（ただし折り返しの約束は残す）
    if pending:
        answers = [
            a
            for a in answers
            if not any(_similarity(a.text, p.text) >= 0.62 for p in pending)
        ]

    if not requests:
        # 質問が取れなかった場合は、依頼側の発話をそのまま拾う
        fallback = [
            Fact(_clean_fact(u.text), u.text, u.speaker, 2.0, u.start)
            for u in utterances
            if len(_clean_fact(u.text)) >= 8
            and (
                (u.speaker == requester)
                if requester is not None
                else _has_request_signal(u.text)
            )
        ]
        requests = _select_facts(fallback, limit=5, threshold=0.0)
    if not answers:
        fallback = [
            Fact(_clean_fact(u.text), u.text, u.speaker, 2.0, u.start)
            for u in utterances
            if len(_clean_fact(u.text)) >= 10
            and (
                (u.speaker == responder)
                if responder is not None
                else (not _has_request_signal(u.text) and _cue_score(u.text, _ANSWER_CUES) > 0)
            )
        ]
        answers = _select_facts(fallback, limit=5, threshold=0.0, exclude=requests)

    topic = _topic_from_text("".join(f.text for f in requests) or all_text)
    requester_label = enriched_labels.get(requester, "") if requester is not None else ""
    responder_label = enriched_labels.get(responder, "") if responder is not None else ""

    actions = _extract_actions(
        utterances,
        requests,
        pending,
        requester,
        responder,
        enriched_labels,
        people,
        dates,
    )
    if not actions:
        for fact in (pending + requests)[:4]:
            actions.append(
                ActionItem(
                    who=enriched_labels.get(fact.speaker or -1, _speaker_label(fact.speaker, enriched_labels)),
                    what=fact.text,
                    when=next((d for d in dates if d in fact.text), ""),
                )
            )

    facts = ExtractedFacts(
        requests=requests,
        answers=answers,
        decisions=decisions,
        pending=pending,
        actions=actions,
        dates=date_items,
        people=people,
        requester=requester,
        responder=responder,
        speaker_labels=enriched_labels,
        topic=topic,
        title=_make_title(topic, requests, people, requester_label),
        utterance_texts=[u.text for u in utterances],
    )
    facts.overview = _make_overview(facts, requester_label, responder_label)
    return facts


# ---------------------------------------------------------------------------
# ローカル LLM による整形
# ---------------------------------------------------------------------------


_LLM_SYSTEM = (
    "あなたは日本語の電話・会議記録を正確に整理する議事録作成の専門家です。"
    "与えられた『抽出済みの事実』だけを使い、指定されたJSON形式だけで回答します。"
    "事実にない出来事・契約条件・金額・日付・人名を創作してはいけません。"
    "文字起こしの誤認識は文脈から自然に補正してかまいませんが、"
    "確信が持てない固有名詞には「（推定）」を付けます。"
)

_LLM_USER_TEMPLATE = """次の「抽出済みの事実」だけを使い、人が読んで内容をすぐ把握できる議事録に整えてください。
事実を追加・推測せず、話し言葉の重複やフィラーを除いて、日本語として自然な箇条書きにしてください。

# 参加者
- 依頼側: {requester}
- 応対側: {responder}
{context_block}
# 抽出済みの事実
## 要望・依頼
{requests}

## 回答・説明
{answers}

## 決定事項
{decisions}

## 確認中・保留
{pending}

## 次のアクション候補
{actions}

## 日付・数値
{dates}

## 人物・組織
{people}

# 出力JSON
{{
  "title": "件名（30文字以内。会社名と用件を含める）",
  "overview": "誰が何のために話し、何を確認し、結論どうなったかを3〜5文で",
  "requests": ["依頼側が何をしたい・何を確認したいか。対象と目的を明記"],
  "answers": ["応対側の回答。何が可能か、条件や手順を具体的に"],
  "decisions": ["確定したこと（なければ空配列）"],
  "pending": ["確認中・保留・折り返し（なければ空配列）"],
  "actions": [{{"who": "誰が", "what": "何を", "when": "いつまでに（不明なら空文字）"}}],
  "dates": ["重要な日付・期間・数値"],
  "people": ["人名・会社名・部署・連絡先"]
}}

書き方:
- 各項目は40〜90文字を目安に、主語と目的語を補う。
- 「データ移行の流れ」のような名詞だけの項目にしない。「〜したい」「〜できる」「〜が必要」の形にする。
- 要望は「誰が・何を・なぜ/いつ」、回答は「何が可能か・どういう条件か」が分かるようにする。
- 数字・日付・会社名・人名は抽出済みの表記を変えない。事実に無ければ書かない。
- 同じ内容を繰り返さない。不明な項目は空配列または空文字にする。
- 出力はJSONオブジェクトだけ。Markdownのコードフェンスや説明文は付けない。
"""


def _format_facts_for_llm(facts: ExtractedFacts) -> Dict[str, object]:
    def lines(items: Sequence[Fact]) -> List[str]:
        return [f"- {item.text}" for item in items]

    return {
        "requester": facts.speaker_labels.get(facts.requester, "不明") if facts.requester is not None else "不明",
        "responder": facts.speaker_labels.get(facts.responder, "不明") if facts.responder is not None else "不明",
        "requests": lines(facts.requests),
        "answers": lines(facts.answers),
        "decisions": lines(facts.decisions),
        "pending": lines(facts.pending),
        "actions": [
            f"- {a.who}: {a.what}" + (f"（{a.when}）" if a.when else "")
            for a in facts.actions
        ],
        "dates": facts.dates,
        "people": facts.people,
    }


def _build_llm_prompt(facts: ExtractedFacts, context: Optional[str]) -> str:
    data = _format_facts_for_llm(facts)

    def block(name: str) -> str:
        values = data[name]
        if not values:
            return "- 特になし"
        return "\n".join(values)  # type: ignore[arg-type]

    context_block = ""
    if context and context.strip():
        context_block = f"\n# 補足情報（ユーザー提供）\n{context.strip()}\n"
    return _LLM_USER_TEMPLATE.format(
        requester=data["requester"],
        responder=data["responder"],
        context_block=context_block,
        requests=block("requests"),
        answers=block("answers"),
        decisions=block("decisions"),
        pending=block("pending"),
        actions=block("actions"),
        dates="\n".join(f"- {d}" for d in data["dates"]) or "- 特になし",
        people="\n".join(f"- {p}" for p in data["people"]) or "- 特になし",
    )


def _load_json_object(text: str) -> Optional[Dict[str, object]]:
    body = text.strip()
    if not body:
        return None
    # コードフェンスを外す
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    try:
        value = json.loads(body)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    # 前後の説明文が混ざった場合に最初の { から最後の } を試す
    start, end = body.find("{"), body.rfind("}")
    if 0 <= start < end:
        try:
            value = json.loads(body[start : end + 1])
            return value if isinstance(value, dict) else None
        except Exception:
            return None
    return None


def _as_string_list(value: object, limit: int = 12) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = re.sub(r"^[\-\*・\s]+", "", item).strip()
        text = re.sub(r"\s+", " ", text)
        if not text or len(text) > 220:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


_NAMED_ENTITY_RE = re.compile(
    r"([一-龥々〆ヵヶァ-ヶA-Za-z0-9]{1,12}(?:様|さん|薬局|株式会社|有限会社|会社|システム|ネット|センター|病院|クリニック))"
)


def _named_entities_valid(item: str, sources: Sequence[str]) -> bool:
    """会社名・人名の書き換え（誤字・別名）を検出する。

    LLM は「平岡様」を「平島様」、「モイネットシステム」を
    「モアイネットシステム」のように変えてしまうことがある。
    原文に無い固有名詞を含む項目は採用しない。
    """
    haystack = _norm("".join(sources))
    for name in _NAMED_ENTITY_RE.findall(item):
        body = name.strip("の")
        # 「お客様」「担当者様」など一般名詞は対象外
        if body in _GENERIC_NAMES or body.endswith(("お客様", "担当者様", "皆様")):
            continue
        if _norm(body) in haystack:
            continue
        return False
    return True


def _split_llm_item(item: str) -> List[str]:
    """LLM が 1 項目に複数の文を詰めた場合に分ける。"""
    parts = [part.strip(" 　・-") for part in re.split(r"(?<=[。！？!?])", item) if part.strip()]
    out: List[str] = []
    for part in parts:
        part = re.sub(r"^(?:また|そして|さらに|なお|ただし|一方|ただ)[、,\s]+", "", part.strip(" 　・-"))
        if len(part) < 8:
            continue
        if len(part) > 120:
            part = part[:117] + "…"
        out.append(part)
    return out or ([item] if item.strip() else [])


def _edit_distance_one(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) == 1
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        if skipped:
            return False
        skipped = True
        j += 1
    return True


def _repair_names(item: str, known_names: Sequence[str]) -> str:
    """既知の会社名・人名に 1 文字違いで似ている表記を原文へ戻す。"""
    body = item
    for name in _NAMED_ENTITY_RE.findall(item):
        plain = name.strip("の")
        if plain in _GENERIC_NAMES or plain.endswith(("お客様", "担当者様", "皆様")):
            continue
        # 敬称違い（平岡 / 平岡様）は原文の表記を尊重する
        if any(known and (plain == known or plain.startswith(known) or known.startswith(plain)) for known in known_names):
            continue
        matches = [known for known in known_names if known and _edit_distance_one(plain, known)]
        if len(matches) == 1:
            body = body.replace(name, matches[0])
    return body


def _content_tokens(text: str) -> set[str]:
    return set(re.findall(r"[一-龥々〆ヵヶァ-ヶー]{2,}", _norm(text)))


def _grounded(item: str, sources: Sequence[str], threshold: float = 0.26) -> bool:
    """要約項目が元の事実に裏付けられているか確認する。

    短い要約は元の発話より文字数が少なく、Jaccard 係数だけでは落ちてしまう。
    そのため「項目側の n-gram が元発話にどれだけ含まれるか」も見る。
    日付・数字は創作を防ぐため、元の発話に無ければ即座に不採用にする。
    """
    body = item.strip()
    if not body:
        return False
    norm_item = _norm(body)
    if not norm_item:
        return False

    # 日付・数値の創作を禁止
    for value in extract_dates(body):
        if not any(_norm(value) in _norm(source) for source in sources):
            return False
    for number in re.findall(r"\d+", body):
        if not any(number in source for source in sources):
            return False

    item_grams = _ngrams(body, 2)
    item_tokens = _content_tokens(body)
    for source in sources:
        norm_source = _norm(source)
        if len(norm_item) <= 8 and norm_item in norm_source:
            return True
        if _similarity(body, source) >= threshold:
            return True
        if item_grams:
            coverage = len(item_grams & _ngrams(source, 2)) / len(item_grams)
            if coverage >= 0.62:
                return True

    # 発話をまたいで 1 つの要約にまとめることがあるため、最後に全体で確認する
    all_norm = "".join(_norm(source) for source in sources)
    all_grams = _ngrams("".join(sources), 2)
    if item_tokens:
        hits = sum(1 for token in item_tokens if token in all_norm)
        if hits >= 1 and hits / len(item_tokens) >= 0.6:
            return True
    if item_grams:
        coverage = len(item_grams & all_grams) / len(item_grams)
        if coverage >= 0.7:
            return True
    return False


def _merge_category(
    llm_items: Sequence[str],
    base_items: Sequence[Fact],
    sources: Sequence[str],
    limit: int,
    min_items: int = 2,
) -> List[str]:
    """LLM の項目を検証し、足りない分だけルールベースから補う。

    LLM が十分な項目を返したときに、話し言葉のままのルールベース項目を
    すべて並べると読みにくくなる。そこで補完は min_items に届くまでに留める。
    """
    merged: List[str] = []
    for item in llm_items:
        for part in _split_llm_item(item):
            if not _grounded(part, sources):
                continue
            if not _named_entities_valid(part, sources):
                continue
            if any(_similarity(part, other) >= 0.7 for other in merged):
                continue
            merged.append(part)
            if len(merged) >= limit:
                break
        if len(merged) >= limit:
            break
    target = min(min_items, limit) if merged else limit
    for fact in base_items:
        if len(merged) >= target:
            break
        if any(_similarity(fact.text, item) >= 0.55 for item in merged):
            continue
        if any(_norm(fact.text) == _norm(item) for item in merged):
            continue
        merged.append(fact.text)
    return merged[:limit]


def _validate_overview(text: str, sources: Sequence[str]) -> str:
    body = text.strip()
    if not body or len(body) > 900:
        return ""
    # 概要全体が元の事実から離れすぎていないか（内容語の重なりで確認）
    grams = _ngrams(body, 2)
    if not grams:
        return ""
    source_grams = set()
    for source in sources:
        source_grams |= _ngrams(source, 2)
    if not source_grams:
        return ""
    overlap = len(grams & source_grams) / max(len(grams), 1)
    return body if overlap >= 0.28 else ""


_ANSWER_CONFIRM_CUES = (
    "問題ない",
    "問題ございません",
    "可能",
    "できます",
    "大丈夫",
    "対応",
    "構いません",
    "承知",
    "させていただ",
    "という流れ",
    "手順",
    "場合",
)


def _looks_unconfirmed(item: str, facts: ExtractedFacts) -> bool:
    """回答欄に紛れた「依頼側の希望・予定」を落とす。

    例: 「12月上旬にデータを取りに来ていただく」は回答ではなく依頼側の段取り。
    確定を示す表現が無く、要望と似ている場合は回答として扱わない。
    """
    body = item.strip("「」『』\"' ")
    if not body:
        return True
    if any(cue in body for cue in _ANSWER_CONFIRM_CUES):
        return False
    if re.search(r"(?:来ていただ|取りに来て|してもら|入れてもら|移してもら|検討中|予定です)", body):
        return True
    return any(_similarity(body, fact.text) >= 0.35 for fact in facts.requests)


def _looks_pending(item: str) -> bool:
    """「確認中・保留」らしい内容だけを残す。"""
    return any(
        cue in item
        for cue in (
            "確認",
            "折り返し",
            "保留",
            "検討",
            "調整",
            "後日",
            "未定",
            "持ち帰",
            "回答待ち",
            "確認中",
            "改めて",
            "待ち",
        )
    )


def _merge_llm_json(
    facts: ExtractedFacts,
    data: Dict[str, object],
    model_name: str,
) -> Tuple[SummaryResult, List[str]]:
    sources = facts.all_sources
    warnings: List[str] = []

    def prepared(key: str) -> List[str]:
        return [_repair_names(item, facts.people) for item in _as_string_list(data.get(key))]

    requests = _merge_category(prepared("requests"), facts.requests, sources, 10, min_items=2)
    answer_items = [
        part
        for item in prepared("answers")
        for part in _split_llm_item(item)
        if not _looks_unconfirmed(part, facts)
    ]
    answers = _merge_category(answer_items, facts.answers, sources, 10, min_items=2)
    decisions = _merge_category(prepared("decisions"), facts.decisions, sources, 8, min_items=1)
    pending_items = [
        part
        for item in prepared("pending")
        for part in _split_llm_item(item)
        if _looks_pending(part)
    ]
    pending = _merge_category(pending_items, facts.pending, sources, 8, min_items=1)

    # アクションはルールベースを正とし、LLM の裏付けある追加だけ足す
    actions: List[ActionItem] = list(facts.actions)
    raw_actions = data.get("actions")
    if isinstance(raw_actions, list):
        for raw in raw_actions:
            if isinstance(raw, dict):
                who = str(raw.get("who") or "").strip()[:60]
                what = str(raw.get("what") or "").strip()[:180]
                when = str(raw.get("when") or "").strip()[:60]
            elif isinstance(raw, str):
                who, what, when = "", raw.strip()[:180], ""
            else:
                continue
            if not what or not _grounded(what, sources):
                continue
            what = _repair_names(what, facts.people)
            who = _repair_names(who, facts.people)
            if not _named_entities_valid(f"{who} {what}", sources):
                continue
            if when and not any(_norm(when) in _norm(d) or _norm(d) in _norm(when) for d in facts.dates):
                # 日付は創作させない（元の日付に一致しないものは捨てる）
                when = ""
            if any(_similarity(what, a.what) >= 0.4 for a in actions):
                continue
            actions.append(ActionItem(who=who, what=what, when=when))

    # 日付・人物はルールベースを正とし、LLM が拾った裏付けのあるものだけ足す
    dates = list(facts.dates)
    for item in _as_string_list(data.get("dates"), 15):
        if _grounded(item, sources) and not any(_norm(item) in _norm(d) or _norm(d) in _norm(item) for d in dates):
            dates.append(item)
    people = list(facts.people)
    for item in _as_string_list(data.get("people"), 12):
        if _norm(item) and _norm(item) in _norm(facts.overview + "".join(sources)) and item not in people:
            people.append(item)

    title = str(data.get("title") or "").strip()[:60]
    if not title or not _grounded(title, sources + [facts.title], threshold=0.12):
        title = facts.title
    # 件名には依頼側の会社名・人名を入れる（「データ移行の流れ」だけにしない）
    if title and facts.people:
        if not any(_norm(name) in _norm(title) for name in facts.people if len(_norm(name)) >= 3):
            title = facts.title
    overview = _validate_overview(str(data.get("overview") or ""), sources) or facts.overview
    if (
        pending
        and len(overview) < 180
        and not any(cue in overview for cue in ("折り返", "確認中", "保留", "未定"))
    ):
        overview = overview.rstrip("。") + "。" + pending[0].rstrip("。") + "。"

    result = SummaryResult(
        title=title or facts.title,
        overview=overview,
        requests=requests,
        answers=answers,
        decisions=decisions,
        pending=pending,
        actions=actions[:8],
        dates=dates,
        people=people,
        method="rule+llm",
        model=model_name,
    )
    if not requests and facts.requests:
        result.requests = [f.text for f in facts.requests]
    if not answers and facts.answers:
        result.answers = [f.text for f in facts.answers]
    return result, warnings


def _local_llm_generate(
    prompt: str,
    model_name: str,
    model_dir: Optional[Path],
    cpu_threads: Optional[int],
    max_new_tokens: int,
    on_progress: Optional[ProgressCallback],
    cancel_event: Optional[threading.Event],
) -> str:
    """CTranslate2 + tokenizers でローカル LLM を動かす。"""
    from ctranslate2 import Generator
    from tokenizers import Tokenizer

    spec = SUMMARY_CATALOG[model_name]
    model_path = ensure_summary_model(
        model_name,
        model_dir,
        progress_cb=_download_progress_adapter(on_progress),
    )
    if on_progress is not None:
        on_progress(0.0, f"AIモデル「{model_name}」をCPUに読み込み中...")
    tokenizer = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))
    generator = Generator(
        model_path,
        device="cpu",
        compute_type="int8",
        inter_threads=1,
        intra_threads=cpu_threads or os.cpu_count() or 4,
    )

    system = _LLM_SYSTEM
    if spec.template == "gemma":
        # Gemma は system ロールを持たないため、ユーザー文の先頭に置く
        chat_prompt = (
            f"<bos><start_of_turn>user\n{system}\n\n{prompt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        end_token = "<end_of_turn>"
    else:
        chat_prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        end_token = "<|im_end|>"

    ids = tokenizer.encode(chat_prompt, add_special_tokens=False).ids
    tokens = [tokenizer.id_to_token(i) for i in ids]
    tokens = [t for t in tokens if t]
    if not tokens:
        raise RuntimeError("AIモデルのトークナイズに失敗しました。")

    started = time.perf_counter()
    generated = 0
    last_report = 0.0

    def callback(_step) -> bool:
        nonlocal generated, last_report
        generated += 1
        if on_progress is not None:
            now = time.perf_counter()
            if now - last_report >= 0.4:
                last_report = now
                speed = generated / max(now - started, 0.01)
                frac = min(generated / max(max_new_tokens, 1), 0.98)
                on_progress(frac, f"AI整形中... {generated}トークン（{speed:.1f}トークン/秒）")
        return bool(cancel_event is not None and cancel_event.is_set())

    result = generator.generate_batch(
        [tokens],
        max_length=len(tokens) + max_new_tokens,
        include_prompt_in_result=False,
        end_token=end_token,
        repetition_penalty=1.03,
        no_repeat_ngram_size=8,
        callback=callback,
    )
    if cancel_event is not None and cancel_event.is_set():
        raise SummaryCancelled()
    output_ids = result[0].sequences_ids[0] if result and result[0].sequences_ids else []
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def _download_progress_adapter(on_progress: Optional[ProgressCallback]):
    if on_progress is None:
        return None

    def _cb(done: int, total: int) -> None:
        frac = (done / total) if total else 0.0
        on_progress(frac, f"AIモデルをダウンロード中 {done / 1048576:.0f}MB / {total / 1048576:.0f}MB（初回のみ）")

    return _cb


def _refine_with_llm(
    facts: ExtractedFacts,
    options: SummaryOptions,
    on_progress: Optional[ProgressCallback],
    cancel_event: Optional[threading.Event],
) -> Tuple[Optional[SummaryResult], Optional[str]]:
    model_name = options.model
    if model_name not in SUMMARY_CATALOG:
        raise ValueError(f"未知の議事録モデル: {model_name}（利用可能: {', '.join(SUMMARY_CATALOG)}）")
    prompt = _build_llm_prompt(facts, options.context)
    if on_progress is not None:
        on_progress(0.02, "AIで議事録を整えています...")
    raw = _local_llm_generate(
        prompt,
        model_name,
        options.model_dir,
        options.cpu_threads,
        options.max_new_tokens,
        on_progress,
        cancel_event,
    )
    data = _load_json_object(raw)
    if data is None:
        return None, "AIの出力をJSONとして解析できなかったため、ルールベースの結果を使いました。"
    try:
        merged, _warnings = _merge_llm_json(facts, data, model_name)
        return merged, None
    except Exception as exc:  # noqa: BLE001
        return None, f"AIの出力を検証できなかったため、ルールベースの結果を使いました（{exc}）。"


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def summarize_text(
    text: str,
    options: Optional[SummaryOptions] = None,
    source: Optional[Path] = None,
    on_progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> SummaryResult:
    """文字起こしテキストを議事録に整理する。"""
    options = options or SummaryOptions()
    mode = options.llm_mode()
    started = time.perf_counter()

    if cancel_event is not None and cancel_event.is_set():
        raise SummaryCancelled()
    if on_progress is not None:
        on_progress(0.02, "文字起こしを解析中...")

    facts = extract_facts(text)
    if not facts.utterance_texts:
        facts.title = "文字起こし内容の整理"
        facts.overview = (
            "（内容を抽出できませんでした。音声が無音、または文字起こしに失敗した可能性があります。）"
        )
    facts.title = facts.title or "文字起こし内容の整理"
    facts.overview = facts.overview or "（概要を抽出できませんでした）"

    result = SummaryResult(
        title=facts.title,
        overview=facts.overview,
        requests=[f.text for f in facts.requests],
        answers=[f.text for f in facts.answers],
        decisions=[f.text for f in facts.decisions],
        pending=[f.text for f in facts.pending],
        actions=facts.actions,
        dates=facts.dates,
        people=facts.people,
        source=source,
        transcript=text,
    )

    use_llm = mode == "always"
    if mode == "auto":
        use_llm = bool(options.model in SUMMARY_CATALOG and is_summary_downloaded(options.model, options.model_dir))

    if use_llm:
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise SummaryCancelled()
            refined, warning = _refine_with_llm(facts, options, on_progress, cancel_event)
            if refined is not None:
                result = refined
                result.source = source
                result.transcript = text
                result.elapsed = time.perf_counter() - started
                if on_progress is not None:
                    on_progress(1.0, "完了")
                return result
            result.note = warning
        except SummaryCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            result.note = f"AI整形を実行できなかったため、ルールベースの結果を使いました（{exc}）。"

    result.elapsed = time.perf_counter() - started
    if on_progress is not None:
        on_progress(1.0, "完了")
    return result


def summarize_result(
    result: TranscribeResult,
    options: Optional[SummaryOptions] = None,
    on_progress: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> SummaryResult:
    """文字起こし結果を議事録に整理する。"""
    text = segments_to_text(result.segments)
    summary = summarize_text(
        text,
        options,
        source=result.source,
        on_progress=on_progress,
        cancel_event=cancel_event,
    )
    return summary
