"""用語集（固有名詞ヒントと誤認識の置換）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Tuple

MAX_PROMPT_CHARS = 400
_REPLACEMENT_SEPARATOR = re.compile(r"\s*(?:=>|->|→|\t)\s*")


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"用語集の文字コードを判別できません（UTF-8 または Shift_JIS で保存してください）: {path}")


@dataclass
class Glossary:
    terms: Tuple[str, ...] = ()
    replacements: Tuple[Tuple[str, str], ...] = ()
    _pattern: Optional[Pattern[str]] = field(init=False, default=None, repr=False, compare=False)
    _mapping: Dict[str, str] = field(init=False, default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.terms = tuple(_dedupe(term.strip() for term in self.terms))
        pairs: List[Tuple[str, str]] = []
        for source, target in self.replacements:
            source = source.strip()
            target = target.strip()
            if not source or not target:
                raise ValueError("置換ルールは「誤り=>正しい」の形式で、両方を空でない文字列にしてください")
            pairs.append((source, target))
        self.replacements = tuple(pairs)
        if pairs:
            ordered = sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)
            self._mapping = dict(ordered)
            self._pattern = re.compile("|".join(re.escape(source) for source, _ in ordered))

    @property
    def is_empty(self) -> bool:
        return not self.terms and not self.replacements

    def build_prompt(self, base: Optional[str] = None) -> Optional[str]:
        """Whisper に渡すヒント文を作る。用語は末尾にまとめる。"""
        base = (base or "").strip()
        terms = list(self.terms)
        for _, target in self.replacements:
            if target not in terms:
                terms.append(target)
        if not terms:
            return base or None

        separator = ""
        if base and not base.endswith(("。", "．", ".", "！", "？", "!", "?")):
            separator = "。"
        prefix = f"{base}{separator}用語: " if base else "用語: "
        selected: List[str] = []
        used = len(prefix)
        for term in terms:
            addition = len(term) + (1 if selected else 0)
            if used + addition > MAX_PROMPT_CHARS:
                break
            selected.append(term)
            used += addition
        if not selected:
            return base or None
        return prefix + "、".join(selected)

    def apply(self, text: str) -> str:
        if self._pattern is None:
            return text
        return self._pattern.sub(lambda match: self._mapping[match.group(0)], text)


def load_glossary(path: Path) -> Glossary:
    """用語集ファイルを読む。

    1行1項目。``#`` で始まる行はコメント。
    区切り（``=>`` / ``->`` / ``→`` / タブ）があれば誤認識の置換、
    なければ固有名詞としてヒント文に追加する。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"用語集が見つかりません: {path}")

    terms: List[str] = []
    replacements: List[Tuple[str, str]] = []
    seen: Dict[str, str] = {}

    for line_number, raw_line in enumerate(_read_text(path).splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = _REPLACEMENT_SEPARATOR.split(line, maxsplit=1)
        if len(parts) == 2:
            source, target = (part.strip() for part in parts)
            if not source or not target:
                raise ValueError(
                    f"{path}:{line_number}: 置換は「誤り=>正しい」の形式で指定してください: {line}"
                )
            previous = seen.get(source)
            if previous is not None:
                if previous != target:
                    raise ValueError(
                        f"{path}:{line_number}: 「{source}」の置換先が重複しています"
                    )
                continue
            seen[source] = target
            replacements.append((source, target))
        else:
            terms.append(line)

    return Glossary(tuple(terms), tuple(replacements))
