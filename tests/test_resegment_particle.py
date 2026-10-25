
from koemoji.core import Segment, resegment


def _pieces(text: str, max_chars: int = 32):
    out = resegment([Segment(0.0, 10.0, text)], max_chars=max_chars, language="ja")
    return [seg.text for seg in out]


def test_no_mid_verb_split():
    """"上(あ)がって" の "が" を助詞と誤認して語中で切らない。"""
    text = "大丈夫ですけれどももう今は屋外の気温昼も夜も上がってますので空気の入れ替え"
    pieces = _pieces(text, max_chars=20)
    assert "".join(pieces) == text
    for piece in pieces[:-1]:
        assert not piece.endswith("上が")
        assert not piece.endswith("入れ替")


def test_keeps_since_before_break():
    """"ので" は接続表現としてまとめ、その直後で切る。"""
    text = "屋外の気温昼も夜も上がってますので空気の入れ替えだけ"
    pieces = _pieces(text, max_chars=18)
    assert len(pieces) > 1
    assert pieces[0].endswith("ので")
    assert pieces[1].startswith("空気")
    assert "".join(pieces) == text


def test_breaks_after_particle():
    text = "やっぱりその街の良さをアピールしていくっていうそういう姿勢が基本にあった上での"
    pieces = _pieces(text, max_chars=32)
    assert all(len(p) <= 32 for p in pieces)
    assert "".join(pieces) == text
    assert pieces[0].endswith("が")
