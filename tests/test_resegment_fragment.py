"""語の途中で切れた「切れ端」の連結と、話者ラベルの孤立補正。"""

from koemoji.core import Segment, resegment


def test_merge_fragment_across_long_gap():
    """「ドク」＋（2秒の無音）＋「ターは薬剤の」は語の途中なので連結する。"""
    segments = [
        Segment(42.0, 43.0, "ドク", 0),
        Segment(45.0, 47.0, "ターは薬剤の", 0),
    ]
    out = resegment(segments, max_chars=32)
    assert [s.text for s in out] == ["ドクターは薬剤の"]


def test_merge_chained_fragments():
    """切れ端が連続しても、長くなりすぎない範囲で連結する。"""
    segments = [
        Segment(213.0, 214.0, "電", 0),
        Segment(215.0, 216.0, "話に連", 0),
        Segment(218.0, 221.0, "絡していただけると", 0),
    ]
    out = resegment(segments, max_chars=32)
    assert [s.text for s in out] == ["電話に連絡していただけると"]


def test_long_previous_segment_is_not_glued_across_gap():
    """直前が切れ端でなければ、無音を跨いで短い相づちを吸い上げない。"""
    segments = [
        Segment(10.0, 12.0, "そうですね", 0),
        Segment(30.0, 31.0, "はい", 0),
    ]
    out = resegment(segments, max_chars=32)
    assert [s.text for s in out] == ["そうですね", "はい"]


def test_speaker_island_fragment_is_relabelled_and_merged():
    """1 行だけ別話者になった切れ端は前後に合わせ、そのうえで連結する。

    「再」は漢字で終わる語頭切れなので、23 秒の無音を跨いでも連結してよい
    （ひらがなの切れ端と違い、相づちである可能性が低い）。
    """
    segments = [
        Segment(163.0, 166.0, "ご相談させていただいてもよろしいですか", 1),
        Segment(178.0, 179.0, "再", 0),
        Segment(202.0, 206.0, "度ちょっとその部分この機能", 1),
    ]
    out = resegment(segments, max_chars=32)
    assert all(s.speaker == 1 for s in out)
    assert out[-1].text.startswith("再度")


def test_speaker_change_is_kept_when_fragment_is_long():
    """切れ端でない行の話者は動かさない。"""
    segments = [
        Segment(0.0, 2.0, "ありがとうございます", 0),
        Segment(3.0, 5.0, "いえいえこちらこそ", 1),
        Segment(6.0, 8.0, "よろしくお願いします", 0),
    ]
    out = resegment(segments, max_chars=32)
    assert [s.speaker for s in out] == [0, 1, 0]


def test_tail_fragment_not_joined_across_very_long_gap():
    """4 文字の切れ端でも、18 秒の無音を跨いでは連結しない。"""
    segments = [
        Segment(10.0, 12.0, "なるほど", 0),
        Segment(30.0, 31.0, "はい", 0),
    ]
    out = resegment(segments, max_chars=32)
    assert [s.text for s in out] == ["なるほど", "はい"]


def test_tail_fragment_joined_at_word_join_gap_boundary():
    """切れ端の救済は _WORD_JOIN_GAP（3.0 秒）ちょうどまで（閉区間）。"""
    segments = [
        Segment(0.0, 2.0, "なるほど", 0),
        Segment(5.0, 6.0, "はい", 0),
    ]
    out = resegment(segments, max_chars=32)
    assert [s.text for s in out] == ["なるほどはい"]
