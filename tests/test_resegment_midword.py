"""行末が漢字で終わる「語の途中切れ」の検出と、話者を跨いだ連結。"""

from koemoji.core import Segment, _ends_mid_word, resegment


def test_ends_mid_word_detects_kanji_tail():
    assert _ends_mid_word("思いますので電") is True
    assert _ends_mid_word("では資料") is True


def test_ends_mid_word_ignores_katakana_word():
    """カタカナ語は完結した外来語かもしれないので、漢字だけで判定する。

    「ドク」のような短い切れ端は _is_tail_fragment（4 文字以下）側で拾う。
    """
    assert _ends_mid_word("ドク") is False
    assert _ends_mid_word("カメラ") is False


def test_ends_mid_word_ignores_sentence_end_and_particles():
    assert _ends_mid_word("ありがとうございます。") is False
    assert _ends_mid_word("ファーミーのほうに") is False
    assert _ends_mid_word("よろしくお願いします") is False
    assert _ends_mid_word("") is False


def test_join_mid_word_split_across_speakers():
    """語の途中で切れた境界は話者が違って見えても連結する（境界は幻）。"""
    segments = [
        Segment(333.0, 336.0, "思いますので電", 1),
        Segment(338.0, 339.0, "話に連", 0),
    ]
    out = resegment(segments, max_chars=32)
    assert [s.text for s in out] == ["思いますので電話に連"]
    assert out[0].speaker == 1


def test_mid_word_join_respects_max_chars():
    """連結した結果が max_chars を超えるなら切ったままにする。"""
    segments = [
        Segment(0.0, 2.0, "あ" * 20 + "電", 0),
        Segment(2.0, 3.0, "話に連", 0),
    ]
    out = resegment(segments, max_chars=12)
    texts = [s.text for s in out]
    assert "".join(texts) == "あ" * 20 + "電話に連"
    assert all(len(t) <= 12 for t in texts)


def test_no_join_when_previous_ends_with_particle():
    """助詞で終わる行は語の途中ではないので、話者が違えば連結しない。"""
    segments = [
        Segment(0.0, 2.0, "資料はこちらに", 0),
        Segment(2.5, 4.0, "お願いします", 1),
    ]
    out = resegment(segments, max_chars=32)
    assert [s.text for s in out] == ["資料はこちらに", "お願いします"]


def test_far_gap_mid_word_is_not_joined():
    """無音が長すぎる場合は語中切れでも連結しない（別の話題の可能性）。"""
    segments = [
        Segment(0.0, 2.0, "では資料", 0),
        Segment(20.0, 22.0, "お願いします", 1),
    ]
    out = resegment(segments, max_chars=32)
    assert [s.text for s in out] == ["では資料", "お願いします"]
