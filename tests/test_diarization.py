"""話者分離のマージロジックと会話ログ出力のテスト（モデル不要）。"""

import numpy as np

from koemoji.core import Segment, Word, to_log, to_srt, to_txt, to_vtt
from koemoji.diarization import (
    SpeakerSpan,
    _agglomerative_cluster,
    _build_spans,
    _windows_for_region,
    attribute_segments,
    merge_words_with_speakers,
)
from koemoji.models import DEFAULT_EMBEDDING, DIARIZATION_CATALOG


def test_merge_basic_speaker_change():
    words = [
        Word(0.0, 0.5, "あ"),
        Word(0.5, 1.0, "い"),
        Word(2.0, 2.5, "う"),
        Word(2.5, 3.0, "え"),
    ]
    spans = [SpeakerSpan(0.0, 1.2, 5), SpeakerSpan(1.9, 3.1, 9)]
    merged = merge_words_with_speakers(words, spans)
    assert len(merged) == 2
    assert merged[0].text == "あい"
    assert merged[0].speaker == 0
    assert merged[0].start == 0.0 and merged[0].end == 1.0
    assert merged[1].text == "うえ"
    assert merged[1].speaker == 1


def test_merge_gap_splits_same_speaker():
    words = [Word(0.0, 0.5, "A"), Word(5.0, 5.5, "B")]
    spans = [SpeakerSpan(0.0, 0.6, 1), SpeakerSpan(4.9, 5.6, 1)]
    merged = merge_words_with_speakers(words, spans)
    assert len(merged) == 2
    assert all(seg.speaker == 0 for seg in merged)


def test_merge_speaker_numbering_by_first_appearance():
    words = [Word(0.0, 0.5, "x"), Word(2.0, 2.5, "y"), Word(4.0, 4.5, "z")]
    spans = [
        SpeakerSpan(0.0, 0.6, 42),
        SpeakerSpan(1.9, 2.6, 7),
        SpeakerSpan(3.9, 4.6, 42),
    ]
    merged = merge_words_with_speakers(words, spans)
    assert [seg.speaker for seg in merged] == [0, 1, 0]


def test_merge_no_spans_returns_empty():
    assert merge_words_with_speakers([Word(0.0, 1.0, "x")], []) == []


def test_attribute_segments_fallback():
    segments = [Segment(0.0, 1.0, "one"), Segment(2.0, 3.0, "two")]
    spans = [SpeakerSpan(0.0, 1.1, 3), SpeakerSpan(1.9, 3.1, 3)]
    attributed = attribute_segments(segments, spans)
    assert len(attributed) == 2
    assert all(seg.speaker == 0 for seg in attributed)


def test_to_log_format():
    segments = [
        Segment(3.0, 5.0, "こんにちは。", speaker=0),
        Segment(7.0, 9.0, "よろしくお願いします。", speaker=1),
    ]
    out = to_log(segments)
    assert out == (
        "[00:00:03] 話者1: こんにちは。\n"
        "[00:00:07] 話者2: よろしくお願いします。\n"
    )


def test_to_txt_auto_log_when_speakers():
    segments = [Segment(0.0, 1.0, "テスト", speaker=0)]
    assert to_txt(segments).startswith("[00:00:00] 話者1: テスト")
    plain = [Segment(0.0, 1.0, "テスト")]
    assert to_txt(plain) == "テスト\n"


def test_srt_vtt_speaker_prefix():
    segments = [Segment(0.0, 1.5, "おはよう。", speaker=0)]
    assert "話者1: おはよう。" in to_srt(segments)
    assert "話者1: おはよう。" in to_vtt(segments)


def test_diarization_catalog():
    assert DEFAULT_EMBEDDING in DIARIZATION_CATALOG
    assert "3dspeaker" in DIARIZATION_CATALOG
    for key, spec in DIARIZATION_CATALOG.items():
        assert spec.key == key
        assert spec.url.startswith("https://")
        assert spec.filename.endswith(".onnx")


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def test_cluster_threshold_separates_two_speakers():
    a = _unit([1.0, 0.0, 0.0])
    b = _unit([0.9, 0.1, 0.0])
    c = _unit([0.0, 1.0, 0.0])
    d = _unit([0.1, 0.9, 0.0])
    labels = _agglomerative_cluster([a, b, c, d], threshold=0.5, num_clusters=-1)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert sorted(set(labels)) == [0, 1]


def test_cluster_forced_count():
    a = _unit([1.0, 0.0, 0.0])
    b = _unit([0.9, 0.1, 0.0])
    c = _unit([0.0, 1.0, 0.0])
    labels = _agglomerative_cluster([a, b, c], threshold=0.5, num_clusters=2)
    assert len(set(labels)) == 2
    labels_one = _agglomerative_cluster([a, b, c], threshold=0.5, num_clusters=1)
    assert len(set(labels_one)) == 1


def test_cluster_empty_and_single():
    assert _agglomerative_cluster([], threshold=0.5) == []
    assert _agglomerative_cluster([_unit([1.0, 0.0])], threshold=0.5) == [0]


def test_windows_for_short_region():
    assert _windows_for_region(1.0, 2.0) == [(1.0, 2.0)]


def test_windows_for_long_region_covers_full_range():
    windows = _windows_for_region(0.0, 10.0)
    assert windows[0][0] == 0.0
    assert windows[-1][1] == 10.0
    for (start, end) in windows:
        assert end - start <= 3.0 + 1e-9
        assert end > start


def test_build_spans_merges_same_speaker():
    labeled = [(0.0, 3.0, 0), (3.2, 6.0, 0), (6.0, 8.0, 1), (12.0, 13.0, 0)]
    spans = _build_spans(labeled)
    assert len(spans) == 3
    assert spans[0].start == 0.0 and spans[0].end == 6.0
    assert spans[1].speaker == 1
    assert spans[2].speaker == 0
