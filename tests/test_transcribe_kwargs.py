"""transcribe_file が faster-whisper へ渡す引数の検証（モデル不要）。"""

import pytest

from koemoji import core
from koemoji.core import TranscribeOptions, transcribe_file


class _Seg:
    def __init__(self, start, end, text, no_speech_prob=0.0, avg_logprob=0.0):
        self.start = start
        self.end = end
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


class _Info:
    duration = 2.0
    language = "ja"
    language_probability = 1.0


class _FakeModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        return [_Seg(0.0, 1.0, "こんにちは。")], _Info()


@pytest.fixture
def fake(monkeypatch, tmp_path):
    model = _FakeModel()
    monkeypatch.setattr(core, "load_model", lambda *a, **k: model)
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    return model, audio


def test_temperature_not_passed_when_unset(fake):
    model, audio = fake
    transcribe_file(audio, TranscribeOptions())
    # temperature=None を渡すと faster-whisper が [None] として扱い落ちる
    assert "temperature" not in model.kwargs


def test_temperature_passed_when_set(fake):
    model, audio = fake
    transcribe_file(audio, TranscribeOptions(temperature=0.2))
    assert model.kwargs["temperature"] == 0.2


def test_thresholds_forwarded(fake):
    model, audio = fake
    transcribe_file(
        audio,
        TranscribeOptions(no_speech_threshold=0.5, log_prob_threshold=-0.8, hotwords="寿司"),
    )
    assert model.kwargs["no_speech_threshold"] == 0.5
    assert model.kwargs["log_prob_threshold"] == -0.8
    assert model.kwargs["hotwords"] == "寿司"


def test_hallucination_dropped_and_repeats_collapsed(fake):
    model, audio = fake

    def transcribe(path, **kwargs):
        return [
            _Seg(0.0, 1.0, "ありがとうございます。"),
            _Seg(1.0, 2.0, "ありがとうございます。"),
            _Seg(2.0, 3.0, "ご視聴ありがとうございました。", no_speech_prob=0.95, avg_logprob=-1.5),
        ], _Info()

    model.transcribe = transcribe
    result = transcribe_file(audio, TranscribeOptions(drop_hallucinations=True))
    assert [s.text for s in result.segments] == ["ありがとうございます。"]


def test_hallucinations_kept_when_disabled(fake):
    model, audio = fake

    def transcribe(path, **kwargs):
        return [_Seg(0.0, 1.0, "ご視聴ありがとうございました。", 0.95, -1.5)], _Info()

    model.transcribe = transcribe
    result = transcribe_file(audio, TranscribeOptions(drop_hallucinations=False))
    assert len(result.segments) == 1
