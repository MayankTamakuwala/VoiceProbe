import pytest

from voiceprobe.audio import pcm
from voiceprobe.audio.tts import MODELS_DIR, Tts

pytestmark = pytest.mark.skipif(
    not any(MODELS_DIR.glob("*.onnx")), reason="no piper voice downloaded"
)


@pytest.fixture(scope="module")
def tts():
    return Tts()


def test_synthesize_returns_16k_pcm(tts):
    audio = tts.synthesize("Hello, I'd like to book an appointment.")
    assert len(audio) % pcm.SAMPLE_WIDTH == 0
    assert 1.0 < pcm.duration(audio) < 6.0


def test_stream_splits_sentences(tts):
    chunks = list(tts.stream("Hi there. Is Tuesday at three free?"))
    assert len(chunks) == 2
