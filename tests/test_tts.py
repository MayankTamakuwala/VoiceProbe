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


def test_parallel_synthesis(tts):
    from concurrent.futures import ThreadPoolExecutor

    texts = [f"Your appointment number {i} is confirmed." for i in range(8)]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(tts.synthesize, texts))
    assert all(pcm.duration(r) > 1.0 for r in results)
