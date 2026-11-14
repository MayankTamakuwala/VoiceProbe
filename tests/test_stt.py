import pytest

from voiceprobe.audio import pcm
from voiceprobe.audio.tts import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not any(MODELS_DIR.glob("*.onnx")), reason="no piper voice downloaded"
)


@pytest.fixture(scope="module")
def stt():
    from voiceprobe.audio.stt import Stt

    return Stt()


def test_short_audio_is_empty(stt):
    assert stt.transcribe(pcm.silence(50)) == ""


def test_roundtrip_through_tts(stt):
    from voiceprobe.audio.tts import Tts

    audio = Tts().synthesize("I would like to book Tuesday at three.")
    text = stt.transcribe(audio).lower()
    assert "tuesday" in text
    assert "book" in text
