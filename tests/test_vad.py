import pytest

from voiceprobe.audio import pcm
from voiceprobe.audio.tts import MODELS_DIR
from voiceprobe.audio.vad import Vad

needs_voice = pytest.mark.skipif(
    not any(MODELS_DIR.glob("*.onnx")), reason="no piper voice downloaded"
)


@pytest.fixture(scope="module")
def speech():
    from voiceprobe.audio.tts import Tts

    return Tts().synthesize("Hi, I'd like to book a table for Tuesday.")


def feed(vad, audio):
    events = []
    for frame in pcm.chunk(audio, frame_ms=20):
        events.extend(vad.push(frame))
    return events


def test_silence_has_no_events():
    assert feed(Vad(), pcm.silence(2000)) == []


@needs_voice
def test_detects_start_and_end(speech):
    events = feed(Vad(), pcm.silence(500) + speech + pcm.silence(1000))
    assert [e.kind for e in events] == ["start", "end"]
    # utterance should cover the speech plus the trailing silence window
    assert pcm.duration(events[1].audio) >= pcm.duration(speech) * 0.8


@needs_voice
def test_no_end_while_still_talking(speech):
    vad = Vad()
    events = feed(vad, pcm.silence(300) + speech)
    assert [e.kind for e in events] == ["start"]
    assert vad.speaking


@needs_voice
def test_reset_clears_state(speech):
    vad = Vad()
    feed(vad, pcm.silence(300) + speech)
    vad.reset()
    assert not vad.speaking
    assert feed(vad, pcm.silence(1000)) == []
