from voiceprobe.agent.text import SentenceBuffer


def stream(text, step=3):
    buf = SentenceBuffer()
    out = []
    for i in range(0, len(text), step):
        out.extend(buf.push(text[i : i + step]))
    tail = buf.flush()
    return out + ([tail] if tail else [])


def test_splits_sentences():
    assert stream("Sure. Tuesday at three works! Anything else?") == [
        "Sure.",
        "Tuesday at three works!",
        "Anything else?",
    ]


def test_holds_until_boundary():
    buf = SentenceBuffer()
    assert buf.push("Let me check") == []
    assert buf.push(" that. One") == ["Let me check that."]
    assert buf.flush() == "One"


def test_does_not_split_on_am_pm():
    assert stream("You're booked for 3 p.m. on Tuesday. Bye.") == [
        "You're booked for 3 p.m. on Tuesday.",
        "Bye.",
    ]


def test_strips_markdown():
    assert stream("**Great**, you're all set.") == ["Great, you're all set."]
