from voiceprobe.probe.wer import distance, normalize, wer


def test_normalize_times():
    assert normalize("Monday at 9am.") == normalize("monday at 9 a.m.")
    assert normalize("Ten o'clock") == ["10", "o'clock"]
    assert normalize("15:00") == ["15"]


def test_distance():
    assert distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert distance(["a", "b", "c"], ["a", "c"]) == 1
    assert distance(["a", "b"], ["a", "x", "b"]) == 1
    assert distance([], ["a"]) == 1


def test_wer():
    assert wer([("My name is Lena Park.", "My name is Leena Park.")]) == 0.2
    # one error over eight words, not the average of the two turns
    pairs = [("Hi.", "Hi."), ("Can I get Friday at ten please", "can I get friday at 10 peas")]
    assert abs(wer(pairs) - 1 / 8) < 1e-9
    assert wer([]) is None
