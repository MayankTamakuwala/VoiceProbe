import numpy as np

from voiceprobe.audio import pcm


def test_roundtrip():
    samples = np.array([0.0, 0.5, -0.5, 0.99], dtype=np.float32)
    back = pcm.to_float(pcm.to_pcm(samples))
    assert np.allclose(samples, back, atol=1e-4)


def test_to_pcm_clips():
    out = pcm.to_float(pcm.to_pcm(np.array([2.0, -2.0], dtype=np.float32)))
    assert out[0] > 0.99
    assert out[1] < -0.99


def test_resample_length():
    samples = np.zeros(22050, dtype=np.float32)
    assert len(pcm.resample(samples, 22050, 16000)) == 16000


def test_chunk_pads_last_frame():
    data = pcm.silence(50)
    frames = list(pcm.chunk(data, frame_ms=20))
    assert len(frames) == 3
    assert all(len(f) == 640 for f in frames)


def test_duration():
    assert pcm.duration(pcm.silence(1000)) == 1.0
