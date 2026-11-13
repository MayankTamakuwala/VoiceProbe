from collections.abc import Iterator

import numpy as np

# everything on the wire is 16 kHz mono s16le
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2


def to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def to_pcm(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(samples) == 0:
        return samples
    n_out = int(round(len(samples) * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def frame_bytes(frame_ms: int, rate: int = SAMPLE_RATE) -> int:
    return rate * frame_ms // 1000 * SAMPLE_WIDTH


def chunk(pcm: bytes, frame_ms: int = 20, rate: int = SAMPLE_RATE) -> Iterator[bytes]:
    size = frame_bytes(frame_ms, rate)
    for i in range(0, len(pcm), size):
        frame = pcm[i : i + size]
        if len(frame) < size:
            frame += b"\x00" * (size - len(frame))
        yield frame


def duration(pcm: bytes, rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / SAMPLE_WIDTH / rate


def silence(ms: int, rate: int = SAMPLE_RATE) -> bytes:
    return b"\x00" * frame_bytes(ms, rate)
