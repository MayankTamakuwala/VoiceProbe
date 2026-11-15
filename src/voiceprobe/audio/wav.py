import wave
from pathlib import Path

from voiceprobe.audio import pcm


def read_wav(path: str | Path) -> bytes:
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != pcm.SAMPLE_WIDTH:
            raise ValueError(f"{path}: only 16-bit wav is supported")
        rate, channels = f.getframerate(), f.getnchannels()
        data = f.readframes(f.getnframes())

    samples = pcm.to_float(data)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return pcm.to_pcm(pcm.resample(samples, rate, pcm.SAMPLE_RATE))


def write_wav(path: str | Path, audio: bytes, rate: int = pcm.SAMPLE_RATE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(pcm.SAMPLE_WIDTH)
        f.setframerate(rate)
        f.writeframes(audio)
