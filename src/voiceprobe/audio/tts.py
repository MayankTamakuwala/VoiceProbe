import os
from collections.abc import Iterator
from pathlib import Path

from piper import PiperVoice

from voiceprobe.audio import pcm

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))


class Tts:
    def __init__(self, voice: str | None = None, models_dir: Path = MODELS_DIR):
        name = voice or os.getenv("PIPER_VOICE", "en_US-lessac-medium")
        path = models_dir / f"{name}.onnx"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found, run: python -m piper.download_voices {name} --data-dir {models_dir}"
            )
        self.voice = PiperVoice.load(path)

    def stream(self, text: str) -> Iterator[bytes]:
        # piper yields one chunk per sentence, which lets callers start
        # sending audio before the whole reply is synthesized
        for chunk in self.voice.synthesize(text):
            samples = pcm.resample(chunk.audio_float_array, chunk.sample_rate, pcm.SAMPLE_RATE)
            yield pcm.to_pcm(samples)

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.stream(text))
