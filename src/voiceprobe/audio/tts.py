import os
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from piper import PiperVoice, SynthesisConfig

from voiceprobe.audio import pcm

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

# espeak-ng keeps global state, so phonemizing has to be serialized.
# the onnx part is fine to run from several threads at once.
_espeak_lock = threading.Lock()


class Tts:
    def __init__(self, voice: str | None = None, models_dir: Path = MODELS_DIR):
        name = voice or os.getenv("PIPER_VOICE", "en_US-lessac-medium")
        path = models_dir / f"{name}.onnx"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found, run: python -m piper.download_voices {name} --data-dir {models_dir}"
            )
        self.voice = PiperVoice.load(path)
        self.rate = self.voice.config.sample_rate
        self.config = SynthesisConfig()

    def stream(self, text: str) -> Iterator[bytes]:
        # one chunk per sentence, so callers can start sending audio early
        with _espeak_lock:
            sentences = self.voice.phonemize(text)
        for phonemes in sentences:
            if not phonemes:
                continue
            ids = self.voice.phonemes_to_ids(phonemes)
            audio = self.voice.phoneme_ids_to_audio(ids, self.config)
            peak = np.max(np.abs(audio)) if len(audio) else 0
            if peak > 1e-8:
                audio = audio / peak
            yield pcm.to_pcm(pcm.resample(audio.astype(np.float32), self.rate, pcm.SAMPLE_RATE))

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.stream(text))
