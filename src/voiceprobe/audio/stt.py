import os

from faster_whisper import WhisperModel

from voiceprobe.audio import pcm
from voiceprobe.audio.tts import MODELS_DIR


class Stt:
    def __init__(self, model: str | None = None, workers: int = 1):
        name = model or os.getenv("WHISPER_MODEL", "base.en")
        # num_workers lets several calls transcribe at once under load
        self.model = WhisperModel(
            name,
            device="cpu",
            compute_type="int8",
            num_workers=workers,
            download_root=str(MODELS_DIR / "whisper"),
        )

    def transcribe(self, audio: bytes) -> str:
        if pcm.duration(audio) < 0.1:
            return ""
        segments, _ = self.model.transcribe(
            pcm.to_float(audio),
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
