import os
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from faster_whisper.utils import get_assets_path

from voiceprobe.audio import pcm

# silero works on 512 sample windows at 16k, plus 64 samples of context
WINDOW = 512
CONTEXT = 64
WINDOW_MS = WINDOW * 1000 // pcm.SAMPLE_RATE


@lru_cache
def _session() -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 4
    # faster-whisper ships the silero onnx model, no need for torch
    path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)


@dataclass
class VadEvent:
    kind: str
    audio: bytes = b""


def _windows(ms: int) -> int:
    return max(1, -(-ms // WINDOW_MS))


class Vad:
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 96,
        silence_ms: int = 500,
        preroll_ms: int = 200,
    ):
        self.threshold = threshold
        self.off_threshold = threshold - 0.15
        self.min_speech = _windows(min_speech_ms)
        self.min_silence = _windows(silence_ms)
        self.preroll = _windows(preroll_ms) + self.min_speech
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(CONTEXT, dtype=np.float32)
        self._buf = b""
        self._pre: deque[bytes] = deque(maxlen=self.preroll)
        self._utterance: list[bytes] = []
        self._voiced = 0
        self._silent = 0
        self.speaking = False

    def prob(self, window: np.ndarray) -> float:
        x = np.concatenate([self._ctx, window])[None]
        out, self._h, self._c = _session().run(None, {"input": x, "h": self._h, "c": self._c})
        self._ctx = window[-CONTEXT:]
        return float(out.ravel()[0])

    def push(self, audio: bytes) -> list[VadEvent]:
        self._buf += audio
        size = WINDOW * pcm.SAMPLE_WIDTH
        events = []
        while len(self._buf) >= size:
            chunk, self._buf = self._buf[:size], self._buf[size:]
            event = self._step(chunk, self.prob(pcm.to_float(chunk)))
            if event:
                events.append(event)
        return events

    def _step(self, chunk: bytes, p: float) -> VadEvent | None:
        if not self.speaking:
            self._pre.append(chunk)
            self._voiced = self._voiced + 1 if p >= self.threshold else 0
            if self._voiced >= self.min_speech:
                self.speaking = True
                self._silent = 0
                self._utterance = list(self._pre)
                self._pre.clear()
                return VadEvent("start")
            return None

        self._utterance.append(chunk)
        self._silent = self._silent + 1 if p < self.off_threshold else 0
        if self._silent >= self.min_silence:
            self.speaking = False
            self._voiced = 0
            audio = b"".join(self._utterance)
            self._utterance = []
            return VadEvent("end", audio)
        return None
