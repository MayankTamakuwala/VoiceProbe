import asyncio
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from voiceprobe.agent.dialog import GREETING, Dialog
from voiceprobe.agent.llm import Ollama
from voiceprobe.agent.tools import BookingTools
from voiceprobe.audio import pcm
from voiceprobe.audio.stt import Stt
from voiceprobe.audio.tts import Tts
from voiceprobe.audio.vad import Vad

log = logging.getLogger(__name__)

FRAME_MS = 20
# how far ahead of real time we let outgoing audio run. keeping this small
# means a barge-in actually cuts the agent off instead of the caller
# already having the whole reply buffered
LEAD_S = 0.1
MAX_CALLS_KEPT = 500


class Models:
    def __init__(self, stt=None, tts=None, llm=None, booking=None):
        workers = int(os.getenv("STT_WORKERS", "2"))
        self.stt = stt or Stt(workers=workers)
        self.tts = tts or Tts()
        self.llm = llm or Ollama()
        self.booking = booking or httpx.AsyncClient(
            base_url=os.getenv("BOOKING_URL", "http://127.0.0.1:8001"), timeout=5.0
        )


class Call:
    def __init__(self, ws: WebSocket, models: Models, call_id: str):
        self.ws = ws
        self.models = models
        self.call_id = call_id
        self.vad = Vad(silence_ms=int(os.getenv("VAD_SILENCE_MS", "500")))
        self.tools = BookingTools(call_id, client=models.booking)
        self.dialog = Dialog(models.llm, self.tools)
        self.response: asyncio.Task | None = None
        self.speaking = False
        self.heard: list[str] = []
        self.carry = b""
        self.tools_sent = 0
        self.started = time.time()
        self.barge_ins = 0
        self.closed = False
        self._send_lock = asyncio.Lock()

    async def event(self, kind: str, **data) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps({"type": kind, "t": time.time(), **data}))

    async def send_audio(self, frame: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(frame)

    async def run(self) -> None:
        await self.event("ready", call_id=self.call_id)
        self.response = asyncio.create_task(self._speak(_once(GREETING)))
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    await self.on_audio(msg["bytes"])
                elif msg.get("text") and json.loads(msg["text"]).get("type") == "hangup":
                    break
        finally:
            self.closed = True
            await self.cancel_response()

    async def on_audio(self, data: bytes) -> None:
        for ev in self.vad.push(data):
            if ev.kind == "start":
                await self.event("speech_start")
                if self.response and not self.response.done():
                    await self.barge_in()
            else:
                await self.event("speech_end", duration=round(pcm.duration(ev.audio), 3))
                utterance, self.carry = self.carry + ev.audio, b""
                self.response = asyncio.create_task(self.respond(utterance))

    async def barge_in(self) -> None:
        was_speaking = self.speaking
        heard = " ".join(self.heard)
        await self.cancel_response()
        if was_speaking:
            self.barge_ins += 1
            self.dialog.interrupted(heard)
            await self.event("interrupted", heard=heard)

    async def cancel_response(self) -> None:
        task, self.response = self.response, None
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def respond(self, audio: bytes) -> None:
        # if we get cut off before transcribing, the audio carries over to the next turn
        self.carry = audio
        try:
            t0 = time.perf_counter()
            text = await asyncio.to_thread(self.models.stt.transcribe, audio)
            self.carry = b""
            stt_ms = (time.perf_counter() - t0) * 1000
            await self.event("user_transcript", text=text, stt_ms=round(stt_ms))
            if text:
                await self._speak(self.dialog.reply(text))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("call %s: response failed", self.call_id)
            with suppress(Exception):
                await self.event("error", message=str(e))

    async def _speak(self, sentences: AsyncIterator[str]) -> None:
        self.heard = []
        clock = 0.0
        try:
            async for sentence in sentences:
                await self._flush_tools()
                t0 = time.perf_counter()
                audio = await asyncio.to_thread(self.models.tts.synthesize, sentence)
                tts_ms = (time.perf_counter() - t0) * 1000
                await self.event("agent_text", text=sentence, tts_ms=round(tts_ms))
                if not self.speaking:
                    self.speaking = True
                    await self.event("agent_audio_start")
                clock = await self._play(audio, clock)
                self.heard.append(sentence)
            await self._flush_tools()
            await asyncio.sleep(max(0.0, clock - time.perf_counter()))
        finally:
            with suppress(Exception):
                await sentences.aclose()
            if self.speaking:
                self.speaking = False
                with suppress(Exception):
                    await self.event("agent_audio_end")

    async def _play(self, audio: bytes, clock: float) -> float:
        step = FRAME_MS / 1000
        for frame in pcm.chunk(audio, FRAME_MS):
            now = time.perf_counter()
            clock = max(clock, now)
            if clock - now > LEAD_S:
                await asyncio.sleep(clock - now - LEAD_S)
            await self.send_audio(frame)
            clock += step
        return clock

    async def _flush_tools(self) -> None:
        for rec in self.tools.history[self.tools_sent :]:
            await self.event("tool_call", **asdict(rec))
        self.tools_sent = len(self.tools.history)

    def summary(self) -> dict:
        return {
            "call_id": self.call_id,
            "started": self.started,
            "barge_ins": self.barge_ins,
            "booking_id": self.tools.booking_id,
            "transcript": self.dialog.transcript(),
            "tools": [asdict(r) for r in self.tools.history],
        }


async def _once(text: str) -> AsyncIterator[str]:
    yield text


def create_app(models: Models | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = models or Models()
        app.state.calls = OrderedDict()
        try:
            await app.state.models.llm.warmup()
        except httpx.HTTPError as e:
            # still start, but make it obvious why every call is going to fail
            log.warning("couldn't reach ollama at %s: %r", app.state.models.llm.host, e)
        log.info("agent ready")
        yield
        await app.state.models.booking.aclose()

    app = FastAPI(title="voiceprobe agent", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ok": True, "active": sum(not c.closed for c in app.state.calls.values())}

    @app.get("/calls/{call_id}")
    def get_call(call_id: str):
        call = app.state.calls.get(call_id)
        if not call:
            raise HTTPException(404, "unknown call")
        return call.summary()

    @app.websocket("/call")
    async def call(ws: WebSocket, call_id: str | None = None):
        await ws.accept()
        session = Call(ws, app.state.models, call_id or uuid.uuid4().hex[:8])
        calls = app.state.calls
        calls[session.call_id] = session
        while len(calls) > MAX_CALLS_KEPT:
            calls.popitem(last=False)
        log.info("call %s connected", session.call_id)
        try:
            await session.run()
        except WebSocketDisconnect:
            pass
        finally:
            log.info("call %s ended after %.1fs", session.call_id, time.time() - session.started)

    return app
