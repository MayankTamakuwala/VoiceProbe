import asyncio
import json
import threading
import time

import httpx
import pytest
import uvicorn
from websockets.asyncio.client import connect

from voiceprobe.agent.server import Models, create_app
from voiceprobe.audio import pcm
from voiceprobe.audio.tts import MODELS_DIR
from voiceprobe.booking.api import create_app as create_booking_app

pytestmark = pytest.mark.skipif(
    not any(MODELS_DIR.glob("*.onnx")), reason="no piper voice downloaded"
)


class ScriptedLlm:
    def __init__(self, turns):
        self.turns = list(turns)

    async def chat(self, messages, tools=None):
        for part in self.turns.pop(0) if self.turns else ["Sorry?"]:
            await asyncio.sleep(0.01)
            yield part

    async def warmup(self):
        pass


@pytest.fixture(scope="module")
def speech_models():
    from voiceprobe.audio.stt import Stt
    from voiceprobe.audio.tts import Tts

    return Stt(), Tts()


@pytest.fixture
def start_agent(tmp_path, speech_models):
    servers = []

    def start(turns):
        stt, tts = speech_models
        booking = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_booking_app(str(tmp_path / "b.db"))),
            base_url="http://booking",
        )
        models = Models(stt=stt, tts=tts, llm=ScriptedLlm(turns), booking=booking)
        server = uvicorn.Server(
            uvicorn.Config(create_app(models), host="127.0.0.1", port=0, log_level="warning")
        )
        threading.Thread(target=server.run, daemon=True).start()
        while not server.started:
            time.sleep(0.05)
        servers.append(server)
        port = server.servers[0].sockets[0].getsockname()[1]
        return f"127.0.0.1:{port}"

    yield start
    for s in servers:
        s.should_exit = True


async def stream(ws, audio):
    # real time, like a phone line
    for frame in pcm.chunk(audio, 20):
        await ws.send(frame)
        await asyncio.sleep(0.02)


async def read_until(ws, kind, events, timeout=15):
    async with asyncio.timeout(timeout):
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                continue
            ev = json.loads(msg)
            events.append(ev)
            if ev["type"] == kind:
                return ev


async def test_greeting_then_turn(start_agent, speech_models):
    host = start_agent([["Monday at nine is free."]])
    _, tts = speech_models
    question = tts.synthesize("Do you have anything free on Monday?")
    events = []
    async with connect(f"ws://{host}/call?call_id=t1") as ws:
        await read_until(ws, "agent_audio_end", events)
        sender = asyncio.create_task(stream(ws, question + pcm.silence(1000)))
        said = await read_until(ws, "user_transcript", events)
        await read_until(ws, "agent_audio_end", events)
        await sender

    kinds = [e["type"] for e in events]
    assert kinds[0] == "ready"
    assert "speech_start" in kinds and "speech_end" in kinds
    assert "monday" in said["text"].lower()
    assert any(e["type"] == "agent_text" and "nine" in e["text"] for e in events)

    call = httpx.get(f"http://{host}/calls/t1").json()
    assert call["transcript"][-1]["content"] == "Monday at nine is free."


async def test_barge_in_stops_agent(start_agent, speech_models):
    long_reply = [
        "We have lots of openings this week. ",
        "Monday has nine, ten and eleven. ",
        "Tuesday has the whole afternoon free. ",
        "Wednesday is wide open too. ",
        "Thursday is mostly booked but we have four. ",
    ]
    host = start_agent([long_reply])
    _, tts = speech_models
    events = []
    async with connect(f"ws://{host}/call?call_id=t2") as ws:
        await read_until(ws, "agent_audio_end", events)
        await stream(ws, tts.synthesize("What's open this week?") + pcm.silence(700))
        await read_until(ws, "agent_audio_start", events)
        # let the first sentence play out, then talk over the second
        await asyncio.sleep(3.0)
        interrupt = asyncio.create_task(stream(ws, tts.synthesize("Wait, sorry.") + pcm.silence(200)))
        ev = await read_until(ws, "interrupted", events)
        await interrupt

    kinds = [e["type"] for e in events]
    # audio stops before the interrupted event goes out
    assert kinds[-2:] == ["agent_audio_end", "interrupted"]
    assert ev["heard"].startswith("We have lots of openings")
    spoken = [e["text"] for e in events if e["type"] == "agent_text"]
    assert len(spoken) < len(long_reply)
    call = httpx.get(f"http://{host}/calls/t2").json()
    assert call["barge_ins"] == 1
    assert call["transcript"][-1]["content"].endswith("...")
