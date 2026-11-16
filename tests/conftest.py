import asyncio
import threading
import time

import httpx
import pytest
import uvicorn

from voiceprobe.audio.tts import MODELS_DIR


class ScriptedLlm:
    """Stands in for ollama on the agent side, plays back canned turns."""

    def __init__(self, turns):
        self.turns = list(turns)

    async def chat(self, messages, tools=None):
        for part in self.turns.pop(0) if self.turns else ["Sorry?"]:
            await asyncio.sleep(0.01)
            yield part

    async def warmup(self):
        pass


@pytest.fixture(scope="session")
def speech_models():
    if not any(MODELS_DIR.glob("*.onnx")):
        pytest.skip("no piper voice downloaded")
    from voiceprobe.audio.stt import Stt
    from voiceprobe.audio.tts import Tts

    return Stt(), Tts()


@pytest.fixture
def start_agent(tmp_path, speech_models):
    from voiceprobe.agent.server import Models, create_app
    from voiceprobe.booking.api import create_app as create_booking_app

    servers = []

    def start(turns, booking_app=None):
        stt, tts = speech_models
        booking_app = booking_app or create_booking_app(str(tmp_path / "b.db"))
        booking = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=booking_app), base_url="http://booking"
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
