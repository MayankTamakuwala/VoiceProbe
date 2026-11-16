import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed


@dataclass
class AgentEvent:
    # audio, agent_text, user_transcript, tool_call, interrupted, turn_end, error
    kind: str
    t: float
    audio: bytes = b""
    data: dict = field(default_factory=dict)


class Adapter:
    name = "base"
    # true if the agent tells us when it's done talking. otherwise the runner
    # waits for a gap in the audio, which is slower and less precise
    has_turn_end = False

    async def connect(self, call_id: str) -> None:
        raise NotImplementedError

    async def send_audio(self, frame: bytes) -> None:
        raise NotImplementedError

    def events(self) -> AsyncIterator[AgentEvent]:
        raise NotImplementedError

    async def close(self) -> None:
        pass


_REFERENCE_KINDS = {
    "agent_text": "agent_text",
    "user_transcript": "user_transcript",
    "tool_call": "tool_call",
    "interrupted": "interrupted",
    "agent_audio_end": "turn_end",
    "error": "error",
}


class ReferenceAdapter(Adapter):
    name = "reference"
    has_turn_end = True

    def __init__(self, url: str = "ws://127.0.0.1:8000/call"):
        self.url = url
        self.ws: ClientConnection | None = None

    async def connect(self, call_id: str) -> None:
        self.ws = await connect(f"{self.url}?call_id={call_id}", max_size=None)

    async def send_audio(self, frame: bytes) -> None:
        await self.ws.send(frame)

    async def events(self) -> AsyncIterator[AgentEvent]:
        try:
            async for msg in self.ws:
                now = time.perf_counter()
                if isinstance(msg, bytes):
                    yield AgentEvent("audio", now, audio=msg)
                    continue
                data = json.loads(msg)
                kind = _REFERENCE_KINDS.get(data.get("type"))
                if kind:
                    yield AgentEvent(kind, now, data=data)
        except ConnectionClosed:
            return

    async def close(self) -> None:
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "hangup"}))
            except ConnectionClosed:
                pass
            await self.ws.close()
