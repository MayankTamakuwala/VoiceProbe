import base64
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from voiceprobe.agent.tools import BookingTools
from voiceprobe.audio import pcm

log = logging.getLogger(__name__)


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


ELEVENLABS_API = "https://api.elevenlabs.io"
ELEVENLABS_WS = "wss://api.elevenlabs.io/v1/convai/conversation"
# expressive voices put tags like [happy] in the text, they aren't spoken
_AUDIO_TAG = re.compile(r"\[[a-z ]+\]\s*")


class ElevenLabsAdapter(Adapter):
    """Talks to an ElevenLabs agent. Its booking tools are client tools, so
    they get sent to us over the socket and we run them against the local
    booking api, same as the reference agent would."""

    name = "elevenlabs"
    # it does send an interruption event, but nothing that marks the end of a reply
    has_turn_end = False

    def __init__(
        self,
        agent_id: str | None = None,
        api_key: str | None = None,
        booking_url: str | None = None,
        ws_url: str = ELEVENLABS_WS,
        booking_client: httpx.AsyncClient | None = None,
    ):
        self.agent_id = agent_id or os.environ["ELEVENLABS_AGENT_ID"]
        self.api_key = api_key if api_key is not None else os.getenv("ELEVENLABS_API_KEY")
        self.booking_url = booking_url or os.getenv("BOOKING_URL", "http://127.0.0.1:8001")
        self.ws_url = ws_url
        self.booking_client = booking_client
        self.ws: ClientConnection | None = None
        self.tools: BookingTools | None = None
        self.conversation_id: str | None = None
        self.rate = pcm.SAMPLE_RATE

    async def _signed_url(self) -> str:
        async with httpx.AsyncClient(base_url=ELEVENLABS_API, timeout=10.0) as client:
            r = await client.get(
                "/v1/convai/conversation/get-signed-url",
                params={"agent_id": self.agent_id},
                headers={"xi-api-key": self.api_key},
            )
            r.raise_for_status()
            return r.json()["signed_url"]

    async def connect(self, call_id: str) -> None:
        # private agents need a signed url, public ones just take the id
        url = await self._signed_url() if self.api_key else f"{self.ws_url}?agent_id={self.agent_id}"
        self.ws = await connect(url, max_size=None)
        await self.ws.send(json.dumps({"type": "conversation_initiation_client_data"}))
        self.tools = BookingTools(
            call_id, self.booking_url, client=self.booking_client, move_rebookings=False
        )

    async def send_audio(self, frame: bytes) -> None:
        await self.ws.send(json.dumps({"user_audio_chunk": base64.b64encode(frame).decode()}))

    async def events(self) -> AsyncIterator[AgentEvent]:
        try:
            async for msg in self.ws:
                now = time.perf_counter()
                data = json.loads(msg)
                kind = data.get("type")
                if kind == "audio":
                    audio = base64.b64decode(data["audio_event"]["audio_base_64"])
                    if self.rate != pcm.SAMPLE_RATE:
                        audio = pcm.to_pcm(pcm.resample(pcm.to_float(audio), self.rate, pcm.SAMPLE_RATE))
                    yield AgentEvent("audio", now, audio=audio)
                elif kind == "agent_response":
                    text = _AUDIO_TAG.sub("", data["agent_response_event"]["agent_response"]).strip()
                    yield AgentEvent("agent_text", now, data={"text": text})
                elif kind == "user_transcript":
                    text = data["user_transcription_event"]["user_transcript"]
                    yield AgentEvent("user_transcript", now, data={"text": text})
                elif kind == "interruption":
                    yield AgentEvent("interrupted", now)
                elif kind == "ping":
                    event = data["ping_event"]
                    await self.ws.send(json.dumps({"type": "pong", "event_id": event["event_id"]}))
                elif kind == "client_tool_call":
                    yield await self._run_tool(data["client_tool_call"])
                elif kind == "conversation_initiation_metadata":
                    meta = data["conversation_initiation_metadata_event"]
                    self.conversation_id = meta.get("conversation_id")
                    fmt = meta.get("agent_output_audio_format", "pcm_16000")
                    if not fmt.startswith("pcm_"):
                        raise ValueError(f"agent sends {fmt} audio, set it to pcm_16000")
                    self.rate = int(fmt.split("_")[1])
                    log.info("elevenlabs conversation %s", self.conversation_id)
        except ConnectionClosed:
            return

    async def _run_tool(self, call: dict) -> AgentEvent:
        result = await self.tools.run(call["tool_name"], call.get("parameters") or {})
        await self.ws.send(
            json.dumps(
                {
                    "type": "client_tool_result",
                    "tool_call_id": call["tool_call_id"],
                    "result": json.dumps(result),
                    "is_error": "error" in result,
                }
            )
        )
        return AgentEvent("tool_call", time.perf_counter(), data=asdict(self.tools.history[-1]))

    async def close(self) -> None:
        if self.ws:
            await self.ws.close()
        if self.tools:
            await self.tools.close()
