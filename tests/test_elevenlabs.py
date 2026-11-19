import asyncio
import base64
import json

import httpx
import numpy as np
from websockets.asyncio.server import serve

from voiceprobe.agent.tools import TOOLS
from voiceprobe.audio import pcm
from voiceprobe.booking.api import create_app
from voiceprobe.probe.adapters import ElevenLabsAdapter
from voiceprobe.probe.elevenlabs import agent_config, tool_config


def test_tool_config():
    by_name = {t["function"]["name"]: tool_config(t)["tool_config"] for t in TOOLS}
    book = by_name["book_appointment"]
    assert book["type"] == "client" and book["expects_response"]
    assert book["parameters"]["required"] == ["name", "day", "time"]
    # no arguments, so no schema at all
    assert "parameters" not in by_name["cancel_appointment"]


def test_agent_config():
    cfg = agent_config(["t1", "t2"], "gemini-2.5-flash", 150, "test")
    conv = cfg["conversation_config"]
    assert conv["agent"]["prompt"]["tool_ids"] == ["t1", "t2"]
    assert conv["agent"]["first_message"].startswith("Thanks for calling Maple Dental")
    assert conv["tts"]["agent_output_audio_format"] == "pcm_16000"
    assert "client_tool_call" in conv["conversation"]["client_events"]


def send(ws, kind, **body):
    return ws.send(json.dumps({"type": kind, **body}))


async def fake_agent(ws, seen):
    seen.append(json.loads(await ws.recv()))
    meta = {"conversation_id": "conv1", "agent_output_audio_format": "pcm_8000"}
    await send(ws, "conversation_initiation_metadata", conversation_initiation_metadata_event=meta)
    await send(ws, "ping", ping_event={"event_id": 7, "ping_ms": 20})
    await send(ws, "agent_response", agent_response_event={"agent_response": "[happy] Hi, Maple Dental."})
    # one second at 8k
    tone = pcm.to_pcm((0.3 * np.sin(np.arange(8000) / 5)).astype(np.float32))
    await send(ws, "audio", audio_event={"audio_base_64": base64.b64encode(tone).decode(), "event_id": 1})

    while not ({"pong", "audio"} <= {m.get("type", "audio") for m in seen[1:]}):
        seen.append(json.loads(await ws.recv()))

    args = {"name": "Priya Shah", "day": "tuesday", "time": "15:00"}
    call = {"tool_name": "book_appointment", "tool_call_id": "tc1", "parameters": args, "expects_response": True}
    await send(ws, "client_tool_call", client_tool_call=call)
    while seen[-1].get("type") != "client_tool_result":
        seen.append(json.loads(await ws.recv()))

    await send(ws, "user_transcript", user_transcription_event={"user_transcript": "Tuesday at three"})
    await send(ws, "interruption", interruption_event={"event_id": 2})
    await ws.close()


async def test_elevenlabs_adapter(tmp_path):
    seen = []
    app = create_app(str(tmp_path / "b.db"))
    booking = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://booking")
    async with serve(lambda ws: fake_agent(ws, seen), "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        adapter = ElevenLabsAdapter(
            "agent1", api_key="", ws_url=f"ws://127.0.0.1:{port}", booking_client=booking
        )
        await adapter.connect("call-1")
        await adapter.send_audio(pcm.silence(20))
        events = []
        async with asyncio.timeout(5):
            async for ev in adapter.events():
                events.append(ev)
        await adapter.close()

    assert seen[0] == {"type": "conversation_initiation_client_data"}
    assert {"type": "pong", "event_id": 7} in seen
    assert len(base64.b64decode(next(m for m in seen if "user_audio_chunk" in m)["user_audio_chunk"])) == 640

    kinds = [e.kind for e in events]
    assert kinds == ["agent_text", "audio", "tool_call", "user_transcript", "interrupted"]
    assert events[0].data["text"] == "Hi, Maple Dental."
    # 8k audio comes out at 16k, still one second long
    assert abs(pcm.duration(events[1].audio) - 1.0) < 0.01
    assert events[2].data["name"] == "book_appointment"
    assert events[2].data["result"]["status"] == "confirmed"

    result = seen[-1]
    assert result["tool_call_id"] == "tc1" and result["is_error"] is False
    assert json.loads(result["result"])["day"] == "tuesday"
    assert [b.name for b in app.state.store.bookings()] == ["Priya Shah"]
    assert adapter.conversation_id == "conv1"
