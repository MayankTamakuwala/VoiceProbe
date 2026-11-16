import asyncio
import json

import httpx
from websockets.asyncio.client import connect

from voiceprobe.audio import pcm


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
