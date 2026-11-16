"""Quick manual caller: speaks a few fixed lines at the agent and prints what happens."""

import asyncio
import json
import time

from websockets.asyncio.client import connect

from voiceprobe.audio import pcm
from voiceprobe.audio.tts import Tts


class Line:
    """Full duplex phone line: sends queued speech, silence otherwise."""

    def __init__(self, ws):
        self.ws = ws
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.speech_done = asyncio.Event()
        self.last_speech_at = 0.0

    def say(self, audio: bytes) -> None:
        self.speech_done.clear()
        for frame in pcm.chunk(audio, 20):
            self.queue.put_nowait(frame)

    async def run(self) -> None:
        silence = pcm.silence(20)
        clock = time.perf_counter()
        while True:
            try:
                frame = self.queue.get_nowait()
                speaking = True
            except asyncio.QueueEmpty:
                frame, speaking = silence, False
            await self.ws.send(frame)
            if speaking and self.queue.empty():
                self.last_speech_at = time.perf_counter()
                self.speech_done.set()
            clock += 0.02
            await asyncio.sleep(max(0.0, clock - time.perf_counter()))


async def dial(url: str, lines: list[str], out: str | None = None) -> list[float]:
    tts = Tts()
    agent_audio = bytearray()
    latencies: list[float] = []
    turn_done = asyncio.Event()
    waiting_since: float | None = None

    async with connect(url) as ws:
        line = Line(ws)
        sender = asyncio.create_task(line.run())

        async def reader():
            nonlocal waiting_since
            async for msg in ws:
                if isinstance(msg, bytes):
                    agent_audio.extend(msg)
                    if waiting_since is not None:
                        latencies.append(time.perf_counter() - waiting_since)
                        print(f"  first audio after {latencies[-1] * 1000:.0f}ms")
                        waiting_since = None
                    continue
                ev = json.loads(msg)
                kind = ev["type"]
                if kind == "agent_text":
                    print(f"agent> {ev['text']}")
                elif kind == "user_transcript":
                    print(f"  heard: {ev['text']!r} (stt {ev['stt_ms']}ms)")
                elif kind == "tool_call":
                    print(f"  tool {ev['name']} {ev['arguments']} -> {ev['result']}")
                elif kind in ("interrupted", "error"):
                    print(f"  {kind}: {ev}")
                elif kind == "agent_audio_end":
                    turn_done.set()

        reading = asyncio.create_task(reader())
        await asyncio.wait_for(turn_done.wait(), 30)

        for text in lines:
            turn_done.clear()
            print(f"you> {text}")
            line.say(tts.synthesize(text))
            await line.speech_done.wait()
            waiting_since = line.last_speech_at
            try:
                await asyncio.wait_for(turn_done.wait(), 60)
            except TimeoutError:
                print("  (no reply)")
            await asyncio.sleep(0.5)

        sender.cancel()
        reading.cancel()

    if out:
        from voiceprobe.audio.wav import write_wav

        write_wav(out, bytes(agent_audio))
        print(f"saved agent audio to {out}")
    return latencies
