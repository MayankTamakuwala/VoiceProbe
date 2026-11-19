import asyncio
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field

import numpy as np

from voiceprobe.audio import pcm
from voiceprobe.probe.adapters import Adapter
from voiceprobe.probe.caller import Caller
from voiceprobe.probe.scenario import Scenario

log = logging.getLogger(__name__)

FRAME_MS = 20


@dataclass
class CallerTurn:
    text: str
    kind: str = "turn"
    start: float | None = None
    end: float | None = None
    heard_as: str | None = None
    first_audio: float | None = None


@dataclass
class AgentTurn:
    index: int
    start: float
    end: float | None = None
    text: list[str] = field(default_factory=list)
    interrupted_at: float | None = None
    stopped_ms: float | None = None
    flagged: bool = False


@dataclass
class CallRecord:
    scenario: str
    call_id: str
    agent: str
    started_at: float
    duration: float = 0.0
    ended_by: str = ""
    caller_turns: list[CallerTurn] = field(default_factory=list)
    agent_turns: list[AgentTurn] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def trim(audio: bytes, threshold: float = 0.02) -> bytes:
    # piper pads its output with a bit of silence, which would make the agent
    # look slower than it is when we time from the end of our audio
    x = pcm.to_float(audio)
    loud = np.flatnonzero(np.abs(x) > threshold)
    if len(loud) == 0:
        return audio
    pad = pcm.SAMPLE_RATE // 100
    start, end = max(0, loud[0] - pad), min(len(x), loud[-1] + pad)
    return audio[start * pcm.SAMPLE_WIDTH : end * pcm.SAMPLE_WIDTH]


class CallRunner:
    def __init__(
        self,
        scenario: Scenario,
        adapter: Adapter,
        caller: Caller,
        tts,
        stt=None,
        hear: str = "text",
        reply_timeout: float = 30.0,
        greeting_timeout: float = 10.0,
        gap_s: float = 1.5,
    ):
        self.scenario = scenario
        self.adapter = adapter
        self.caller = caller
        self.tts = tts
        self.stt = stt
        self.hear = hear
        self.reply_timeout = reply_timeout
        self.greeting_timeout = greeting_timeout
        self.gap_s = gap_s

        self.out: asyncio.Queue = asyncio.Queue()
        self.pending = list(scenario.interrupts)
        self.agent: AgentTurn | None = None
        self.agent_said: list[str] = []
        self.early_text: list[str] = []
        self.agent_audio = bytearray()
        self.playout_end = 0.0
        self.turn_end_flag = False
        self.caller_speaking = False
        self.awaiting_reply = True
        self.last_caller: CallerTurn | None = None
        self.barge_turns: dict[int, CallerTurn] = {}
        self.closed = False

    def now(self) -> float:
        return time.perf_counter() - self.t0

    async def run(self, call_id: str | None = None) -> CallRecord:
        call_id = call_id or f"{self.scenario.id}-{uuid.uuid4().hex[:6]}"
        self.record = CallRecord(self.scenario.id, call_id, self.adapter.name, time.time())
        self.t0 = time.perf_counter()
        await self.adapter.connect(call_id)
        tasks = [
            asyncio.create_task(self._send_loop()),
            asyncio.create_task(self._recv_loop()),
            asyncio.create_task(self._watch_loop()),
        ]
        try:
            self.record.ended_by = await self._converse()
        except Exception as e:
            log.exception("call %s failed", call_id)
            self.record.errors.append(f"{type(e).__name__}: {e}")
            self.record.ended_by = "error"
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with suppress(Exception):
                await self.adapter.close()
            self.record.duration = round(self.now(), 3)
            if self.agent and self.agent.end is None:
                self.agent.end = round(min(self.playout_end, self.record.duration), 3)
        return self.record

    async def _converse(self) -> str:
        if not await self._wait_ready(self.greeting_timeout):
            if self.record.agent_turns:
                return "no_reply"
            # no greeting, caller goes first
            self.awaiting_reply = False

        while True:
            if self.closed:
                return "agent_hangup"
            if self.caller.done:
                return "caller"
            heard = await self._hear()
            line = await self.caller.next_line(heard)
            if not line:
                return "caller"
            turn = CallerTurn(line)
            self.record.caller_turns.append(turn)
            await self._say(turn)
            if not await self._wait_ready(self.reply_timeout):
                return "no_reply"

    def _ready(self) -> bool:
        return (
            self.agent is None
            and not self.caller_speaking
            and not self.awaiting_reply
            and self.out.empty()
        )

    async def _wait_ready(self, timeout: float) -> bool:
        deadline = time.perf_counter() + timeout
        while True:
            await asyncio.sleep(0.05)
            if self.closed or self._ready():
                return True
            if self.agent is not None or self.caller_speaking:
                # someone is still talking, only give up once the line goes quiet
                deadline = time.perf_counter() + self.reply_timeout
            elif time.perf_counter() > deadline:
                return False

    async def _hear(self) -> str:
        if self.hear == "audio" and self.stt and self.agent_audio:
            text = await asyncio.to_thread(self.stt.transcribe, bytes(self.agent_audio))
        else:
            text = " ".join(self.agent_said)
        self.agent_said.clear()
        self.agent_audio.clear()
        return text

    async def _say(self, turn: CallerTurn) -> None:
        audio = trim(await asyncio.to_thread(self.tts.synthesize, turn.text))
        frames = list(pcm.chunk(audio, FRAME_MS))
        for i, frame in enumerate(frames):
            self.out.put_nowait((frame, turn, i == 0, i == len(frames) - 1))

    async def _send_loop(self) -> None:
        silence = pcm.silence(FRAME_MS)
        step = FRAME_MS / 1000
        clock = time.perf_counter()
        while True:
            try:
                frame, turn, first, last = self.out.get_nowait()
            except asyncio.QueueEmpty:
                frame, turn, first, last = silence, None, False, False
            if first:
                turn.start = round(self.now(), 3)
                self.caller_speaking = True
            await self.adapter.send_audio(frame)
            if last:
                turn.end = round(self.now(), 3)
                self.caller_speaking = False
                self.awaiting_reply = True
                self.last_caller = turn
            clock += step
            await asyncio.sleep(max(0.0, clock - time.perf_counter()))

    async def _recv_loop(self) -> None:
        async for ev in self.adapter.events():
            now = ev.t - self.t0
            if ev.kind == "audio":
                self._on_audio(ev.audio, now)
            elif ev.kind == "agent_text":
                self.agent_said.append(ev.data["text"])
                # text usually shows up just before its audio
                (self.agent.text if self.agent else self.early_text).append(ev.data["text"])
            elif ev.kind == "user_transcript":
                self._on_transcript(ev.data.get("text", ""))
            elif ev.kind == "tool_call":
                d = ev.data
                self.record.tool_calls.append(
                    {
                        "t": round(now, 3),
                        "name": d.get("name"),
                        "arguments": d.get("arguments", {}),
                        "result": d.get("result"),
                        "ms": d.get("ms"),
                        "attempts": d.get("attempts"),
                    }
                )
            elif ev.kind == "interrupted":
                if self.agent:
                    self.agent.flagged = True
                # a real client drops whatever it still had buffered
                self.playout_end = min(self.playout_end, now)
            elif ev.kind == "turn_end":
                self.turn_end_flag = True
            elif ev.kind == "error":
                self.record.errors.append(str(ev.data.get("message", ev.data)))
        self.closed = True

    def _on_audio(self, frame: bytes, now: float) -> None:
        if self.agent is None:
            self.agent = AgentTurn(len(self.record.agent_turns), round(max(now, self.playout_end), 3))
            self.record.agent_turns.append(self.agent)
            self.agent.text, self.early_text = self.early_text, []
            self.turn_end_flag = False
            if self.awaiting_reply:
                if self.last_caller and self.last_caller.first_audio is None:
                    self.last_caller.first_audio = round(now, 3)
                self.awaiting_reply = False
        self.playout_end = max(self.playout_end, now) + pcm.duration(frame)
        self.agent_audio.extend(frame)

    def _on_transcript(self, text: str) -> None:
        ended = [t for t in self.record.caller_turns if t.end is not None]
        if not ended:
            return
        turn = ended[-1]
        turn.heard_as = f"{turn.heard_as} {text}".strip() if turn.heard_as else text

    async def _watch_loop(self) -> None:
        while True:
            await asyncio.sleep(FRAME_MS / 1000)
            a = self.agent
            if a is None:
                continue
            now = self.now()
            if now < self.playout_end:
                await self._maybe_interrupt(a, now)
                continue
            quiet = now - self.playout_end
            limit = 5.0 if self.adapter.has_turn_end else self.gap_s
            if self.turn_end_flag or quiet >= limit:
                a.end = round(self.playout_end, 3)
                barge = self.barge_turns.get(a.index)
                if barge and barge.start is not None:
                    # time from when our audio actually started, not when we decided to talk
                    a.interrupted_at = barge.start
                if a.interrupted_at is not None:
                    a.stopped_ms = round(max(0.0, a.end - a.interrupted_at) * 1000)
                self.agent = None
                self.turn_end_flag = False

    async def _maybe_interrupt(self, a: AgentTurn, now: float) -> None:
        if not self.pending or self.caller_speaking or a.interrupted_at is not None:
            return
        it = self.pending[0]
        if a.index < it.after_turn or (now - a.start) * 1000 < it.after_ms:
            return
        self.pending.pop(0)
        turn = CallerTurn(it.say, kind="interrupt")
        self.record.caller_turns.append(turn)
        self.caller.interrupted(" ".join(self.agent_said), it.say)
        self.agent_said.clear()
        self.agent_audio.clear()
        a.interrupted_at = round(now, 3)
        self.barge_turns[a.index] = turn
        await self._say(turn)
        log.info("barge-in on agent turn %d at %.2fs", a.index, now)
