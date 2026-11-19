import numpy as np

from voiceprobe.agent.llm import ToolCall
from voiceprobe.audio import pcm
from voiceprobe.probe.adapters import ReferenceAdapter
from voiceprobe.probe.caller import Caller
from voiceprobe.probe.runner import CallRunner, trim
from voiceprobe.probe.scenario import parse


class FakeCallerLlm:
    def __init__(self, lines):
        self.lines = list(lines)

    async def complete(self, messages):
        return self.lines.pop(0) if self.lines else "[END]"


def test_trim_drops_padding():
    tone = pcm.to_pcm(0.5 * np.sin(np.linspace(0, 400, 8000)).astype(np.float32))
    padded = pcm.silence(300) + tone + pcm.silence(400)
    out = trim(padded)
    assert abs(pcm.duration(out) - pcm.duration(tone)) < 0.05


async def run_call(host, scenario, caller_lines, speech_models, **kw):
    stt, tts = speech_models
    caller = Caller(scenario, FakeCallerLlm(caller_lines))
    runner = CallRunner(scenario, ReferenceAdapter(f"ws://{host}/call"), caller, tts, stt, **kw)
    return await runner.run()


async def test_full_call(start_agent, speech_models):
    host = start_agent(
        [
            ["Sure, what name is it under?"],
            [ToolCall("book_appointment", {"name": "Priya Shah", "day": "tuesday", "time": "15:00"})],
            ["You're booked for Tuesday at 3 pm."],
            ["Goodbye!"],
        ]
    )
    scenario = parse({"id": "full", "name": "Priya Shah", "goal": "book tuesday 3pm"})
    lines = ["Hi, I'd like Tuesday at 3 pm.", "Priya Shah.", "Thanks, bye. [END]"]
    rec = await run_call(host, scenario, lines, speech_models)

    assert rec.ended_by == "caller", rec.errors
    assert [t.text for t in rec.caller_turns] == ["Hi, I'd like Tuesday at 3 pm.", "Priya Shah.", "Thanks, bye."]
    for t in rec.caller_turns:
        assert t.first_audio is not None and t.first_audio > t.end
        assert t.heard_as
    assert "tuesday" in rec.caller_turns[0].heard_as.lower()
    assert [c["name"] for c in rec.tool_calls] == ["book_appointment"]
    assert rec.tool_calls[0]["result"]["status"] == "confirmed"
    # greeting + three replies
    assert len(rec.agent_turns) == 4
    assert all(a.end is not None for a in rec.agent_turns)
    assert rec.agent_turns[0].text == ["Thanks for calling Maple Dental, how can I help you today?"]


async def test_barge_in_on_greeting(start_agent, speech_models):
    host = start_agent([["Monday at 9 works. What's your name?"]])
    scenario = parse(
        {
            "id": "barge",
            "goal": "book monday 9am",
            "interrupts": [{"after_turn": 0, "after_ms": 1000, "say": "Hi, I need Monday at 9."}],
        }
    )
    # the reply to the barge-in lands after the greeting timeout, that's fine
    rec = await run_call(host, scenario, ["[END]"], speech_models, greeting_timeout=2)
    assert rec.ended_by == "caller", rec.errors

    greeting = rec.agent_turns[0]
    assert greeting.interrupted_at is not None
    assert greeting.flagged
    assert greeting.stopped_ms < 1000
    first = rec.caller_turns[0]
    assert first.kind == "interrupt"
    assert first.first_audio is not None
    # exact words depend on where the agent's vad cuts in, just check it was heard
    assert first.heard_as


async def test_timeout_only_counts_quiet_time(start_agent, speech_models):
    host = start_agent(
        [
            ["Monday has nine, ten and eleven open. ", "Tuesday has the whole afternoon. ", "Wednesday is wide open. "],
            ["Friday at ten works."],
        ]
    )
    scenario = parse(
        {
            "id": "slow",
            "goal": "x",
            "interrupts": [{"after_turn": 1, "after_ms": 1000, "say": "Sorry, I meant Friday."}],
        }
    )
    # the whole exchange takes longer than the timeout, but the line is never quiet that long
    rec = await run_call(host, scenario, ["What's open on Monday?"], speech_models, reply_timeout=4)
    assert rec.ended_by == "caller", rec.errors
    assert [t.kind for t in rec.caller_turns] == ["turn", "interrupt"]
    assert rec.agent_turns[-1].text == ["Friday at ten works."]
