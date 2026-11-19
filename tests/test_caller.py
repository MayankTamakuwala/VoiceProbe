import pytest

from voiceprobe.probe.caller import Caller, ScriptedCaller, clean
from voiceprobe.probe.scenario import parse


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Hi, I'd like Tuesday at 3.", ("Hi, I'd like Tuesday at 3.", False)),
        ("Thanks, bye! [END]", ("Thanks, bye!", True)),
        ('"Sure, it\'s Priya."', ("Sure, it's Priya.", False)),
        ("Caller: Monday please", ("Monday please", False)),
        ("Priya Shah: Yes", ("Yes", False)),
        ("*sighs* Okay, Friday then.", ("Okay, Friday then.", False)),
        ("Yes please.\nReceptionist: Great!", ("Yes please.", False)),
    ],
)
def test_clean(raw, expected):
    assert clean(raw) == expected


class FakeLlm:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def complete(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def scenario(**kw):
    return parse({"id": "t", "name": "Priya Shah", "goal": ["Book Tuesday 3pm"], **kw})


async def test_prompt_has_persona_and_goals():
    caller = Caller(scenario(), FakeLlm([]))
    system = caller.messages[0]["content"]
    assert "Priya Shah" in system
    assert "1. Book Tuesday 3pm" in system


async def test_end_marker_finishes_call():
    caller = Caller(scenario(), FakeLlm(["Tuesday at 3 please.", "Thanks, bye. [END]"]))
    assert await caller.next_line("How can I help?") == "Tuesday at 3 please."
    assert not caller.done
    assert await caller.next_line("You're booked.") == "Thanks, bye."
    assert caller.done


async def test_max_turns():
    caller = Caller(scenario(max_turns=1), FakeLlm(["Hello?"]))
    await caller.next_line("Hi")
    assert caller.done


async def test_retries_empty_reply():
    llm = FakeLlm(["*nods*", "Yes, that works."])
    caller = Caller(scenario(), llm)
    assert await caller.next_line("Tuesday at 3?") == "Yes, that works."
    assert len(llm.calls) == 2


async def test_interrupt_goes_into_history():
    caller = Caller(scenario(), FakeLlm([]))
    caller.interrupted("We have openings on Monday at", "Sorry, I meant Friday.")
    assert caller.messages[-2] == {"role": "user", "content": "We have openings on Monday at..."}
    assert caller.messages[-1] == {"role": "assistant", "content": "Sorry, I meant Friday."}


async def test_bare_end_means_hang_up():
    llm = FakeLlm(["[END]"])
    caller = Caller(scenario(), llm)
    assert await caller.next_line("Goodbye!") == ""
    assert caller.done
    assert len(llm.calls) == 1


async def test_goodbye_without_marker_ends():
    caller = Caller(scenario(), FakeLlm(["Great, thanks, bye!"]))
    await caller.next_line("You're booked.")
    assert caller.done


async def test_scripted_caller():
    caller = ScriptedCaller(["hi", "bye"])
    assert await caller.next_line("hello") == "hi"
    assert not caller.done
    assert await caller.next_line("yes?") == "bye"
    assert caller.done
    assert await caller.next_line("") == ""
