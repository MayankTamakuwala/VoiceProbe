import asyncio

import httpx
import pytest

from voiceprobe.agent.dialog import Dialog
from voiceprobe.agent.llm import ToolCall
from voiceprobe.agent.tools import BookingTools
from voiceprobe.booking.api import create_app


class ScriptedLlm:
    """Plays back canned turns instead of hitting ollama."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []
        self.offered = []

    async def chat(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        self.offered.append([t["function"]["name"] for t in tools or []])
        for part in self.turns.pop(0):
            await asyncio.sleep(0)
            yield part


@pytest.fixture
def app(tmp_path):
    return create_app(str(tmp_path / "bookings.db"))


@pytest.fixture
def tools(app):
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://b")
    return BookingTools("call-1", client=client)


async def collect(gen):
    return [s async for s in gen]


async def test_plain_reply_streams_sentences(tools):
    llm = ScriptedLlm([["Sure", ", what day", " works? I can", " check."]])
    dialog = Dialog(llm, tools)
    assert await collect(dialog.reply("I need a cleaning")) == [
        "Sure, what day works?",
        "I can check.",
    ]
    assert dialog.messages[-1] == {"role": "assistant", "content": "Sure, what day works? I can check."}


async def test_tool_round_trip(tools, app):
    llm = ScriptedLlm(
        [
            [ToolCall("book_appointment", {"name": "Sam", "day": "tuesday", "time": "15:00"})],
            ["You're booked for Tuesday at 3 pm."],
        ]
    )
    dialog = Dialog(llm, tools)
    out = await collect(dialog.reply("Yes, Tuesday 3pm, I'm Sam"))
    assert out == ["One moment, booking that now.", "You're booked for Tuesday at 3 pm."]
    assert len(app.state.store.bookings()) == 1
    # second llm round should see the tool result
    assert llm.seen[1][-1]["role"] == "tool"
    assert '"status": "confirmed"' in llm.seen[1][-1]["content"]


async def test_interrupt_keeps_only_heard_text(tools):
    llm = ScriptedLlm([["We have openings at nine. ", "Also ten and eleven. ", "And noon."]])
    dialog = Dialog(llm, tools)
    gen = dialog.reply("What's free Monday?")
    first = await gen.__anext__()
    await gen.aclose()
    dialog.interrupted(first)
    assert dialog.messages[-1]["content"] == "We have openings at nine..."
    assert "tool_calls" not in dialog.messages[-1]


async def test_goodbye_does_not_cancel(tools, app):
    llm = ScriptedLlm(
        [
            [ToolCall("book_appointment", {"name": "Sam", "day": "monday", "time": "10:00"})],
            ["Booked."],
            [ToolCall("cancel_appointment", {})],
            ["Bye!"],
        ]
    )
    dialog = Dialog(llm, tools)
    await collect(dialog.reply("Book Monday 10, I'm Sam"))
    await collect(dialog.reply("Thanks, bye"))
    assert [b.status for b in app.state.store.bookings()] == ["confirmed"]
    assert "did not ask to cancel" in llm.seen[3][-1]["content"]
    assert tools.history[-1].name == "cancel_appointment" and tools.history[-1].blocked


async def test_cancel_only_offered_when_asked(tools, app):
    llm = ScriptedLlm(
        [
            [ToolCall("book_appointment", {"name": "Sam", "day": "monday", "time": "10:00"})],
            ["Booked."],
            ["Are you sure you want to cancel?"],
            [ToolCall("cancel_appointment", {})],
            ["Done, it's cancelled."],
        ]
    )
    dialog = Dialog(llm, tools)
    await collect(dialog.reply("Book Monday 10, I'm Sam"))
    assert "cancel_appointment" not in llm.offered[0]
    await collect(dialog.reply("Actually can I cancel that?"))
    assert "cancel_appointment" in llm.offered[2]
    # a plain yes to "are you sure" still has to be able to cancel
    await collect(dialog.reply("Yes."))
    assert [b.status for b in app.state.store.bookings()] == ["cancelled"]
    assert not tools.history[-1].blocked


async def test_no_booking_without_a_name(tools, app):
    llm = ScriptedLlm(
        [
            [ToolCall("book_appointment", {"name": "caller", "day": "monday", "time": "09:00"})],
            ["Sure, can I get your name?"],
            [ToolCall("book_appointment", {"name": "Tom Reyes", "day": "monday", "time": "09:00"})],
            ["Booked, Tom."],
        ]
    )
    dialog = Dialog(llm, tools)
    # no "booking that now" for a booking that isn't going to happen
    assert await collect(dialog.reply("Monday at 9am please")) == ["Sure, can I get your name?"]
    assert app.state.store.bookings() == []
    assert "name" in llm.seen[1][-1]["content"]
    await collect(dialog.reply("It's Tom Reyes."))
    assert [b.name for b in app.state.store.bookings()] == ["Tom Reyes"]


def test_name_check():
    from voiceprobe.agent.dialog import _name_given

    said = "Hi, I need the first slot on Monday. My name is Leena Park."
    assert _name_given("Leena Park", said)
    assert _name_given("Lena Park", said)
    assert not _name_given("the caller", said)
    assert not _name_given("Sam", said)
