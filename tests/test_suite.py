import json

import httpx

from voiceprobe.agent.llm import ToolCall
from voiceprobe.booking.api import create_app as create_booking_app
from voiceprobe.probe.adapters import ReferenceAdapter
from voiceprobe.probe.scenario import parse
from voiceprobe.probe.suite import Bookings, run_suite


class Lines:
    def __init__(self, lines):
        self.lines = list(lines)

    async def complete(self, messages):
        return self.lines.pop(0) if self.lines else "[END]"


async def test_suite_seeds_and_saves(start_agent, speech_models, tmp_path):
    app = create_booking_app(str(tmp_path / "shared.db"))
    # leftovers from an earlier run should be wiped
    app.state.store.book("old", "Someone Else", "friday", "09:00")
    host = start_agent(
        [
            [ToolCall("book_appointment", {"name": "Dana Lopez", "day": "thursday", "time": "11:00"})],
            ["Eleven is taken, how about noon?"],
            [ToolCall("book_appointment", {"name": "Dana Lopez", "day": "thursday", "time": "12:00"})],
            ["Okay, you're in Thursday at noon."],
        ],
        booking_app=app,
    )
    scenario = parse(
        {
            "id": "taken",
            "name": "Dana Lopez",
            "goal": "thursday 11am",
            "existing_bookings": [{"name": "Alex Kim", "day": "thursday", "time": "11:00"}],
        }
    )
    bookings = Bookings(
        client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://booking")
    )
    stt, tts = speech_models
    results = await run_suite(
        [scenario],
        lambda: ReferenceAdapter(f"ws://{host}/call"),
        # name first, so it's heard even if the agent's vad splits the sentence
        Lines(["Dana Lopez here, Thursday at 11 please.", "Noon is fine.", "Great, bye. [END]"]),
        tts,
        stt,
        bookings=bookings,
        out=tmp_path,
    )
    await bookings.close()

    rec = results[0]["record"]
    assert rec["ended_by"] == "caller", rec["errors"]
    assert [c["name"] for c in rec["tool_calls"]] == ["book_appointment"] * 2
    assert "error" in rec["tool_calls"][0]["result"]

    final = {(b["name"], b["day"], b["time"]) for b in results[0]["bookings"]}
    assert final == {("Alex Kim", "thursday", "11:00"), ("Dana Lopez", "thursday", "12:00")}
    saved = json.loads((tmp_path / "taken.json").read_text())
    assert saved["record"]["scenario"] == "taken"
