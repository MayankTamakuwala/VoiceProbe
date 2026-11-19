from voiceprobe.probe.scenario import parse
from voiceprobe.probe.score import barge_ins, check_bookings, check_tools, score_call, summarize, ttfa

CHANGE = parse(
    {
        "id": "change",
        "goal": "x",
        "expect": {
            "booking": {"name": "Marcus Bell", "day": "wednesday", "time": "10:00"},
            "tools": ["book_appointment", "reschedule_appointment"],
            "forbidden_tools": ["cancel_appointment"],
        },
    }
)


def booking(name, day, time, status="confirmed"):
    return {"id": f"{name}{day}{time}", "name": name, "day": day, "time": time, "status": status}


def caller_turn(text, end, first_audio, heard=None, kind="turn"):
    return {"text": text, "kind": kind, "start": end - 1, "end": end, "heard_as": heard, "first_audio": first_audio}


def agent_turn(index, start, end, interrupted_at=None, stopped_ms=None, flagged=False):
    return {
        "index": index, "start": start, "end": end, "text": [],
        "interrupted_at": interrupted_at, "stopped_ms": stopped_ms, "flagged": flagged,
    }


def record(**kw):
    rec = {
        "scenario": "change", "call_id": "c1", "agent": "reference", "ended_by": "caller",
        "duration": 30.0, "caller_turns": [], "agent_turns": [], "tool_calls": [], "errors": [],
    }
    rec.update(kw)
    return rec


def test_booking_ok():
    assert check_bookings(CHANGE, [booking("Marcus Bell", "wednesday", "10:00")]) == (True, [])
    # a cancelled leftover doesn't count against it
    ok, _ = check_bookings(
        CHANGE,
        [booking("Marcus Bell", "tuesday", "15:00", "cancelled"), booking("marcus  bell", "wednesday", "10:00")],
    )
    assert ok


def test_double_booking():
    ok, notes = check_bookings(
        CHANGE, [booking("Marcus Bell", "tuesday", "15:00"), booking("Marcus Bell", "wednesday", "10:00")]
    )
    assert not ok
    assert notes == ["double booked: Marcus Bell tuesday 15:00, Marcus Bell wednesday 10:00"]


def test_wrong_name_and_slot():
    ok, notes = check_bookings(CHANGE, [booking("caller", "wednesday", "10:00")])
    assert notes == ["wrong name: caller"]
    ok, notes = check_bookings(CHANGE, [booking("Marcus Bell", "wednesday", "11:00")])
    assert notes == ["wrong slot: Marcus Bell wednesday 11:00"]
    assert check_bookings(CHANGE, []) == (False, ["nothing booked"])
    assert check_bookings(CHANGE, None)[0] is None


def test_seeded_bookings():
    s = parse(
        {
            "id": "taken",
            "goal": "x",
            "existing_bookings": [{"name": "Alex Kim", "day": "thursday", "time": "11:00"}],
            "expect": {"booking": {"name": "Dana Lopez", "day": "thursday"}},
        }
    )
    seed = booking("Alex Kim", "thursday", "11:00")
    assert check_bookings(s, [seed, booking("Dana Lopez", "thursday", "12:00")]) == (True, [])
    moved = booking("Alex Kim", "friday", "11:00")
    ok, notes = check_bookings(s, [moved, booking("Dana Lopez", "thursday", "11:00")])
    assert not ok and notes[0].startswith("existing booking was changed")


def test_no_booking():
    s = parse({"id": "cancel", "goal": "x", "expect": {"no_booking": True}})
    assert check_bookings(s, [booking("Owen Grant", "monday", "09:00", "cancelled")]) == (True, [])
    assert not check_bookings(s, [booking("Owen Grant", "monday", "09:00")])[0]


def test_tools():
    calls = [{"name": "book_appointment"}, {"name": "book_appointment"}]
    assert check_tools(CHANGE, calls) == (False, ["never called reschedule_appointment"])
    calls = [{"name": "book_appointment"}, {"name": "reschedule_appointment"}, {"name": "cancel_appointment"}]
    assert check_tools(CHANGE, calls) == (False, ["called cancel_appointment"])


def test_blocked_calls_dont_count():
    calls = [
        {"name": "book_appointment"},
        {"name": "reschedule_appointment"},
        {"name": "cancel_appointment", "blocked": True},
    ]
    assert check_tools(CHANGE, calls) == (True, [])


def test_ttfa_and_barge_ins():
    rec = record(
        caller_turns=[caller_turn("hi", 2.0, 3.5), caller_turn("bye", 10.0, None)],
        agent_turns=[
            agent_turn(0, 0.1, 1.3, interrupted_at=1.0, stopped_ms=300, flagged=True),
            agent_turn(1, 3.5, 8.0, interrupted_at=5.0, stopped_ms=3000),
            agent_turn(2, 11.0, 12.0),
        ],
    )
    assert ttfa(rec) == [1500.0]
    b = barge_ins(rec)
    assert [x["handled"] for x in b] == [True, False]


def test_score_and_summary():
    rec = record(
        caller_turns=[
            caller_turn("Tuesday at 3pm please", 2.0, 4.0, "tuesday at 3 p.m. please"),
            caller_turn("Actually make it Wednesday at ten", 8.0, 10.5, "actually make it wednesday at tan"),
            caller_turn("Bye", 14.0, None),
        ],
        tool_calls=[
            {"name": "book_appointment", "result": {"id": "a"}},
            {"name": "book_appointment", "result": {"error": "taken"}},
            {"name": "cancel_appointment", "result": {"error": "caller did not ask"}, "blocked": True},
        ],
    )
    result = {
        "record": rec,
        "bookings": [booking("Marcus Bell", "tuesday", "15:00"), booking("Marcus Bell", "wednesday", "10:00")],
    }
    call = score_call(result, CHANGE)
    assert call["task"] is False and call["tools_ok"] is False
    assert call["ttfa_ms"] == [2000.0, 2500.0]
    assert call["tool_errors"] == 1
    assert call["blocked_tools"] == ["cancel_appointment"]
    assert (call["wer_errors"], call["wer_words"]) == (1, 11)
    assert call["unheard"] == 1

    s = summarize([call, {**call, "task": True, "ttfa_ms": [1000.0]}])
    assert s["task_success"] == {"passed": 1, "scored": 2}
    assert s["ttfa_ms"]["turns"] == 3 and s["ttfa_ms"]["p50"] == 2000.0
    assert s["wer"]["rate"] == round(2 / 22, 4)
