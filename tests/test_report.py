import json

from voiceprobe.probe.report import format_report, score_run
from voiceprobe.probe.scenario import parse


def test_score_run(tmp_path):
    scenario = parse(
        {
            "id": "book",
            "goal": "x",
            "interrupts": [{"say": "wait", "after_turn": 0}],
            "expect": {"booking": {"name": "Priya Shah", "day": "tuesday"}, "tools": ["book_appointment"]},
        }
    )
    record = {
        "scenario": "book", "call_id": "book-1", "agent": "reference", "ended_by": "caller",
        "duration": 20.0, "errors": [],
        "caller_turns": [
            {"text": "Tuesday please", "kind": "interrupt", "start": 1.0, "end": 2.0,
             "heard_as": "Tuesday please", "first_audio": 3.2},
        ],
        "agent_turns": [
            {"index": 0, "start": 0.1, "end": 1.2, "text": ["Hi"], "interrupted_at": 1.0,
             "stopped_ms": 200, "flagged": True},
        ],
        "tool_calls": [{"name": "book_appointment", "result": {"id": "a"}}],
    }
    bookings = [{"id": "a", "name": "Priya Shah", "day": "tuesday", "time": "10:00", "status": "confirmed"}]
    (tmp_path / "book.json").write_text(json.dumps({"record": record, "bookings": bookings}))

    report = score_run(tmp_path, [scenario])
    assert (tmp_path / "report.json").exists()
    s = report["summary"]
    assert s["task_success"] == {"passed": 1, "scored": 1}
    assert s["barge_in"]["handled"] == 1
    assert s["ttfa_ms"]["p50"] == 1200.0

    # scoring again shouldn't pick up its own report
    assert score_run(tmp_path, [scenario])["summary"]["calls"] == 1
    text = format_report(report)
    assert "book" in text and "1/1 stopped within 500ms" in text
