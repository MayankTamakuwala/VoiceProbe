from collections import Counter

import numpy as np

from voiceprobe.booking.store import parse_time
from voiceprobe.probe import wer
from voiceprobe.probe.scenario import Scenario

# how quickly the agent has to go quiet once the caller talks over it
BARGE_MS = 500


def _name(name: str) -> str:
    return " ".join(name.lower().split())


def _slot(b: dict) -> tuple[str, str, str]:
    return _name(b["name"]), b["day"].lower(), parse_time(b["time"])


def _describe(bookings: list[dict]) -> str:
    return ", ".join(f"{b['name']} {b['day']} {b['time']}" for b in bookings)


def percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def ttfa(record: dict) -> list[float]:
    """Ms from the end of each caller turn to the first agent audio after it."""
    return [
        round((t["first_audio"] - t["end"]) * 1000, 1)
        for t in record["caller_turns"]
        if t["first_audio"] is not None and t["end"] is not None
    ]


def barge_ins(record: dict, limit_ms: float = BARGE_MS) -> list[dict]:
    out = []
    for a in record["agent_turns"]:
        if a["interrupted_at"] is None:
            continue
        stopped = a["stopped_ms"]
        if stopped is None and a["end"] is not None:
            stopped = round((a["end"] - a["interrupted_at"]) * 1000)
        out.append(
            {
                "turn": a["index"],
                "stopped_ms": stopped,
                "flagged": a["flagged"],
                "handled": stopped is not None and stopped <= limit_ms,
            }
        )
    return out


def check_bookings(scenario: Scenario, bookings: list[dict] | None) -> tuple[bool | None, list[str]]:
    if bookings is None:
        return None, ["no booking data for this agent"]
    notes = []
    confirmed = [b for b in bookings if b["status"] == "confirmed"]

    # whatever was seeded before the call should come through untouched
    mine = list(confirmed)
    for seed in scenario.existing_bookings:
        match = next((b for b in mine if _slot(b) == _slot(seed)), None)
        if match:
            mine.remove(match)
        else:
            notes.append(f"existing booking was changed: {_describe([seed])}")

    exp = scenario.expect
    if exp.no_booking:
        if mine:
            notes.append(f"expected no booking, found {_describe(mine)}")
    elif exp.booking:
        want = exp.booking
        if not mine:
            notes.append("nothing booked")
        else:
            if len(mine) > 1:
                notes.append(f"double booked: {_describe(mine)}")
            hits = [
                b
                for b in mine
                if ("day" not in want or b["day"] == want["day"].lower())
                and ("time" not in want or b["time"] == parse_time(want["time"]))
            ]
            if not hits:
                notes.append(f"wrong slot: {_describe(mine)}")
            if "name" in want and not any(_name(b["name"]) == _name(want["name"]) for b in hits or mine):
                notes.append(f"wrong name: {', '.join(b['name'] for b in hits or mine)}")
    elif not notes:
        return None, []
    return not notes, notes


def check_tools(scenario: Scenario, tool_calls: list[dict]) -> tuple[bool, list[str]]:
    # the agent's own guards stopped blocked ones, so they never touched the bookings
    used = [c["name"] for c in tool_calls if not c.get("blocked")]
    notes = [f"never called {t}" for t in scenario.expect.tools if t not in used]
    notes += [f"called {t}" for t in scenario.expect.forbidden_tools if t in used]
    return not notes, notes


def heard_pairs(record: dict) -> list[tuple[str, str]]:
    return [(t["text"], t["heard_as"]) for t in record["caller_turns"] if t["heard_as"] is not None]


def score_call(result: dict, scenario: Scenario, barge_ms: float = BARGE_MS) -> dict:
    rec = result["record"]
    task, task_notes = check_bookings(scenario, result.get("bookings"))
    tools_ok, tool_notes = check_tools(scenario, rec["tool_calls"])
    barges = barge_ins(rec, barge_ms)
    fired = sum(t["kind"] == "interrupt" for t in rec["caller_turns"])
    errors, words = wer.count(heard_pairs(rec))
    latencies = ttfa(rec)
    return {
        "scenario": rec["scenario"],
        "call_id": rec["call_id"],
        "agent": rec["agent"],
        "ended_by": rec["ended_by"],
        "duration": rec["duration"],
        "turns": len(rec["caller_turns"]),
        "ttfa_ms": latencies,
        "ttfa_p50_ms": percentile(latencies, 50),
        "barge_ins": barges,
        "barge_missed": max(0, len(scenario.interrupts) - fired),
        "task": task,
        "task_notes": task_notes,
        "tools_ok": tools_ok,
        "tool_notes": tool_notes,
        "tool_errors": sum(
            isinstance(c["result"], dict) and "error" in c["result"] and not c.get("blocked")
            for c in rec["tool_calls"]
        ),
        "blocked_tools": [c["name"] for c in rec["tool_calls"] if c.get("blocked")],
        "wer_errors": errors,
        "wer_words": words,
        "unheard": sum(t["heard_as"] is None and t["end"] is not None for t in rec["caller_turns"]),
        "errors": rec["errors"],
    }


def summarize(calls: list[dict]) -> dict:
    latencies = [x for c in calls for x in c["ttfa_ms"]]
    barges = [b for c in calls for b in c["barge_ins"]]
    stops = [b["stopped_ms"] for b in barges if b["stopped_ms"] is not None]
    tasks = [c["task"] for c in calls if c["task"] is not None]
    errors = sum(c["wer_errors"] for c in calls)
    words = sum(c["wer_words"] for c in calls)
    return {
        "calls": len(calls),
        "ended_by": dict(Counter(c["ended_by"] for c in calls)),
        "ttfa_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "turns": len(latencies),
        },
        "barge_in": {
            "handled": sum(b["handled"] for b in barges),
            "total": len(barges),
            "missed": sum(c["barge_missed"] for c in calls),
            "stop_p50_ms": percentile(stops, 50),
        },
        "task_success": {"passed": sum(tasks), "scored": len(tasks)},
        "tools": {
            "passed": sum(c["tools_ok"] for c in calls),
            "total": len(calls),
            "failed_calls": sum(c["tool_errors"] for c in calls),
            "blocked": sum(len(c["blocked_tools"]) for c in calls),
        },
        "wer": {
            "rate": round(errors / words, 4) if words else None,
            "words": words,
            "unheard_turns": sum(c["unheard"] for c in calls),
        },
    }
