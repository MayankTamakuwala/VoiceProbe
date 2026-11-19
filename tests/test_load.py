import json

from voiceprobe.probe.adapters import ReferenceAdapter
from voiceprobe.probe.load import breaking_point, format_load, peak_overlap, ramp


def level(calls, p95, failed=()):
    return {"calls": calls, "peak": calls, "turns": 3, "p50_ms": p95, "p95_ms": p95, "max_ms": p95, "failed": list(failed)}


def test_peak_overlap():
    records = [
        {"started_at": 0.0, "duration": 10.0},
        {"started_at": 2.0, "duration": 10.0},
        {"started_at": 11.0, "duration": 5.0},
    ]
    assert peak_overlap(records) == 2


def test_breaking_point():
    levels = [level(1, 4000), level(2, 6000), level(4, 9000), level(8, 20000)]
    assert breaking_point(levels) == (8000, 4)
    assert breaking_point(levels, max_p95_ms=10000) == (10000, 8)
    assert breaking_point(levels[:2]) == (8000, None)
    # a dropped call breaks it even if latency looks fine
    assert breaking_point([level(1, 4000), level(2, 4100, ["c1"])]) == (8000, 2)


def test_format_load():
    report = {"levels": [level(1, 4000), level(2, 9000)], "p95_limit_ms": 8000, "breaks_at": 2, "holds_up_to": 1}
    assert "breaks at 2 concurrent calls (p95 limit 8.0s), holds up to 1" in format_load(report)


async def test_ramp(start_agent, speech_models, tmp_path):
    host = start_agent([])
    _, tts = speech_models
    report = await ramp(
        [1, 2],
        lambda: ReferenceAdapter(f"ws://{host}/call"),
        tts,
        out=tmp_path,
        max_p95_ms=60000,
        stagger=0.2,
        script=["Hi there, is anyone free today?", "Okay, bye."],
    )
    assert [lv["calls"] for lv in report["levels"]] == [1, 2]
    two = report["levels"][1]
    assert two["peak"] == 2
    assert two["failed"] == []
    assert two["turns"] == 4
    assert report["breaks_at"] is None and report["holds_up_to"] == 2
    assert len(list((tmp_path / "level-2").glob("*.json"))) == 2
    assert json.loads((tmp_path / "load.json").read_text())["levels"][0]["calls"] == 1
