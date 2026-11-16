from pathlib import Path

import pytest

from voiceprobe.probe.scenario import ScenarioError, load_all, parse

SCENARIOS = Path(__file__).parent.parent / "scenarios"


def test_starter_scenarios_load():
    scenarios = load_all([SCENARIOS])
    assert len(scenarios) >= 6
    ids = {s.id for s in scenarios}
    assert {"book-simple", "change-mind", "interrupt-greeting"} <= ids


def test_parse_defaults():
    s = parse({"id": "x", "goal": "Book Monday at 9"})
    assert s.goal == ["Book Monday at 9"]
    assert s.interrupts == []
    assert s.expect.booking is None
    assert s.max_turns == 8


def test_parse_interrupt():
    s = parse(
        {
            "id": "x",
            "goal": ["a"],
            "interrupts": [{"say": "wait", "after_ms": 800}],
        }
    )
    assert s.interrupts[0].after_turn == 1
    assert s.interrupts[0].after_ms == 800


@pytest.mark.parametrize(
    "data",
    [
        {"goal": ["a"]},
        {"id": "x", "goal": ["a"], "typo": 1},
        {"id": "x", "goal": ["a"], "expect": {"bookings": {}}},
        {"id": "x", "goal": ["a"], "expect": {"booking": {"day": "monday"}, "no_booking": True}},
        {"id": "x", "goal": ["a"], "interrupts": [{"after_ms": 100}]},
    ],
)
def test_parse_rejects_bad_input(data):
    with pytest.raises((ScenarioError, TypeError)):
        parse(data)


def test_duplicate_ids(tmp_path):
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text("id: same\ngoal: hi\n")
    with pytest.raises(ScenarioError):
        load_all([tmp_path])
