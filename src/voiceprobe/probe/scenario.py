from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


class ScenarioError(ValueError):
    pass


@dataclass
class Interrupt:
    # barge in on the first agent turn >= after_turn that is still talking
    # after_ms into its audio. turn 0 is the greeting.
    say: str
    after_turn: int = 1
    after_ms: int = 1000


@dataclass
class Expect:
    booking: dict | None = None
    no_booking: bool = False
    tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    id: str
    goal: list[str]
    name: str = "Sam Taylor"
    style: str = "Friendly and to the point."
    description: str = ""
    interrupts: list[Interrupt] = field(default_factory=list)
    existing_bookings: list[dict] = field(default_factory=list)
    expect: Expect = field(default_factory=Expect)
    max_turns: int = 8
    path: Path | None = None


def _check_keys(cls, data: dict, where: str) -> None:
    allowed = {f.name for f in fields(cls)} - {"path"}
    unknown = set(data) - allowed
    if unknown:
        raise ScenarioError(f"{where}: unknown keys {sorted(unknown)}")


def parse(data: dict, path: Path | None = None) -> Scenario:
    where = str(path or data.get("id", "scenario"))
    if not isinstance(data, dict):
        raise ScenarioError(f"{where}: expected a mapping")
    _check_keys(Scenario, data, where)
    if "id" not in data or "goal" not in data:
        raise ScenarioError(f"{where}: id and goal are required")

    data = dict(data)
    goal = data.pop("goal")
    goal = [goal] if isinstance(goal, str) else list(goal)

    expect = data.pop("expect", None) or {}
    _check_keys(Expect, expect, f"{where} expect")
    if expect.get("booking") and expect.get("no_booking"):
        raise ScenarioError(f"{where}: expect can't have both booking and no_booking")

    interrupts = []
    for item in data.pop("interrupts", None) or []:
        _check_keys(Interrupt, item, f"{where} interrupt")
        interrupts.append(Interrupt(**item))

    return Scenario(
        goal=goal,
        expect=Expect(**expect),
        interrupts=interrupts,
        path=path,
        **data,
    )


def load(path: str | Path) -> Scenario:
    path = Path(path)
    with path.open() as f:
        return parse(yaml.safe_load(f), path)


def load_all(paths: list[str | Path]) -> list[Scenario]:
    out = []
    for p in map(Path, paths):
        files = sorted(p.glob("*.yaml")) if p.is_dir() else [p]
        out.extend(load(f) for f in files)
    ids = [s.id for s in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ScenarioError(f"duplicate scenario ids: {sorted(dupes)}")
    return out
