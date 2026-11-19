import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path

from voiceprobe.probe.adapters import Adapter
from voiceprobe.probe.caller import ScriptedCaller
from voiceprobe.probe.runner import CallRecord, CallRunner
from voiceprobe.probe.scenario import Scenario
from voiceprobe.probe.score import percentile, ttfa

log = logging.getLogger(__name__)

# one turn that needs a tool, one that doesn't, then hang up. nothing gets
# booked so concurrent callers don't fight over slots
SCRIPT = [
    "Hi, what times do you have free on Wednesday?",
    "Okay. Do you do teeth whitening there too?",
    "Alright, thanks, bye.",
]


class CachedTts:
    """Every caller says the same lines, no point synthesizing them per call."""

    def __init__(self, tts):
        self.tts = tts
        self.cache: dict[str, bytes] = {}

    def synthesize(self, text: str) -> bytes:
        if text not in self.cache:
            self.cache[text] = self.tts.synthesize(text)
        return self.cache[text]


def peak_overlap(records: list[dict]) -> int:
    edges = []
    for r in records:
        edges += [(r["started_at"], 1), (r["started_at"] + r["duration"], -1)]
    peak = cur = 0
    for _, step in sorted(edges):
        cur += step
        peak = max(peak, cur)
    return peak


def level_summary(calls: int, records: list[dict]) -> dict:
    latencies = [x for r in records for x in ttfa(r)]
    failed = [r["call_id"] for r in records if r["ended_by"] != "caller" or r["errors"]]
    return {
        "calls": calls,
        "peak": peak_overlap(records),
        "turns": len(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "max_ms": max(latencies, default=None),
        "failed": failed,
    }


def breaking_point(levels: list[dict], max_p95_ms: float | None = None) -> tuple[float | None, int | None]:
    """Returns the p95 limit used and the first level over it (or with failures)."""
    if not levels:
        return max_p95_ms, None
    limit = max_p95_ms
    if limit is None and levels[0]["p95_ms"] is not None:
        # no absolute target, call it broken once p95 doubles from a single call
        limit = 2 * levels[0]["p95_ms"]
    for lv in levels:
        if lv["failed"] or lv["p95_ms"] is None or (limit is not None and lv["p95_ms"] > limit):
            return limit, lv["calls"]
    return limit, None


async def run_level(
    calls: int,
    make_adapter: Callable[[], Adapter],
    tts,
    script: list[str] = SCRIPT,
    stagger: float = 0.5,
) -> list[CallRecord]:
    async def one(i: int) -> CallRecord:
        # a little spread so the calls aren't in perfect lockstep
        await asyncio.sleep(i * stagger)
        scenario = Scenario(f"load-{calls}", goal=[])
        runner = CallRunner(scenario, make_adapter(), ScriptedCaller(script), tts)
        return await runner.run(f"load-{calls}x-{i}")

    return await asyncio.gather(*(one(i) for i in range(calls)))


async def ramp(
    levels: list[int],
    make_adapter: Callable[[], Adapter],
    tts,
    out: Path | None = None,
    max_p95_ms: float | None = None,
    stagger: float = 0.5,
    keep_going: bool = False,
    script: list[str] = SCRIPT,
) -> dict:
    tts = CachedTts(tts)
    for line in script:
        await asyncio.to_thread(tts.synthesize, line)

    results = []
    for n in levels:
        log.info("load: %d concurrent calls", n)
        records = [r.to_dict() for r in await run_level(n, make_adapter, tts, script, stagger)]
        lv = level_summary(n, records)
        results.append(lv)
        log.info("load: %d calls, p95 %s ms, %d failed", n, lv["p95_ms"], len(lv["failed"]))
        if out:
            d = out / f"level-{n}"
            d.mkdir(parents=True, exist_ok=True)
            for r in records:
                (d / f"{r['call_id']}.json").write_text(json.dumps({"record": r, "bookings": None}, indent=2))
        _, broke = breaking_point(results, max_p95_ms)
        if broke is not None and not keep_going:
            break
        # let the agent settle before the next level
        await asyncio.sleep(2)

    limit, broke = breaking_point(results, max_p95_ms)
    held = [lv["calls"] for lv in results if broke is None or lv["calls"] < broke]
    report = {
        "levels": results,
        "p95_limit_ms": limit,
        "breaks_at": broke,
        "holds_up_to": max(held, default=None),
    }
    if out:
        (out / "load.json").write_text(json.dumps(report, indent=2))
    return report


def format_load(report: dict) -> str:
    def secs(ms):
        return f"{ms / 1000:.1f}s" if ms is not None else "-"

    lines = [f"{'calls':>5}  {'peak':>4}  {'p50':>6}  {'p95':>6}  {'max':>6}  {'turns':>5}  failed"]
    for lv in report["levels"]:
        lines.append(
            f"{lv['calls']:>5}  {lv['peak']:>4}  {secs(lv['p50_ms']):>6}  {secs(lv['p95_ms']):>6}  "
            f"{secs(lv['max_ms']):>6}  {lv['turns']:>5}  {len(lv['failed'])}"
        )
    lines.append("")
    limit = secs(report["p95_limit_ms"])
    if report["breaks_at"] is None:
        lines.append(f"p95 stayed under {limit} at every level tried")
    else:
        held = report["holds_up_to"]
        lines.append(
            f"breaks at {report['breaks_at']} concurrent calls (p95 limit {limit}), "
            + (f"holds up to {held}" if held else "even a single call is over")
        )
    return "\n".join(lines)
