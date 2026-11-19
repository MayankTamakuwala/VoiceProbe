import json
import logging
from pathlib import Path

from voiceprobe.probe.scenario import Scenario
from voiceprobe.probe.score import BARGE_MS, score_call, summarize

log = logging.getLogger(__name__)


def load_results(run_dir: Path) -> list[dict]:
    results = (json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json")))
    return [r for r in results if "record" in r]


def score_run(run_dir: str | Path, scenarios: list[Scenario], barge_ms: float = BARGE_MS) -> dict:
    run_dir = Path(run_dir)
    by_id = {s.id: s for s in scenarios}
    calls = []
    for result in load_results(run_dir):
        sid = result["record"]["scenario"]
        if sid not in by_id:
            log.warning("no scenario %s, scoring without expectations", sid)
        calls.append(score_call(result, by_id.get(sid) or Scenario(sid, []), barge_ms))
    report = {"run": run_dir.name, "barge_ms": barge_ms, "summary": summarize(calls), "calls": calls}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _secs(ms: float | None) -> str:
    return f"{ms / 1000:.1f}s" if ms is not None else "-"


def _verdict(ok: bool | None) -> str:
    return {True: "ok", False: "FAIL", None: "-"}[ok]


def format_report(report: dict) -> str:
    lines = [
        f"{'scenario':24}{'ended':14}{'time':>7}  {'task':6}{'tools':7}{'ttfa':>6}  {'barge':8}{'wer':>6}"
    ]
    for c in report["calls"]:
        barges = c["barge_ins"]
        barge = f"{sum(b['handled'] for b in barges)}/{len(barges) + c['barge_missed']}" if barges or c["barge_missed"] else "-"
        wer = f"{c['wer_errors'] / c['wer_words']:.1%}" if c["wer_words"] else "-"
        lines.append(
            f"{c['scenario']:24}{c['ended_by']:14}{c['duration']:>6.0f}s  "
            f"{_verdict(c['task']):6}{_verdict(c['tools_ok']):7}{_secs(c['ttfa_p50_ms']):>6}  {barge:8}{wer:>6}"
        )
        for note in c["task_notes"] + c["tool_notes"] + c["errors"]:
            lines.append(f"    - {note}")

    s = report["summary"]
    t, b, w = s["ttfa_ms"], s["barge_in"], s["wer"]
    ended = ", ".join(f"{k} {v}" for k, v in s["ended_by"].items())
    lines += [
        "",
        f"calls              {s['calls']} ({ended})",
        f"time to 1st audio  p50 {_secs(t['p50'])}  p95 {_secs(t['p95'])}  over {t['turns']} turns",
    ]
    if b["total"] or b["missed"]:
        line = f"barge-in           {b['handled']}/{b['total']} stopped within {report['barge_ms']:.0f}ms"
        if b["stop_p50_ms"] is not None:
            line += f"  (p50 {b['stop_p50_ms']:.0f}ms)"
        if b["missed"]:
            line += f", {b['missed']} never fired"
        lines.append(line)
    ts, tl = s["task_success"], s["tools"]
    lines.append(f"task success       {ts['passed']}/{ts['scored']}")
    failed = tl["failed_calls"]
    lines.append(f"tool use           {tl['passed']}/{tl['total']}  ({failed} failed tool call{'' if failed == 1 else 's'})")
    if w["rate"] is not None:
        unheard = f", {w['unheard_turns']} turns never transcribed" if w["unheard_turns"] else ""
        lines.append(f"transcription wer  {w['rate']:.1%} over {w['words']} words{unheard}")
    return "\n".join(lines)
