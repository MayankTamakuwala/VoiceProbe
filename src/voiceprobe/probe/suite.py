import fcntl
import json
import logging
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

from voiceprobe.probe.adapters import Adapter
from voiceprobe.probe.caller import Caller
from voiceprobe.probe.runner import CallRunner
from voiceprobe.probe.scenario import Scenario

log = logging.getLogger(__name__)


class Bookings:
    """Talks to the booking api to set up each call and read back what happened."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001", client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(base_url=base_url, timeout=5.0)

    async def prepare(self, existing: list[dict]) -> None:
        r = await self.client.post("/admin/reset")
        r.raise_for_status()
        for i, b in enumerate(existing):
            r = await self.client.post("/bookings", json=b, headers={"Idempotency-Key": f"seed-{i}"})
            r.raise_for_status()

    async def all(self) -> list[dict]:
        r = await self.client.get("/bookings")
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        await self.client.aclose()


LOCK_PATH = Path(tempfile.gettempdir()) / "voiceprobe.lock"


class Busy(Exception):
    pass


@contextmanager
def one_at_a_time(path: Path = LOCK_PATH) -> Iterator[None]:
    # two runs at once reset each other's bookings and share the same ollama,
    # so both results end up wrong. the os drops the lock if we crash
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise Busy("another voiceprobe run or load test is going, wait for it to finish") from None
    try:
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def run_dir(root: str | Path = "out/runs", label: str = "") -> Path:
    name = time.strftime("%Y%m%d-%H%M%S") + (f"-{label}" if label else "")
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


async def run_scenario(
    scenario: Scenario,
    adapter: Adapter,
    caller_llm,
    tts,
    stt=None,
    bookings: Bookings | None = None,
    hear: str = "text",
) -> dict:
    if bookings:
        await bookings.prepare(scenario.existing_bookings)
    caller = Caller(scenario, caller_llm)
    runner = CallRunner(scenario, adapter, caller, tts, stt, hear=hear)
    record = await runner.run()
    result = {"record": record.to_dict(), "bookings": None}
    if bookings:
        result["bookings"] = await bookings.all()
    return result


async def run_suite(
    scenarios: list[Scenario],
    make_adapter: Callable[[], Adapter],
    caller_llm,
    tts,
    stt=None,
    bookings: Bookings | None = None,
    out: Path | None = None,
    hear: str = "text",
    repeat: int = 1,
) -> list[dict]:
    results = []
    for n in range(repeat):
        for s in scenarios:
            log.info("running %s (%d/%d)", s.id, n + 1, repeat)
            result = await run_scenario(s, make_adapter(), caller_llm, tts, stt, bookings, hear)
            results.append(result)
            if out:
                name = s.id if repeat == 1 else f"{s.id}-{n + 1}"
                (out / f"{name}.json").write_text(json.dumps(result, indent=2))
    return results
