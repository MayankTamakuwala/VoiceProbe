import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)


def _fn(name: str, description: str, /, **params: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


DAY = "weekday, monday to friday"
TIME = "24h time like 15:00"

TOOLS = [
    _fn("check_availability", "List the free appointment times for a day.", day=DAY),
    _fn(
        "book_appointment",
        "Book an appointment. Only call this once the caller confirmed the day and time.",
        name="caller's name",
        day=DAY,
        time=TIME,
    ),
    _fn("reschedule_appointment", "Move this caller's booking to another slot.", day=DAY, time=TIME),
    _fn("cancel_appointment", "Cancel this caller's booking."),
]


@dataclass
class ToolRecord:
    name: str
    arguments: dict
    result: dict
    started: float
    ms: float
    attempts: int


@dataclass
class BookingTools:
    call_id: str
    base_url: str = field(default_factory=lambda: os.getenv("BOOKING_URL", "http://127.0.0.1:8001"))
    client: httpx.AsyncClient | None = None
    retries: int = 3
    booking_id: str | None = None
    history: list[ToolRecord] = field(default_factory=list)

    def __post_init__(self):
        if self.client is None:
            self.client = httpx.AsyncClient(base_url=self.base_url, timeout=5.0)

    def _key(self, name: str, args: dict) -> str:
        # same call + same tool + same args -> same key, so if the model fires
        # a duplicate tool call it gets deduped by the booking api too
        digest = hashlib.sha1(json.dumps(args, sort_keys=True).encode()).hexdigest()[:12]
        return f"{self.call_id}:{name}:{digest}"

    async def _request(self, method: str, path: str, key: str | None, body: dict | None):
        headers = {"Idempotency-Key": key} if key else {}
        last_error = "booking system unavailable"
        for attempt in range(1, self.retries + 1):
            try:
                r = await self.client.request(method, path, json=body, headers=headers)
                if r.status_code < 500:
                    return r, attempt
                last_error = f"booking system error {r.status_code}"
            except httpx.TransportError as e:
                last_error = f"booking system unavailable ({type(e).__name__})"
            log.warning("%s %s failed (attempt %d): %s", method, path, attempt, last_error)
            await asyncio.sleep(0.1 * 2**attempt)
        return last_error, self.retries

    async def run(self, name: str, args: dict) -> dict:
        started = time.time()
        t0 = time.perf_counter()
        result, attempts = await self._dispatch(name, args)
        self.history.append(
            ToolRecord(name, args, result, started, (time.perf_counter() - t0) * 1000, attempts)
        )
        return result

    async def _dispatch(self, name: str, args: dict) -> tuple[dict, int]:
        if name == "check_availability":
            req = ("GET", f"/slots/{args.get('day', '')}", None, None)
        elif name == "book_appointment":
            body = {k: str(args.get(k, "")) for k in ("name", "day", "time")}
            req = ("POST", "/bookings", self._key(name, body), body)
        elif name in ("reschedule_appointment", "cancel_appointment"):
            if not self.booking_id:
                return {"error": "this caller has no booking yet"}, 0
            if name == "cancel_appointment":
                req = ("POST", f"/bookings/{self.booking_id}/cancel", self._key(name, {}), None)
            else:
                body = {k: str(args.get(k, "")) for k in ("day", "time")}
                path = f"/bookings/{self.booking_id}/reschedule"
                req = ("POST", path, self._key(name, body), body)
        else:
            return {"error": f"unknown tool {name}"}, 0

        r, attempts = await self._request(*req)
        if isinstance(r, str):
            return {"error": r}, attempts
        data = r.json()
        if r.status_code >= 400:
            return {"error": data.get("error") or data.get("detail") or r.text}, attempts
        if name == "book_appointment":
            self.booking_id = data["id"]
        return data, attempts

    async def close(self) -> None:
        await self.client.aclose()
