import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
TIMES = [f"{h:02d}:00" for h in range(9, 17)]

SCHEMA = """
create table if not exists bookings (
    id text primary key,
    name text not null,
    day text not null,
    time text not null,
    status text not null default 'confirmed',
    created_at text not null
);

create unique index if not exists one_booking_per_slot
    on bookings(day, time) where status = 'confirmed';

create table if not exists idempotency (
    key text primary key,
    request_hash text not null,
    response text not null,
    created_at text not null
);
"""


class BookingError(Exception):
    pass


class InvalidSlot(BookingError):
    pass


class SlotTaken(BookingError):
    pass


class NotFound(BookingError):
    pass


class KeyReused(BookingError):
    pass


@dataclass
class Booking:
    id: str
    name: str
    day: str
    time: str
    status: str = "confirmed"


def parse_time(value: str) -> str:
    v = value.strip().lower().replace(" ", "").replace(".", "")
    suffix = None
    if v.endswith(("am", "pm")):
        suffix, v = v[-2:], v[:-2]
    hour, _, minute = v.partition(":")
    try:
        h, m = int(hour), int(minute or 0)
    except ValueError:
        raise InvalidSlot(f"can't parse time {value!r}") from None
    if suffix == "pm" and h < 12:
        h += 12
    elif suffix == "am" and h == 12:
        h = 0
    return f"{h:02d}:{m:02d}"


def normalize_slot(day: str, time: str) -> tuple[str, str]:
    d, t = day.strip().lower(), parse_time(time)
    if d not in DAYS or t not in TIMES:
        raise InvalidSlot(f"{day} {time} is outside opening hours")
    return d, t


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BookingStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("pragma journal_mode=wal")
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            # take the write lock up front so two retries can't both miss the key
            conn.execute("begin immediate")
            yield conn
            conn.execute("commit")
        except BaseException:
            conn.execute("rollback")
            raise
        finally:
            conn.close()

    def _idempotent(
        self, key: str, request: dict, run: Callable[[sqlite3.Connection], Booking]
    ) -> Booking:
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        with self._tx() as conn:
            row = conn.execute(
                "select request_hash, response from idempotency where key = ?", (key,)
            ).fetchone()
            if row:
                if row["request_hash"] != digest:
                    raise KeyReused(f"key {key} was used for a different request")
                return Booking(**json.loads(row["response"]))

            booking = run(conn)
            conn.execute(
                "insert into idempotency values (?, ?, ?, ?)",
                (key, digest, json.dumps(asdict(booking)), _now()),
            )
            return booking

    def book(self, key: str, name: str, day: str, time: str) -> Booking:
        day, time = normalize_slot(day, time)
        name = name.strip()

        def run(conn):
            booking = Booking(uuid.uuid4().hex[:8], name, day, time)
            try:
                conn.execute(
                    "insert into bookings (id, name, day, time, status, created_at) "
                    "values (?, ?, ?, ?, ?, ?)",
                    (booking.id, name, day, time, booking.status, _now()),
                )
            except sqlite3.IntegrityError:
                raise SlotTaken(f"{day} {time} is already booked") from None
            return booking

        return self._idempotent(key, {"op": "book", "name": name, "day": day, "time": time}, run)

    def get(self, booking_id: str) -> Booking:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "select id, name, day, time, status from bookings where id = ?", (booking_id,)
            ).fetchone()
        if not row:
            raise NotFound(booking_id)
        return Booking(**row)

    def bookings(self, status: str | None = None) -> list[Booking]:
        sql = "select id, name, day, time, status from bookings"
        args: tuple = ()
        if status:
            sql += " where status = ?"
            args = (status,)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql + " order by created_at", args).fetchall()
        return [Booking(**r) for r in rows]

    def available(self, day: str) -> list[str]:
        d = day.strip().lower()
        if d not in DAYS:
            raise InvalidSlot(f"we're closed on {day}")
        with closing(self._connect()) as conn:
            taken = {
                r["time"]
                for r in conn.execute(
                    "select time from bookings where day = ? and status = 'confirmed'", (d,)
                )
            }
        return [t for t in TIMES if t not in taken]
