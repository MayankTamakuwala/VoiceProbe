import os
from dataclasses import asdict

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from voiceprobe.booking.store import (
    BookingError,
    BookingStore,
    InvalidSlot,
    KeyReused,
    NotFound,
    SlotTaken,
)

STATUS = {InvalidSlot: 422, KeyReused: 422, NotFound: 404, SlotTaken: 409}


class BookRequest(BaseModel):
    name: str
    day: str
    time: str


class SlotRequest(BaseModel):
    day: str
    time: str


def idempotency_key(key: str | None = Header(None, alias="Idempotency-Key")) -> str:
    if not key:
        raise HTTPException(400, "Idempotency-Key header is required")
    return key


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(title="bookings")
    store = BookingStore(db_path or os.getenv("BOOKING_DB", "data/bookings.db"))
    app.state.store = store

    @app.exception_handler(BookingError)
    async def booking_error(request: Request, exc: BookingError):
        return JSONResponse({"error": str(exc)}, status_code=STATUS.get(type(exc), 400))

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/slots/{day}")
    def slots(day: str):
        return {"day": day.lower(), "available": store.available(day)}

    @app.get("/bookings")
    def bookings(status: str | None = None):
        return [asdict(b) for b in store.bookings(status)]

    @app.get("/bookings/{booking_id}")
    def get_booking(booking_id: str):
        return asdict(store.get(booking_id))

    @app.post("/bookings", status_code=201)
    def book(req: BookRequest, key: str = Depends(idempotency_key)):
        return asdict(store.book(key, req.name, req.day, req.time))

    @app.post("/bookings/{booking_id}/cancel")
    def cancel(booking_id: str, key: str = Depends(idempotency_key)):
        return asdict(store.cancel(key, booking_id))

    @app.post("/bookings/{booking_id}/reschedule")
    def reschedule(booking_id: str, req: SlotRequest, key: str = Depends(idempotency_key)):
        return asdict(store.reschedule(key, booking_id, req.day, req.time))

    # used by the test harness to start every call from a known state
    @app.post("/admin/reset")
    def reset():
        store.reset()
        return {"ok": True}

    return app
