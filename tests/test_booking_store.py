from concurrent.futures import ThreadPoolExecutor

import pytest

from voiceprobe.booking.store import (
    BookingStore,
    InvalidSlot,
    KeyReused,
    SlotTaken,
    parse_time,
)


@pytest.fixture
def store(tmp_path):
    return BookingStore(tmp_path / "bookings.db")


@pytest.mark.parametrize(
    "raw,expected",
    [("15:00", "15:00"), ("3pm", "15:00"), ("3 PM", "15:00"), ("12pm", "12:00"), ("9", "09:00")],
)
def test_parse_time(raw, expected):
    assert parse_time(raw) == expected


def test_book_and_get(store):
    b = store.book("k1", "Sam", "Tuesday", "3pm")
    assert (b.day, b.time, b.status) == ("tuesday", "15:00", "confirmed")
    assert store.get(b.id) == b


def test_slot_taken(store):
    store.book("k1", "Sam", "tuesday", "15:00")
    with pytest.raises(SlotTaken):
        store.book("k2", "Alex", "tuesday", "15:00")


def test_outside_hours(store):
    with pytest.raises(InvalidSlot):
        store.book("k1", "Sam", "saturday", "10:00")
    with pytest.raises(InvalidSlot):
        store.book("k1", "Sam", "monday", "8pm")


def test_retry_returns_same_booking(store):
    first = store.book("k1", "Sam", "tuesday", "15:00")
    again = store.book("k1", "Sam", "tuesday", "3pm")
    assert again == first
    assert len(store.bookings()) == 1


def test_key_reuse_with_other_payload(store):
    store.book("k1", "Sam", "tuesday", "15:00")
    with pytest.raises(KeyReused):
        store.book("k1", "Sam", "wednesday", "15:00")


def test_concurrent_retries_book_once(store):
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: store.book("k1", "Sam", "friday", "10:00"), range(8)))
    assert len({r.id for r in results}) == 1
    assert len(store.bookings()) == 1


def test_available(store):
    store.book("k1", "Sam", "monday", "09:00")
    free = store.available("Monday")
    assert "09:00" not in free
    assert len(free) == 7
