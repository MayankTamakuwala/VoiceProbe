import pytest
from fastapi.testclient import TestClient

from voiceprobe.booking.api import create_app


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(str(tmp_path / "bookings.db")))


def book(client, key="k1", **body):
    payload = {"name": "Sam", "day": "tuesday", "time": "3pm", **body}
    return client.post("/bookings", json=payload, headers={"Idempotency-Key": key})


def test_requires_key(client):
    r = client.post("/bookings", json={"name": "Sam", "day": "tuesday", "time": "3pm"})
    assert r.status_code == 400


def test_book_and_retry(client):
    first = book(client)
    assert first.status_code == 201
    again = book(client)
    assert again.json() == first.json()
    assert len(client.get("/bookings").json()) == 1


def test_conflict(client):
    book(client, key="k1")
    r = book(client, key="k2", name="Alex")
    assert r.status_code == 409
    assert "already booked" in r.json()["error"]


def test_invalid_slot(client):
    assert book(client, day="sunday").status_code == 422


def test_slots(client):
    book(client)
    free = client.get("/slots/Tuesday").json()["available"]
    assert "15:00" not in free


def test_reschedule_and_cancel(client):
    bid = book(client).json()["id"]
    r = client.post(
        f"/bookings/{bid}/reschedule",
        json={"day": "wednesday", "time": "10:00"},
        headers={"Idempotency-Key": "k2"},
    )
    assert r.json()["day"] == "wednesday"
    r = client.post(f"/bookings/{bid}/cancel", headers={"Idempotency-Key": "k3"})
    assert r.json()["status"] == "cancelled"
    assert client.get(f"/bookings/{bid}").json()["status"] == "cancelled"


def test_unknown_booking(client):
    assert client.get("/bookings/nope").status_code == 404


def test_reset(client):
    book(client)
    assert client.post("/admin/reset").json() == {"ok": True}
    assert client.get("/bookings").json() == []
    # keys are forgotten too, so the same key books again
    assert book(client).status_code == 201
