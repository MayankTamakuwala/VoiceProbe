import httpx
import pytest

from voiceprobe.agent.tools import TOOLS, BookingTools
from voiceprobe.booking.api import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(str(tmp_path / "bookings.db"))


def make_tools(transport, call_id="call-1"):
    client = httpx.AsyncClient(transport=transport, base_url="http://booking")
    return BookingTools(call_id, client=client)


@pytest.fixture
def tools(app):
    return make_tools(httpx.ASGITransport(app=app))


def test_tool_schemas():
    names = [t["function"]["name"] for t in TOOLS]
    assert names == [
        "check_availability",
        "book_appointment",
        "reschedule_appointment",
        "cancel_appointment",
    ]


async def test_book_then_reschedule(tools, app):
    r = await tools.run("book_appointment", {"name": "Sam", "day": "Tuesday", "time": "3pm"})
    assert r["time"] == "15:00"
    r = await tools.run("reschedule_appointment", {"day": "wednesday", "time": "10:00"})
    assert (r["day"], r["time"]) == ("wednesday", "10:00")
    assert [b.day for b in app.state.store.bookings()] == ["wednesday"]
    assert [h.name for h in tools.history] == ["book_appointment", "reschedule_appointment"]


async def test_duplicate_tool_call_books_once(tools, app):
    args = {"name": "Sam", "day": "tuesday", "time": "15:00"}
    first = await tools.run("book_appointment", args)
    second = await tools.run("book_appointment", args)
    assert first["id"] == second["id"]
    assert len(app.state.store.bookings()) == 1


async def test_second_booking_becomes_reschedule(tools, app):
    await tools.run("book_appointment", {"name": "Marcus Bell", "day": "tuesday", "time": "15:00"})
    r = await tools.run("book_appointment", {"name": "Marcus", "day": "wednesday", "time": "10:00"})
    assert (r["day"], r["time"]) == ("wednesday", "10:00")
    assert "no second booking" in r["note"]
    assert [(b.day, b.status) for b in app.state.store.bookings()] == [("wednesday", "confirmed")]
    assert [h.name for h in tools.history] == ["book_appointment", "reschedule_appointment"]


async def test_plain_tools_book_twice(app):
    tools = make_tools(httpx.ASGITransport(app=app))
    tools.move_rebookings = False
    await tools.run("book_appointment", {"name": "Marcus Bell", "day": "tuesday", "time": "15:00"})
    await tools.run("book_appointment", {"name": "Marcus Bell", "day": "wednesday", "time": "10:00"})
    assert len(app.state.store.bookings("confirmed")) == 2


async def test_new_booking_after_cancel_or_for_someone_else(tools, app):
    await tools.run("book_appointment", {"name": "Owen Grant", "day": "monday", "time": "09:00"})
    await tools.run("book_appointment", {"name": "Ada Lee", "day": "monday", "time": "10:00"})
    await tools.run("cancel_appointment", {})
    await tools.run("book_appointment", {"name": "Ada Lee", "day": "friday", "time": "10:00"})
    assert len(app.state.store.bookings("confirmed")) == 2
    assert [h.name for h in tools.history][-1] == "book_appointment"


async def test_cancel_without_booking(tools):
    r = await tools.run("cancel_appointment", {})
    assert "no booking" in r["error"]


async def test_slot_taken_is_reported(app):
    a = make_tools(httpx.ASGITransport(app=app), "call-a")
    b = make_tools(httpx.ASGITransport(app=app), "call-b")
    await a.run("book_appointment", {"name": "Sam", "day": "monday", "time": "09:00"})
    r = await b.run("book_appointment", {"name": "Alex", "day": "monday", "time": "09:00"})
    assert "already booked" in r["error"]


async def test_lost_response_retry_does_not_double_book(app):
    asgi = httpx.ASGITransport(app=app)
    seen = 0

    async def flaky(request):
        nonlocal seen
        seen += 1
        resp = await asgi.handle_async_request(request)
        if seen == 1:
            # booking went through but the caller never hears back
            await resp.aread()
            raise httpx.ReadTimeout("lost", request=request)
        return resp

    tools = make_tools(httpx.MockTransport(flaky))
    r = await tools.run("book_appointment", {"name": "Sam", "day": "friday", "time": "11:00"})
    assert r["status"] == "confirmed"
    assert tools.history[-1].attempts == 2
    assert len(app.state.store.bookings()) == 1


async def test_gives_up_after_retries():
    async def down(request):
        return httpx.Response(503)

    tools = make_tools(httpx.MockTransport(down))
    r = await tools.run("check_availability", {"day": "monday"})
    assert "error" in r
    assert tools.history[-1].attempts == 3
