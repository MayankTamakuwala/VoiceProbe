import json
import logging
import re
from collections.abc import AsyncIterator

from voiceprobe.agent.llm import Ollama, ToolCall
from voiceprobe.agent.text import SentenceBuffer
from voiceprobe.agent.tools import TOOLS, BookingTools

log = logging.getLogger(__name__)

GREETING = "Thanks for calling Maple Dental, how can I help you today?"

SYSTEM_PROMPT = """\
You are the receptionist at Maple Dental, answering the phone.
The clinic is open Monday to Friday, appointments are on the hour from 9:00 to 16:00.

How to talk:
- This is a phone call. Keep every reply to one or two short sentences.
- No lists, no markdown, no emojis. Say times like "3 pm".
- If the caller cuts you off, drop what you were saying and listen to them.

Booking:
- You need the caller's name, a day and a time before booking.
- Check availability first if the caller is unsure or the slot might be taken.
- Read back the name, day and time and get a yes before calling book_appointment.
- If they change their mind after booking, use reschedule_appointment, don't book twice.
- Only cancel when the caller clearly asks to cancel. Saying thanks or goodbye is not a cancellation.
- Never claim something is booked unless the tool said so.
"""

# said while a tool runs so the caller isn't sitting in silence
FILLERS = {
    "check_availability": "Let me check.",
    "book_appointment": "One moment, booking that now.",
    "reschedule_appointment": "Sure, let me move that.",
}


_CANCEL_WORDS = ("cancel", "call it off", "call off", "don't need", "do not need", "scrap")


def _asked_to_cancel(text: str) -> bool:
    text = text.lower()
    return any(w in text for w in _CANCEL_WORDS)


# things the model puts in the name field when it doesn't have one
_NOT_NAMES = {
    "the", "a", "my", "mr", "mrs", "ms", "dr",
    "caller", "patient", "customer", "user", "unknown", "name",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if len(w) > 1}


def _name_given(name: str, said: str) -> bool:
    return bool((_words(name) - _NOT_NAMES) & _words(said))


class Dialog:
    def __init__(self, llm: Ollama, tools: BookingTools, max_rounds: int = 3):
        self.llm = llm
        self.tools = tools
        self.max_rounds = max_rounds
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": GREETING},
        ]

    async def reply(self, user_text: str) -> AsyncIterator[str]:
        self.messages.append({"role": "user", "content": user_text})

        for _ in range(self.max_rounds):
            buf = SentenceBuffer()
            said: list[str] = []
            calls: list[ToolCall] = []
            done = False
            try:
                async for part in self.llm.chat(self.messages, tools=self._tools()):
                    if isinstance(part, ToolCall):
                        calls.append(part)
                        continue
                    said.append(part)
                    for sentence in buf.push(part):
                        yield sentence
                tail = buf.flush()
                if tail:
                    yield tail
                done = True
            finally:
                # if we got cancelled mid reply (barge-in) still keep what was generated,
                # interrupted() trims it down to what the caller actually heard
                msg = {"role": "assistant", "content": "".join(said).strip()}
                if calls and done:
                    msg["tool_calls"] = [
                        {"function": {"name": c.name, "arguments": c.arguments}} for c in calls
                    ]
                self.messages.append(msg)

            if not calls:
                return
            # check first, so we don't say "booking that now" and then not book
            problems = [self._check(c) for c in calls]
            if not said and problems[0] is None and calls[0].name in FILLERS:
                filler = FILLERS[calls[0].name]
                self.messages[-1]["content"] = filler
                yield filler
            for call, problem in zip(calls, problems):
                log.info("tool %s %s", call.name, call.arguments)
                if problem:
                    log.warning("blocked %s %s: %s", call.name, call.arguments, problem)
                    result = self.tools.blocked(call.name, call.arguments, problem)
                else:
                    result = await self.tools.run(call.name, call.arguments)
                self.messages.append(
                    {"role": "tool", "tool_name": call.name, "content": json.dumps(result)}
                )

        log.warning("hit max tool rounds")

    def _check(self, call: ToolCall) -> str | None:
        if call.name == "cancel_appointment" and not self._cancel_requested():
            # 8b models love to "cancel" on goodbye, don't trust them with it
            return "caller did not ask to cancel, booking kept"
        if call.name == "book_appointment" and not _name_given(
            str(call.arguments.get("name", "")), self._caller_said()
        ):
            # it will happily book under "caller" or a made up name
            return "the caller hasn't told you their name yet, ask for it first"
        return None

    def _caller_said(self) -> str:
        return " ".join(m["content"] for m in self.messages if m["role"] == "user")

    def _cancel_requested(self) -> bool:
        # this turn or the one before, so "yes" after "are you sure?" still counts
        said = [m["content"] for m in self.messages if m["role"] == "user"]
        return any(_asked_to_cancel(t) for t in said[-2:])

    def _tools(self) -> list[dict]:
        # if cancel isn't on offer it can't reach for it on goodbye, which also
        # saves a whole llm round every time it would have been blocked
        if self._cancel_requested():
            return TOOLS
        return [t for t in TOOLS if t["function"]["name"] != "cancel_appointment"]

    def interrupted(self, heard: str) -> None:
        last = self.messages[-1]
        if last["role"] == "assistant":
            heard = heard.strip().rstrip(".!?, ")
            last["content"] = f"{heard}..." if heard else ""

    def transcript(self) -> list[dict]:
        return [m for m in self.messages if m["role"] in ("user", "assistant") and m["content"]]
