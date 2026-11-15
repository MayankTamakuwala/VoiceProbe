import json
import logging
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
- Read back the day and time and get a yes before calling book_appointment.
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
                async for part in self.llm.chat(self.messages, tools=TOOLS):
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
            if not said and calls[0].name in FILLERS:
                filler = FILLERS[calls[0].name]
                self.messages[-1]["content"] = filler
                yield filler
            for call in calls:
                log.info("tool %s %s", call.name, call.arguments)
                if call.name == "cancel_appointment" and not _asked_to_cancel(user_text):
                    # 8b models love to "cancel" on goodbye, don't trust them with it
                    log.warning("blocked cancel, caller said: %r", user_text)
                    result = {"error": "caller did not ask to cancel, booking kept"}
                else:
                    result = await self.tools.run(call.name, call.arguments)
                self.messages.append(
                    {"role": "tool", "tool_name": call.name, "content": json.dumps(result)}
                )

        log.warning("hit max tool rounds")

    def interrupted(self, heard: str) -> None:
        last = self.messages[-1]
        if last["role"] == "assistant":
            heard = heard.strip().rstrip(".!?, ")
            last["content"] = f"{heard}..." if heard else ""

    def transcript(self) -> list[dict]:
        return [m for m in self.messages if m["role"] in ("user", "assistant") and m["content"]]
