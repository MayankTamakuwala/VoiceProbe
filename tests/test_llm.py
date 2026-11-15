import os

import httpx
import pytest

from voiceprobe.agent.llm import Ollama, ToolCall
from voiceprobe.agent.tools import TOOLS


def ollama_up() -> bool:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        return httpx.get(f"{host}/api/version", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(not ollama_up(), reason="ollama not running")


async def test_streams_text():
    llm = Ollama(temperature=0)
    text = await llm.complete([{"role": "user", "content": "Say just the word hello."}])
    await llm.close()
    assert "hello" in text.lower()


async def test_emits_tool_call():
    llm = Ollama(temperature=0)
    messages = [
        {"role": "system", "content": "You book appointments. Use the tools."},
        {"role": "user", "content": "Please book Tuesday at 15:00 for Sam. Yes, that's confirmed."},
    ]
    parts = [p async for p in llm.chat(messages, tools=TOOLS)]
    await llm.close()
    calls = [p for p in parts if isinstance(p, ToolCall)]
    assert calls and calls[0].name == "book_appointment"
    assert calls[0].arguments["day"].lower() == "tuesday"
