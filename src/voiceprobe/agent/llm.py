import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: dict


class Ollama:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        client: httpx.AsyncClient | None = None,
    ):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.temperature = temperature
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    async def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[str | ToolCall]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": "30m",
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools

        async with self.client.stream("POST", f"{self.host}/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                msg = json.loads(line).get("message", {})
                if msg.get("content"):
                    yield msg["content"]
                for call in msg.get("tool_calls") or []:
                    fn = call["function"]
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args)
                    yield ToolCall(fn["name"], args)

    async def complete(self, messages: list[dict]) -> str:
        return "".join([p async for p in self.chat(messages) if isinstance(p, str)])

    async def warmup(self) -> None:
        # first request after a cold start loads the model, which takes seconds
        await self.client.post(
            f"{self.host}/api/chat",
            json={"model": self.model, "messages": [], "keep_alive": "30m"},
        )

    async def close(self) -> None:
        await self.client.aclose()
