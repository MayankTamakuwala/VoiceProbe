import httpx

from voiceprobe.agent.dialog import GREETING, SYSTEM_PROMPT
from voiceprobe.agent.tools import TOOLS
from voiceprobe.probe.adapters import ELEVENLABS_API

# everything the adapter listens for, plus tool responses for the dashboard
CLIENT_EVENTS = [
    "conversation_initiation_metadata",
    "audio",
    "interruption",
    "user_transcript",
    "agent_response",
    "client_tool_call",
    "agent_tool_response",
    "ping",
]


class ElevenLabsError(Exception):
    pass


def _check(r: httpx.Response) -> dict:
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        if isinstance(detail, dict):
            detail = detail.get("message", detail)
        raise ElevenLabsError(f"{r.request.method} {r.url.path} failed ({r.status_code}): {detail}")
    return r.json()


def tool_config(tool: dict) -> dict:
    fn = tool["function"]
    config = {
        "type": "client",
        "name": fn["name"],
        "description": fn["description"],
        "expects_response": True,
        "response_timeout_secs": 20,
    }
    params = fn["parameters"]
    if params["properties"]:
        config["parameters"] = {
            "type": "object",
            "properties": params["properties"],
            "required": params["required"],
        }
    return {"tool_config": config}


def agent_config(tool_ids: list[str], llm: str, max_seconds: int, name: str) -> dict:
    # same prompt and greeting as the reference agent so only the stack differs
    return {
        "name": name,
        "conversation_config": {
            "agent": {
                "first_message": GREETING,
                "language": "en",
                "prompt": {"prompt": SYSTEM_PROMPT, "llm": llm, "tool_ids": tool_ids},
            },
            "tts": {"agent_output_audio_format": "pcm_16000"},
            "asr": {"user_input_audio_format": "pcm_16000"},
            "conversation": {"client_events": CLIENT_EVENTS, "max_duration_seconds": max_seconds},
        },
    }


async def create_agent(
    api_key: str,
    llm: str = "gemini-2.5-flash",
    max_seconds: int = 150,
    name: str = "voiceprobe maple dental",
) -> str:
    headers = {"xi-api-key": api_key}
    async with httpx.AsyncClient(base_url=ELEVENLABS_API, headers=headers, timeout=30.0) as client:
        tool_ids = []
        for tool in TOOLS:
            r = await client.post("/v1/convai/tools", json=tool_config(tool))
            tool_ids.append(_check(r)["id"])
        r = await client.post("/v1/convai/agents/create", json=agent_config(tool_ids, llm, max_seconds, name))
        return _check(r)["agent_id"]
