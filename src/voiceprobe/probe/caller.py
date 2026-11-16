import re

from voiceprobe.agent.llm import Ollama
from voiceprobe.probe.scenario import Scenario

PROMPT = """\
You are role-playing a caller phoning a dental clinic. The other side is the clinic's receptionist.
You are {name}. {style}

What you want from this call, in order:
{goals}

How to behave:
- Talk like a real person on the phone: one or two short sentences per turn.
- Only write your own words. No stage directions, no quotes, no "Caller:" labels.
- Give your name when asked. Don't make up other details unless needed.
- If the receptionist can't do what you asked, react like a normal person would.
- When everything you wanted is done, say a short goodbye and then write [END].
"""

_END = re.compile(r"\[\s*end\s*\]", re.IGNORECASE)
_LABEL = re.compile(r"^\s*(caller|me|you|[A-Z][a-z]+(?: [A-Z][a-z]+)?)\s*:\s*")
_STAGE = re.compile(r"\*[^*]*\*|\([^)]*\)")


def clean(text: str) -> tuple[str, bool]:
    ended = bool(_END.search(text))
    text = _END.sub("", text)
    # 8b sometimes writes both sides, keep only the first line
    text = text.strip().split("\n")[0]
    text = _LABEL.sub("", text)
    text = _STAGE.sub("", text)
    text = text.strip().strip('"').strip()
    return re.sub(r"\s{2,}", " ", text), ended


class Caller:
    def __init__(self, scenario: Scenario, llm: Ollama):
        self.scenario = scenario
        self.llm = llm
        goals = "\n".join(f"{i}. {g}" for i, g in enumerate(scenario.goal, 1))
        system = PROMPT.format(name=scenario.name, style=scenario.style, goals=goals)
        self.messages: list[dict] = [{"role": "system", "content": system}]
        self.turns = 0
        self.done = False

    async def next_line(self, agent_said: str) -> str:
        """Returns what to say next. Empty string means hang up."""
        self.messages.append({"role": "user", "content": agent_said.strip() or "(silence)"})
        line, ended = "", False
        for _ in range(2):
            line, ended = clean(await self.llm.complete(self.messages))
            if line or ended:
                break
        if not line and not ended:
            line = "Sorry, could you say that again?"
        self.messages.append({"role": "assistant", "content": line or "[END]"})
        self.turns += 1
        if ended or self.turns >= self.scenario.max_turns:
            self.done = True
        return line

    def interrupted(self, agent_partial: str, said: str) -> None:
        """Record a scripted barge-in so the llm knows it already said it."""
        if agent_partial.strip():
            self.messages.append({"role": "user", "content": agent_partial.strip() + "..."})
        self.messages.append({"role": "assistant", "content": said})
