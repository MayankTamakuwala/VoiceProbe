import re

_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_ABBREV = re.compile(r"(\b[a-z]\.|\b(dr|mr|mrs|ms|st)\.)$", re.IGNORECASE)
_MARKUP = re.compile(r"[*_#`]+")


def clean(text: str) -> str:
    return _MARKUP.sub("", text).strip()


class SentenceBuffer:
    """Collects streamed llm tokens and hands back whole sentences."""

    def __init__(self, min_chars: int = 2):
        self.min_chars = min_chars
        self._buf = ""

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        parts = _BOUNDARY.split(self._buf)
        out, pending = [], ""
        for part in parts[:-1]:
            pending = f"{pending} {part}".strip() if pending else part
            # "3 p.m. works" shouldn't split after "p.m."
            if _ABBREV.search(pending) or len(pending) < self.min_chars:
                continue
            out.append(clean(pending))
            pending = ""
        rest = parts[-1]
        self._buf = f"{pending} {rest}" if pending else rest
        return [s for s in out if s]

    def flush(self) -> str:
        text, self._buf = clean(self._buf), ""
        return text
