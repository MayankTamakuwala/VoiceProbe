import re

_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}


def normalize(text: str) -> list[str]:
    t = text.lower().replace("a.m.", "am").replace("p.m.", "pm")
    t = re.sub(r"(\d)(am|pm)\b", r"\1 \2", t)
    t = re.sub(r"(\d):00\b", r"\1", t)
    t = re.sub(r"[^\w\s']", " ", t)
    return [_NUMBERS.get(w, w) for w in t.split()]


def distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def count(pairs: list[tuple[str, str]]) -> tuple[int, int]:
    errors = words = 0
    for said, heard in pairs:
        ref = normalize(said)
        errors += distance(ref, normalize(heard))
        words += len(ref)
    return errors, words


def wer(pairs: list[tuple[str, str]]) -> float | None:
    """Corpus wer over (said, heard) pairs, so long turns weigh more than short ones."""
    errors, words = count(pairs)
    return errors / words if words else None
