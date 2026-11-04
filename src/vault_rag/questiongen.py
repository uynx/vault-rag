import json
import random
import re

from .llm import pmap

SECRETISH = re.compile(r"password|passwd|secret|api[_ -]?key|token|private key|credential|license number|ssn", re.I)

PROMPT = (
    "Read the passage from a notes vault and write one question that the passage answers, plus a short gold answer. "
    "The question must make sense without the passage, must not say \"passage\" or \"note\", and must not copy more "
    "than three words in a row from the passage.\n\nPassage:\n{text}\n\n"
    'Reply with JSON only: {{"question": "...", "answer": "..."}}'
)


def generate_questions(chunks, llm, n: int = 60, seed: int = 7, workers: int = 8) -> list[dict]:
    pool = [c for c in chunks if len(c.text.split()) >= 60 and not SECRETISH.search(c.text)]
    picked = random.Random(seed).sample(pool, min(n, len(pool)))

    def one(c):
        m = re.search(r"\{.*\}", llm(PROMPT.format(text=c.text)), re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        return (c, d["question"], d["answer"]) if d.get("question") and d.get("answer") else None

    made = [x for x in pmap(one, picked, workers) if x]
    return [{"id": f"g{i:02d}", "type": "synthetic", "question": q, "gold_sources": [c.note], "gold_answer": a}
            for i, (c, q, a) in enumerate(made, 1)]
