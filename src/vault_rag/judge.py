import json
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

from .llm import pmap
from .metrics import cohen_kappa

PAD = (
    " For additional context, the notes cover this topic across several sections, and the details above are "
    "drawn from the passages provided. Nothing further in the notes changes this answer, though readers may "
    "want to consult the relevant note directly for the surrounding discussion and any related figures."
)

RUBRIC = (
    "You grade an answer to a question about a notes vault by comparing it with the gold answer.\n"
    "Score 2: the answer states every fact in the gold answer (wording, order and extra correct detail do not "
    "matter) and contradicts none of them.\n"
    "Score 1: it states some but not all of the gold facts, or states all of them and also contradicts one.\n"
    "Score 0: it misses the key facts, is wrong, or says the notes do not say.\n"
    "Judge facts only. Ignore style and length. Citation markers like [1] are not facts."
)


def judge_prompt(question: str, gold: str, answer: str) -> str:
    return (f"{RUBRIC}\n\nQuestion: {question}\nGold answer: {gold}\n\nAnswer to grade:{answer}\n\n"
            'Reply with JSON only: {"reason": "<one sentence>", "score": <0, 1 or 2>}')


def pair_prompt(question: str, gold: str, first: str, second: str) -> str:
    return ("Two answers to a question about a notes vault are shown with the gold answer. Pick the one that is "
            "more correct against the gold answer. Judge facts only. Ignore style and length. "
            'Say "tie" only if they are equally correct.\n\n'
            f"Question: {question}\nGold answer: {gold}\n\nAnswer A: {first}\n\nAnswer B: {second}\n\n"
            'Reply with JSON only: {"winner": "A" or "B" or "tie"}')


def parse_score(text: str) -> int:
    hits = re.findall(r'"score"\s*:\s*(\d)', text)
    if not hits or int(hits[-1]) not in (0, 1, 2):
        raise ValueError(f"no valid score in {text[:80]!r}")
    return int(hits[-1])


def parse_winner(text: str) -> str:
    hits = re.findall(r'"winner"\s*:\s*"(A|B|tie)"', text)
    if not hits:
        raise ValueError(f"no valid winner in {text[:80]!r}")
    return hits[-1]


def judge_one(llm, question: str, gold: str, answer: str) -> int:
    return parse_score(llm(judge_prompt(question, gold, answer)))


def pad(answer: str) -> str:
    return answer + PAD


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = p + z * z / (2 * n)
    spread = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return (centre - spread) / (1 + z * z / n), (centre + spread) / (1 + z * z / n)


def validate(human: list[int], judge: list[int]) -> dict:
    n = len(human)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for h, j in zip(human, judge):
        confusion[str(h)][str(j)] += 1
    hb, jb = [h == 2 for h in human], [j == 2 for j in judge]
    agree = sum(h == j for h, j in zip(human, judge))
    return {"n": n, "agreement": agree / n, "agreement_ci95": wilson(agree, n),
            "kappa_3class": cohen_kappa(human, judge),
            "binary_agreement": sum(h == j for h, j in zip(hb, jb)) / n, "kappa_binary": cohen_kappa(hb, jb),
            "confusion": {h: dict(c) for h, c in confusion.items()}}


def position_bias(llm, pairs, workers: int = 8) -> dict:
    if not pairs:
        return {"n": 0}
    prompts = [pair_prompt(q, g, a, b)
               for q, g, good, bad in pairs for a, b in ((good, bad), (bad, good))]
    wins = pmap(lambda p: parse_winner(llm(p)), prompts, workers)
    first, second = wins[0::2], wins[1::2]
    picked_good = lambda w, good_first: w == ("A" if good_first else "B")
    gf = [picked_good(w, True) for w in first]
    gs = [picked_good(w, False) for w in second]
    flips = [(w1 == "tie") != (w2 == "tie") or (w1 != "tie" and picked_good(w1, True) != picked_good(w2, False))
             for w1, w2 in zip(first, second)]
    n = len(pairs)
    return {"n": n, "pick_first_rate": (first + second).count("A") / (2 * n), "acc_good_first": sum(gf) / n,
            "acc_good_second": sum(gs) / n, "flip_rate": sum(flips) / n, "acc_overall": (sum(gf) + sum(gs)) / (2 * n)}


def _rho(x, y):
    if len(x) < 5 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y)[0])


def length_bias(llm, rows, workers: int = 8) -> dict:
    prompts = [judge_prompt(q, g, a) for q, g, a, _ in rows] + [judge_prompt(q, g, pad(a)) for q, g, a, _ in rows]
    scores = pmap(lambda p: parse_score(llm(p)), prompts, workers)
    base, padded = np.array(scores[: len(rows)]), np.array(scores[len(rows):])
    labels = np.array([r[3] for r in rows])
    words = np.array([len(r[2].split()) for r in rows])
    wrong = labels < 2
    return {"n": len(rows), "mean_delta": float((padded - base).mean()), "pct_raised": float((padded > base).mean()),
            "pct_lowered": float((padded < base).mean()),
            "wrong_promoted": float((padded > base)[wrong].mean()) if wrong.any() else None,
            "spearman_len_score": _rho(words, base),
            "spearman_within_label": {str(lab): _rho(words[labels == lab], base[labels == lab])
                                      for lab in sorted(set(labels.tolist()))}}
