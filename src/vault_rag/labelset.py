import random

from .answer import answer
from .llm import pmap

CHANGE = ("Write a fluent answer of similar length that sounds confident but is wrong: change exactly one key fact "
          "(a number, name, date, place or reason) to a plausible different value and keep everything else.")
OMIT = ("Write a fluent answer that gives only part of the correct answer: leave out one of its facts completely "
        "and add nothing new.")


def perturb_prompt(mode: str, question: str, gold: str) -> str:
    task = CHANGE if mode == "change" else OMIT
    return (f"Below are a question and its correct answer. {task}\n\n"
            f"Question: {question}\nCorrect answer: {gold}\n\nReply with the answer only.")


def build_labelset(retriever, questions, llm, n_pairs: int = 16, n_weak: int = 8, seed: int = 7, workers: int = 8):
    rng = random.Random(seed)
    weak_pool = [q for q in questions if retriever.rank_notes(q["question"], "bm25")[0] not in q["gold_sources"]]
    weak = rng.sample(weak_pool, min(n_weak, len(weak_pool)))
    rest = [q for q in questions if q not in weak]
    weak += rng.sample(rest, n_weak - len(weak))
    rest = [q for q in rest if q not in weak]
    pairs = rng.sample(rest, n_pairs)

    def make(task):
        q, kind = task
        if kind == "weak":
            return [(q, answer(q["question"], retriever, llm, "bm25", 1)["answer"], "system:bm25-k1")]
        good = answer(q["question"], retriever, llm, "hybrid", 5)["answer"]
        mode = "omit" if " and " in q["question"] or "," in q["question"] else "change"
        bad = llm(perturb_prompt(mode, q["question"], q["gold_answer"]))
        return [(q, good, "system:hybrid-k5"), (q, bad, f"perturbed:{mode}")]

    made = [x for batch in pmap(make, [(q, "weak") for q in weak] + [(q, "pair") for q in pairs], workers) for x in batch]
    rng.shuffle(made)
    return [{"id": f"l{i:02d}", "qid": q["id"], "question": q["question"], "gold_answer": q["gold_answer"],
             "answer": a, "origin": o} for i, (q, a, o) in enumerate(made, 1)]
