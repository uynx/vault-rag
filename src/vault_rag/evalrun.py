import json
from pathlib import Path

from .answer import answer
from .judge import judge_one
from .llm import pmap
from .metrics import mean_metrics
from .retrieval import METHODS


def load_questions(path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def retrieval_metrics(retriever, questions, methods=METHODS, ks=(1, 3, 5)) -> dict:
    retriever.dense.warm([q["question"] for q in questions])
    types = sorted({q["type"] for q in questions})
    overall, by_type = {}, {}
    for m in methods:
        runs = [(retriever.rank_notes(q["question"], m), set(q["gold_sources"])) for q in questions]
        overall[m] = mean_metrics(runs, ks)
        by_type[m] = {t: mean_metrics([r for r, q in zip(runs, questions) if q["type"] == t], ks) for t in types}
    return {"overall": overall, "by_type": by_type}


def answer_metrics(retriever, questions, gen, judge_llm, method: str = "hybrid", k: int = 5, workers: int = 8):
    retriever.dense.warm([q["question"] for q in questions])

    def one(q):
        out = answer(q["question"], retriever, gen, method, k)
        score = judge_one(judge_llm, q["question"], q["gold_answer"], out["answer"])
        return {"id": q["id"], "type": q["type"], "question": q["question"], "gold_answer": q["gold_answer"],
                "score": score, "citation_hit": bool(set(out["cited_notes"]) & set(q["gold_sources"])), **out}

    rows = pmap(one, questions, workers)
    n = len(rows)
    return {"correct": sum(r["score"] == 2 for r in rows) / n, "mean_score": sum(r["score"] for r in rows) / (2 * n),
            "citation_hit": sum(r["citation_hit"] for r in rows) / n}, rows


def markdown_table(rows: dict[str, dict[str, float]]) -> str:
    cols = list(next(iter(rows.values())))
    lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    lines += [f"| {name} | " + " | ".join(f"{vals[c]:.3f}" for c in cols) + " |" for name, vals in rows.items()]
    return "\n".join(lines)
