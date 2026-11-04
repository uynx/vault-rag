from collections import defaultdict

from .judge import judge_one, length_bias, pad, position_bias, validate
from .llm import pmap


def run_check(rows, labels: dict[str, int], llm, workers: int = 8) -> dict:
    scores = pmap(lambda r: judge_one(llm, r["question"], r["gold_answer"], r["answer"]), rows, workers)
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    pairs = []
    for rs in by_q.values():
        good = [r for r in rs if labels[r["id"]] == 2]
        bad = [r for r in rs if labels[r["id"]] < 2]
        if good and bad:
            pairs.append((good[0]["question"], good[0]["gold_answer"], good[0]["answer"], bad[0]["answer"]))
    return {"validation": validate([labels[r["id"]] for r in rows], scores),
            "scores": {r["id"]: s for r, s in zip(rows, scores)}, "n_pairs": len(pairs),
            "position": position_bias(llm, pairs, workers),
            "position_padded_bad": position_bias(llm, [(q, g, good, pad(bad)) for q, g, good, bad in pairs], workers),
            "length": length_bias(llm, [(r["question"], r["gold_answer"], r["answer"], labels[r["id"]]) for r in rows],
                                  workers)}
