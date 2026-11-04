from fakes import QUESTIONS, fake_embed
from vault_rag.corpus import load_corpus
from vault_rag.judge import PAD
from vault_rag.judgecheck import run_check
from vault_rag.labelset import build_labelset
from vault_rag.retrieval import Retriever


def llm(prompt):
    if "Write a fluent answer" in prompt:
        return "perturbed answer"
    return "system answer [1]"


def test_build_labelset_mixes_system_and_perturbed_answers_deterministically(vault):
    r = Retriever(load_corpus(vault / "corpus"), fake_embed)
    a = build_labelset(r, QUESTIONS, llm, n_pairs=1, n_weak=1, seed=3)
    b = build_labelset(r, QUESTIONS, llm, n_pairs=1, n_weak=1, seed=3)
    assert a == b and len(a) == 3
    assert len({x["id"] for x in a}) == 3
    origins = sorted(x["origin"] for x in a)
    assert origins[0].startswith("perturbed:") and "system:bm25-k1" in origins and "system:hybrid-k5" in origins
    weak = next(x for x in a if x["origin"] == "system:bm25-k1")
    assert weak["qid"] == "q3"
    pair = [x for x in a if x is not weak]
    assert pair[0]["qid"] == pair[1]["qid"] != "q3"
    assert all(k in a[0] for k in ("id", "qid", "question", "gold_answer", "answer", "origin"))
    assert not any("score" in x or "label" in x for x in a)


def test_run_check_validates_judge_and_runs_both_bias_checks():
    rows = [
        {"id": "a1", "qid": "q1", "question": "Q1", "gold_answer": "G1", "answer": "good one", "origin": "x"},
        {"id": "a2", "qid": "q1", "question": "Q1", "gold_answer": "G1", "answer": "bad one", "origin": "x"},
        {"id": "a3", "qid": "q2", "question": "Q2", "gold_answer": "G2", "answer": "good two", "origin": "x"},
        {"id": "a4", "qid": "q2", "question": "Q2", "gold_answer": "G2", "answer": "bad two", "origin": "x"},
    ]
    labels = {"a1": 2, "a2": 0, "a3": 2, "a4": 0}

    def fake(prompt):
        if "Answer to grade:" in prompt:
            return '{"reason": "", "score": %d}' % (2 if "good" in prompt.split("Answer to grade:")[1] else 0)
        a = prompt.split("Answer A:")[1].split("Answer B:")[0]
        return '{"winner": "%s"}' % ("A" if "good" in a else "B")

    out = run_check(rows, labels, fake, workers=1)
    assert out["validation"]["n"] == 4 and out["validation"]["kappa_binary"] == 1.0
    assert out["position"]["n"] == 2 and out["position"]["acc_overall"] == 1.0
    assert out["position_padded_bad"]["acc_overall"] == 1.0
    assert out["length"]["n"] == 4 and out["length"]["mean_delta"] == 0.0
    assert PAD  # padding text is shared with the judge module
