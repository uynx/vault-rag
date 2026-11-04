import json

from fakes import fake_embed
from vault_rag.cli import main


def llm(prompt):
    if "Write a fluent answer" in prompt:
        return "perturbed answer"
    if "Answer to grade:" in prompt:
        return '{"reason": "", "score": %d}' % (0 if "perturbed" in prompt.split("Answer to grade:")[1] else 2)
    if "Answer A:" in prompt:
        return '{"winner": "%s"}' % ("B" if "perturbed" in prompt.split("Answer B:")[0] else "A")
    return "system answer [1]"


def args(vault, cmd, *extra):
    return [cmd, "--corpus", str(vault / "corpus"), "--questions", str(vault / "q.jsonl"),
            "--baseline", str(vault / "baseline.json"), "--out", str(vault / "out"), "--cache", str(vault / "cache"),
            "--labelset", str(vault / "ls.jsonl"), "--labels", str(vault / "labels.jsonl"), "--workers", "1", *extra]


def test_labelset_then_judge_check_round_trip(vault, capsys):
    assert main(args(vault, "labelset", "--n-pairs", "1", "--n-weak", "1"), embed=fake_embed, llm=llm) == 0
    rows = [json.loads(x) for x in (vault / "ls.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert main(args(vault, "judge-check"), embed=fake_embed, llm=llm) == 2
    labels = [{"id": r["id"], "label": 0 if r["origin"].startswith("perturbed") else 2, "labeler": "t"} for r in rows]
    (vault / "labels.jsonl").write_text("\n".join(json.dumps(x) for x in labels))
    assert main(args(vault, "judge-check", "--update-baseline"), embed=fake_embed, llm=llm) == 0
    assert main(args(vault, "judge-check", "--offline"), embed=fake_embed, llm=llm) == 0
    out = json.loads((vault / "out" / "judge.json").read_text())
    assert out["validation"]["kappa_binary"] == 1.0 and out["n_pairs"] == 1
    assert "Cohen's kappa" in (vault / "out" / "judge.md").read_text()
    assert json.loads((vault / "baseline.json").read_text())["judge/validation"]["kappa_binary"] == 1.0


def test_eval_with_answers_gates_on_cached_llm_offline(vault):
    assert main(args(vault, "eval", "--answers", "--update-baseline", "-k", "2"), embed=fake_embed, llm=llm) == 0
    base = json.loads((vault / "baseline.json").read_text())
    assert set(base) == {f"{g}/{m}" for g in ("retrieval", "answers") for m in ("bm25", "dense", "hybrid")}
    assert main(args(vault, "eval", "--answers", "--offline", "-k", "2"), embed=fake_embed, llm=None) == 0
    assert (vault / "out" / "answers_hybrid.jsonl").exists()


def test_judge_markdown_survives_a_run_with_no_pairs():
    from vault_rag.cli import judge_markdown

    v = {"n": 1, "agreement": 1.0, "agreement_ci95": (0.2, 1.0), "kappa_3class": 1.0, "binary_agreement": 1.0,
         "kappa_binary": 1.0, "confusion": {}}
    out = {"validation": v, "validation_by_origin": {"generated": v}, "n_pairs": 0, "position": {"n": 0},
           "position_padded_bad": {"n": 0}, "length": {"n": 1, "mean_delta": 0.0, "pct_raised": 0.0, "pct_lowered": 0.0,
                                                      "wrong_promoted": None, "spearman_len_score": None,
                                                      "spearman_within_label": {}}}
    assert "no pair has both" in judge_markdown(out)
