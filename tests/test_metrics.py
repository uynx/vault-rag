import pytest

from vault_rag.metrics import cohen_kappa, mean_metrics, recall_at_k, regressions, reciprocal_rank


def test_recall_at_k_counts_gold_notes_found_in_top_k():
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(ranked, {"a", "b"}, 2) == 0.5
    assert recall_at_k(ranked, {"a", "b"}, 4) == 1.0
    assert recall_at_k(ranked, {"z"}, 4) == 0.0


def test_reciprocal_rank_uses_first_gold_note():
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_mean_metrics_averages_over_questions():
    runs = [(["a", "b"], {"a"}), (["x", "a"], {"a"})]
    m = mean_metrics(runs, ks=(1, 2))
    assert m == {"recall@1": 0.5, "recall@2": 1.0, "mrr": 0.75}


def test_cohen_kappa_matches_worked_example():
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_edges():
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == 1.0
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0


def test_regressions_lists_every_metric_below_baseline_minus_tolerance():
    base = {"hybrid": {"recall@3": 0.9, "mrr": 0.8}, "bm25": {"mrr": 0.7}}
    cur = {"hybrid": {"recall@3": 0.85, "mrr": 0.8}, "bm25": {"mrr": 0.699}}
    assert regressions(cur, base, tol=0.01) == ["hybrid/recall@3: 0.8500 < baseline 0.9000"]
    assert regressions(base, base, tol=0.0) == []


def test_regressions_flags_a_metric_that_disappeared():
    out = regressions({"hybrid": {}}, {"hybrid": {"mrr": 0.8}}, tol=0.0)
    assert out == ["hybrid/mrr: missing, baseline 0.8000"]
