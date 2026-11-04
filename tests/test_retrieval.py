import numpy as np
import pytest

from vault_rag.retrieval import BM25, Dense, rrf


def test_bm25_matches_hand_computed_score():
    bm = BM25(["apple banana", "banana cherry"])
    s = bm.scores("apple")
    assert s[0] == pytest.approx(0.6931, abs=1e-4)
    assert s[1] == 0


def test_bm25_prefers_rare_term_and_ignores_unknown_words():
    bm = BM25(["the cat sat", "the dog sat", "the zebra sat"])
    s = bm.scores("the zebra qqq")
    assert s.argmax() == 2


def test_bm25_saturates_term_frequency():
    bm = BM25(["x " * 3 + "pad", "x " * 30 + "pad", "y pad pad"])
    s = bm.scores("x")
    assert s[1] > s[0] > 0 and s[1] < 2 * s[0]


def test_dense_ranks_by_cosine():
    vecs = {"a": [1.0, 0.0], "b": [0.0, 1.0], "q": [0.9, 0.1]}

    def embed(texts, query=False):
        v = np.array([vecs[t] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    d = Dense(["a", "b"], embed)
    assert d.scores("q").argmax() == 0


def test_rrf_rewards_agreement_between_lists():
    fused = rrf([np.array([0, 1, 2]), np.array([1, 2, 0])])
    assert fused[0] == 1
