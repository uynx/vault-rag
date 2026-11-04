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


def _retriever():
    from fakes import fake_embed
    from vault_rag.corpus import Chunk
    from vault_rag.retrieval import Retriever

    chunks = [
        Chunk("boats", 0, "Boats\nThe Gannet hauls two tonnes per trip"),
        Chunk("boats", 1, "Boats\nThe Skerry has a battery outboard"),
        Chunk("buyers", 0, "Buyers\nCorvane takes sixty percent of each harvest"),
        Chunk("nursery", 0, "Nursery\nSix tanks grow gametophytes under blue light"),
    ]
    return Retriever(chunks, fake_embed)


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid"])
def test_search_returns_best_chunk_first_for_every_method(method):
    r = _retriever()
    top = r.search("how many tonnes does the Gannet haul", method, k=2)
    assert len(top) == 2 and top[0].note == "boats" and top[0].idx == 0


def test_rank_notes_dedupes_chunks_in_rank_order():
    r = _retriever()
    notes = r.rank_notes("Skerry battery outboard Gannet", "bm25")
    assert notes[0] == "boats"
    assert sorted(notes) == ["boats", "buyers", "nursery"]


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        _retriever().search("x", "tfidf")
