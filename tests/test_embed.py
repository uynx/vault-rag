import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_real_model_ranks_paraphrase_and_caches(tmp_path):
    from vault_rag.embed import FastEmbedder

    e = FastEmbedder(cache=tmp_path)
    docs = ["The treasurer keeps the books.", "Water temperature rose in May."]
    first = e(docs)
    assert first.shape == (2, 384)
    assert np.allclose(np.linalg.norm(first, axis=1), 1, atol=1e-4)
    assert (e(["Who handles the money?"], query=True) @ first.T).argmax() == 0
    assert len(list((tmp_path / "vectors").glob("*.npy"))) == 2
    assert np.array_equal(e(docs), first)


def test_long_inputs_are_cached_in_slices_so_an_interrupted_run_keeps_progress(tmp_path):
    from vault_rag.embed import FastEmbedder

    e = FastEmbedder(cache=tmp_path)
    texts = [f"note number {i} about kelp" for i in range(300)]
    full = e(texts)
    assert full.shape == (300, 384)
    assert len(list((tmp_path / "vectors").glob("*.npy"))) == 2
    assert np.array_equal(e(texts[:256]), full[:256])
    assert len(list((tmp_path / "vectors").glob("*.npy"))) == 2
