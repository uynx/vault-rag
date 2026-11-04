import numpy as np
import pytest



@pytest.mark.slow
def test_real_model_ranks_paraphrase_and_caches(tmp_path):
    from vault_rag.embed import FastEmbedder

    e = FastEmbedder(vectors=tmp_path / 'vectors', models=tmp_path / 'models')
    docs = ["The treasurer keeps the books.", "Water temperature rose in May."]
    first = e(docs)
    assert first.shape == (2, 384)
    assert np.allclose(np.linalg.norm(first, axis=1), 1, atol=1e-4)
    assert (e(["Who handles the money?"], query=True) @ first.T).argmax() == 0
    assert len(list((tmp_path / "vectors").glob("*.npy"))) == 2
    assert np.array_equal(e(docs), first)


@pytest.mark.slow
def test_long_inputs_are_cached_in_slices_so_an_interrupted_run_keeps_progress(tmp_path):
    from vault_rag.embed import FastEmbedder

    e = FastEmbedder(vectors=tmp_path / 'vectors', models=tmp_path / 'models')
    texts = [f"note number {i} about kelp" for i in range(300)]
    full = e(texts)
    assert full.shape == (300, 384)
    assert len(list((tmp_path / "vectors").glob("*.npy"))) == 2
    assert np.array_equal(e(texts[:256]), full[:256])
    assert len(list((tmp_path / "vectors").glob("*.npy"))) == 2


def test_offline_embedder_serves_cached_vectors_without_loading_the_model(tmp_path):
    from vault_rag.embed import FastEmbedder
    from vault_rag.llm import CacheMiss

    e = FastEmbedder(vectors=tmp_path, models=tmp_path / "models", offline=True)
    texts = ["alpha", "beta"]
    saved = np.eye(2, 384, dtype=np.float32)
    e.vector_path(texts).parent.mkdir(parents=True, exist_ok=True)
    np.save(e.vector_path(texts), saved)
    assert np.array_equal(e(texts), saved)
    with pytest.raises(CacheMiss):
        e(["never embedded"])
    assert e._model is None
