import hashlib
from pathlib import Path

import numpy as np

from .retrieval import QUERY_PREFIX

MODEL = "BAAI/bge-small-en-v1.5"
CACHE = Path.home() / ".cache" / "vault-rag"
SLICE = 256


class FastEmbedder:
    def __init__(self, model: str = MODEL, cache: Path = CACHE):
        from fastembed import TextEmbedding

        self.model_name, self.cache = model, Path(cache)
        self.model = TextEmbedding(model, cache_dir=str(self.cache / "models"))

    def __call__(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        return np.vstack([self._slice(texts[i : i + SLICE]) for i in range(0, len(texts), SLICE)])

    def _slice(self, texts: list[str]) -> np.ndarray:
        key = hashlib.sha256("\0".join([self.model_name, *texts]).encode()).hexdigest()
        path = self.cache / "vectors" / f"{key}.npy"
        if path.exists():
            return np.load(path)
        vecs = np.asarray(list(self.model.embed(texts, batch_size=64)), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs
