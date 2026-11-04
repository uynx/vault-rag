import hashlib
from pathlib import Path

import numpy as np

from .llm import CacheMiss
from .retrieval import QUERY_PREFIX

MODEL = "BAAI/bge-small-en-v1.5"
HOME = Path.home() / ".cache" / "vault-rag"
SLICE = 256


class FastEmbedder:
    def __init__(self, vectors: Path = HOME / "vectors", models: Path = HOME / "models", model: str = MODEL,
                 offline: bool = False):
        self.model_name, self.vectors, self.models, self.offline = model, Path(vectors), Path(models), offline
        self._model = None

    def vector_path(self, texts: list[str]) -> Path:
        key = hashlib.sha256("\0".join([self.model_name, *texts]).encode()).hexdigest()
        return self.vectors / f"{key}.npy"

    def __call__(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        return np.vstack([self._slice(texts[i : i + SLICE]) for i in range(0, len(texts), SLICE)])

    def _slice(self, texts: list[str]) -> np.ndarray:
        path = self.vector_path(texts)
        if path.exists():
            return np.load(path)
        if self.offline:
            raise CacheMiss(f"no cached vectors for {len(texts)} texts starting {texts[0][:60]!r}")
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, cache_dir=str(self.models))
        vecs = np.asarray(list(self._model.embed(texts, batch_size=64)), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs
