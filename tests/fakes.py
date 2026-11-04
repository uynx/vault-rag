import zlib

import numpy as np

from vault_rag.retrieval import tokens


def fake_embed(texts, query=False):
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokens(t):
            out[i, zlib.crc32(w.encode()) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
