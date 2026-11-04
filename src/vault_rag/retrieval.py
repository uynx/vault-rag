import re

import numpy as np
from scipy.sparse import csc_matrix

TOKEN = re.compile(r"\w+")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        vocab: dict[str, int] = {}
        rows, cols, tf = [], [], []
        for i, doc in enumerate(docs):
            counts: dict[int, int] = {}
            for t in tokens(doc):
                j = vocab.setdefault(t, len(vocab))
                counts[j] = counts.get(j, 0) + 1
            rows += [i] * len(counts)
            cols += counts.keys()
            tf += counts.values()
        rows, cols, tf = np.array(rows), np.array(cols), np.array(tf, dtype=np.float64)
        dl = np.bincount(rows, weights=tf, minlength=len(docs))
        df = np.bincount(cols, minlength=len(vocab))
        idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * dl[rows] / dl.mean())
        weight = idf[cols] * tf * (k1 + 1) / (tf + norm)
        self.vocab = vocab
        self.w = csc_matrix((weight, (rows, cols)), shape=(len(docs), len(vocab)))

    def scores(self, query: str) -> np.ndarray:
        cols = sorted({self.vocab[t] for t in tokens(query) if t in self.vocab})
        return np.asarray(self.w[:, cols].sum(axis=1)).ravel()


class Dense:
    def __init__(self, docs: list[str], embed):
        self.embed = embed
        self.vecs = embed(docs)

    def scores(self, query: str) -> np.ndarray:
        return self.vecs @ self.embed([query], query=True)[0]


def rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    n = max(len(r) for r in rankings)
    fused = np.zeros(n)
    for r in rankings:
        fused[r] += 1.0 / (k + 1 + np.arange(len(r)))
    return np.argsort(-fused, kind="stable")
