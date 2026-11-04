import zlib

import numpy as np

from vault_rag.retrieval import tokens


def fake_embed(texts, query=False):
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in tokens(t):
            out[i, zlib.crc32(w.encode()) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


NOTES = {
    "boats": "# Boats\n## Fleet\nThe Gannet hauls two tonnes per trip.\n",
    "buyers": "# Buyers\n## Split\nCorvane takes sixty percent of each harvest.\n",
    "nursery": "# Nursery\n## Setup\nSix tanks grow gametophytes under blue light.\n",
}
QUESTIONS = [
    {"id": "q1", "type": "lookup", "question": "how many tonnes does the Gannet haul", "gold_sources": ["boats"],
     "gold_answer": "Two tonnes."},
    {"id": "q2", "type": "lookup", "question": "which buyer takes sixty percent", "gold_sources": ["buyers"],
     "gold_answer": "Corvane."},
    {"id": "q3", "type": "paraphrase", "question": "xyzzy", "gold_sources": ["nursery"], "gold_answer": "Six tanks."},
]
