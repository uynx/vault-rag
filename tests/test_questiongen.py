import json

from fakes import fake_embed
from vault_rag.corpus import Chunk, load_corpus
from vault_rag.questiongen import generate_questions


def test_load_corpus_skips_excluded_directories(tmp_path):
    for d in ("topics", "stale", "raw"):
        (tmp_path / d).mkdir()
        (tmp_path / d / f"{d}_note.md").write_text(f"# {d}\n## S\nsome text here\n")
    chunks = load_corpus(tmp_path, exclude=("stale", "raw"))
    assert {c.note for c in chunks} == {"topics_note"}


CHUNKS = [Chunk(f"note{i}", 0, f"Title{i}\n" + " ".join(f"word{j}" for j in range(80))) for i in range(6)] + [
    Chunk("secrets", 0, "Keys\nthe api_key is stored here " + " ".join(["filler"] * 80)),
    Chunk("tiny", 0, "Tiny\nshort"),
]


def llm(prompt):
    return json.dumps({"question": "What is stored?", "answer": "Words."})


def test_generate_questions_samples_long_chunks_and_skips_secrets():
    qs = generate_questions(CHUNKS, llm, n=4, seed=1, workers=1)
    assert len(qs) == 4
    notes = {q["gold_sources"][0] for q in qs}
    assert notes <= {f"note{i}" for i in range(6)}
    assert [q["id"] for q in qs] == ["g01", "g02", "g03", "g04"]
    assert all(q["type"] == "synthetic" and q["gold_answer"] == "Words." for q in qs)
    assert qs == generate_questions(CHUNKS, llm, n=4, seed=1, workers=1)


def test_malformed_model_output_is_dropped():
    out = generate_questions(CHUNKS, lambda p: "not json", n=3, seed=1, workers=1)
    assert out == []
