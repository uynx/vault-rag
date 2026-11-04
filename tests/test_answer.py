from fakes import fake_embed
from vault_rag.answer import answer, build_prompt, cited_indices
from vault_rag.corpus import Chunk
from vault_rag.retrieval import Retriever

CHUNKS = [
    Chunk("boats", 0, "Boats\nThe Gannet hauls two tonnes per trip"),
    Chunk("buyers", 0, "Buyers\nCorvane takes sixty percent"),
    Chunk("nursery", 0, "Nursery\nSix tanks"),
]


def test_prompt_numbers_passages_and_names_notes():
    p = build_prompt("How much does the Gannet haul?", CHUNKS[:2])
    assert "[1] (boats)" in p and "[2] (buyers)" in p
    assert "How much does the Gannet haul?" in p and "Gannet hauls two tonnes" in p


def test_cited_indices_ignores_out_of_range_and_duplicates():
    assert cited_indices("It hauls 2 t [1][3]. Also [1] and [7] and [0].", 3) == [1, 3]


def test_answer_returns_text_and_cited_notes():
    r = Retriever(CHUNKS, fake_embed)
    seen = []

    def llm(prompt):
        seen.append(prompt)
        return "Two tonnes [1]."

    out = answer("How many tonnes does the Gannet haul", r, llm, method="bm25", k=2)
    assert out["answer"] == "Two tonnes [1]."
    assert out["cited_notes"] == ["boats"]
    assert out["retrieved_notes"][0] == "boats"
    assert len(seen) == 1
