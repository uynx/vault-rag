from pathlib import Path

from vault_rag.evalrun import load_questions

ROOT = Path(__file__).parent.parent / "demo"


def test_question_set_is_consistent_with_the_corpus():
    notes = {p.stem for p in (ROOT / "corpus").glob("*.md")}
    qs = load_questions(ROOT / "questions.jsonl")
    assert len({q["id"] for q in qs}) == len(qs) >= 70
    assert {q["type"] for q in qs} == {"lookup", "paraphrase", "multi"}
    for q in qs:
        assert q["gold_sources"] and set(q["gold_sources"]) <= notes, q["id"]
        assert q["question"].strip() and q["gold_answer"].strip()
        assert len(q["gold_sources"]) == 1 or q["type"] == "multi", q["id"]
