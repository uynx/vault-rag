import json

import pytest
from fakes import fake_embed

from vault_rag.cli import main
from vault_rag.corpus import load_corpus
from vault_rag.evalrun import answer_metrics, load_questions, markdown_table, retrieval_metrics
from vault_rag.retrieval import Retriever

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


@pytest.fixture
def vault(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in NOTES.items():
        (corpus / f"{name}.md").write_text(text)
    qpath = tmp_path / "q.jsonl"
    qpath.write_text("\n".join(json.dumps(q) for q in QUESTIONS))
    return tmp_path


def test_load_questions_reads_jsonl(vault):
    assert [q["id"] for q in load_questions(vault / "q.jsonl")] == ["q1", "q2", "q3"]


def test_retrieval_metrics_match_hand_computed_values(vault):
    r = Retriever(load_corpus(vault / "corpus"), fake_embed)
    out = retrieval_metrics(r, QUESTIONS, methods=("bm25",), ks=(1, 3))
    assert out["overall"]["bm25"] == pytest.approx({"recall@1": 2 / 3, "recall@3": 1.0, "mrr": 7 / 9})
    assert out["by_type"]["bm25"]["paraphrase"]["recall@1"] == 0.0
    assert out["by_type"]["bm25"]["lookup"]["mrr"] == 1.0


def test_answer_metrics_aggregate_judge_scores_and_citation_hits(vault):
    r = Retriever(load_corpus(vault / "corpus"), fake_embed)
    scores = iter([2, 1, 0])

    def gen(prompt):
        return "It is [1]."

    def judge(prompt):
        return '{"reason": "", "score": %d}' % next(scores)

    m, rows = answer_metrics(r, QUESTIONS, gen, judge, method="bm25", k=1, workers=1)
    assert m == pytest.approx({"correct": 1 / 3, "mean_score": 0.5, "citation_hit": 2 / 3})
    assert [x["score"] for x in rows] == [2, 1, 0] and rows[0]["id"] == "q1"


def test_markdown_table_lists_every_method_row():
    t = markdown_table({"bm25": {"recall@1": 0.5, "mrr": 0.75}, "dense": {"recall@1": 1.0, "mrr": 1.0}})
    assert "| bm25 | 0.500 | 0.750 |" in t and "| dense | 1.000 | 1.000 |" in t


def run_cli(vault, *extra):
    argv = ["eval", "--corpus", str(vault / "corpus"), "--questions", str(vault / "q.jsonl"),
            "--baseline", str(vault / "baseline.json"), "--out", str(vault / "out"), *extra]
    return main(argv, embed=fake_embed)


def test_gate_exit_codes(vault, capsys):
    assert run_cli(vault) == 2
    assert run_cli(vault, "--update-baseline") == 0
    assert run_cli(vault) == 0
    base = json.loads((vault / "baseline.json").read_text())
    base["retrieval/bm25"]["mrr"] = 0.99
    (vault / "baseline.json").write_text(json.dumps(base))
    assert run_cli(vault) == 1
    assert "retrieval/bm25/mrr" in capsys.readouterr().out
    assert (vault / "out" / "results.md").exists() and (vault / "out" / "results.json").exists()
