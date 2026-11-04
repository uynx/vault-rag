import argparse
import json
import sys
from pathlib import Path

from .corpus import load_corpus
from .evalrun import answer_metrics, load_questions, markdown_table, retrieval_metrics
from .llm import cached, claude_llm
from .metrics import regressions
from .retrieval import METHODS, Retriever

MODEL = "sonnet"


def _retriever(corpus, embed):
    if embed is None:
        from .embed import FastEmbedder

        embed = FastEmbedder()
    return Retriever(load_corpus(corpus), embed)


def cmd_eval(a, embed, llm) -> int:
    r = _retriever(a.corpus, embed)
    qs = load_questions(a.questions)
    ret = retrieval_metrics(r, qs)
    current = {f"retrieval/{m}": v for m, v in ret["overall"].items()}
    md = [f"Corpus: {len({c.note for c in r.chunks})} notes, {len(r.chunks)} chunks, {len(qs)} questions.",
          "\n### Retrieval (note-level, all questions)", markdown_table(ret["overall"])]
    for m in METHODS:
        md += [f"\n### Retrieval by question type, {m}", markdown_table(ret["by_type"][m])]
    answers = {}
    if a.answers:
        call = cached(llm or claude_llm, Path(a.cache) / "llm.jsonl", MODEL, offline=a.offline)
        for m in METHODS:
            current[f"answers/{m}"], answers[m] = answer_metrics(r, qs, call, call, m, a.k, a.workers)
        md += [f"\n### Answers (top-{a.k} chunks, judged by `claude -p --model {MODEL}`)",
               markdown_table({m: current[f"answers/{m}"] for m in METHODS})]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = "\n".join(md)
    (out / "results.md").write_text(table + "\n")
    (out / "results.json").write_text(json.dumps({"metrics": current, "by_type": ret["by_type"]}, indent=2) + "\n")
    for m, rows in answers.items():
        (out / f"answers_{m}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(table)
    base_path = Path(a.baseline)
    if a.update_baseline:
        old = json.loads(base_path.read_text()) if base_path.exists() else {}
        base_path.write_text(json.dumps({**old, **current}, indent=2) + "\n")
        print(f"\nbaseline written to {base_path}")
        return 0
    if not base_path.exists():
        print(f"\nno baseline at {base_path}; run with --update-baseline", file=sys.stderr)
        return 2
    base = {g: v for g, v in json.loads(base_path.read_text()).items() if g in current}
    bad = regressions(current, base, a.tol)
    print("\nREGRESSION\n" + "\n".join(bad) if bad else f"\nOK: no metric below baseline (tolerance {a.tol})")
    return 1 if bad else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vault-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="retrieval and answer metrics, results table, baseline gate")
    e.add_argument("--corpus", default="demo/corpus")
    e.add_argument("--questions", default="demo/questions.jsonl")
    e.add_argument("--baseline", default="results/baseline.json")
    e.add_argument("--out", default="results")
    e.add_argument("--cache", default="demo/cache")
    e.add_argument("--answers", action="store_true", help="also generate and judge answers")
    e.add_argument("--offline", action="store_true", help="fail on an LLM cache miss instead of calling claude")
    e.add_argument("--update-baseline", action="store_true")
    e.add_argument("--tol", type=float, default=0.005)
    e.add_argument("-k", type=int, default=5)
    e.add_argument("--workers", type=int, default=8)
    return p


def main(argv=None, embed=None, llm=None) -> int:
    a = parser().parse_args(argv)
    return {"eval": cmd_eval}[a.cmd](a, embed, llm)


if __name__ == "__main__":
    sys.exit(main())
