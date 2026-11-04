import argparse
import json
import sys
from pathlib import Path

from .corpus import load_corpus
from .evalrun import answer_metrics, load_questions, markdown_table, retrieval_metrics
from .judgecheck import run_check
from .labelset import build_labelset
from .llm import cached, claude_llm
from .questiongen import generate_questions
from .metrics import regressions
from .retrieval import METHODS, Retriever

MODEL = "sonnet"


def _retriever(corpus, embed, exclude=""):
    if embed is None:
        from .embed import FastEmbedder

        embed = FastEmbedder()
    return Retriever(load_corpus(corpus, exclude=tuple(x for x in exclude.split(',') if x)), embed)


def cmd_eval(a, embed, llm) -> int:
    r = _retriever(a.corpus, embed, a.exclude)
    qs = load_questions(a.questions)
    ret = retrieval_metrics(r, qs)
    current = {f"retrieval/{m}": v for m, v in ret["overall"].items()}
    md = [f"Corpus: {len({c.note for c in r.chunks})} notes, {len(r.chunks)} chunks, {len(qs)} questions.",
          "\n### Retrieval (note-level, all questions)", markdown_table(ret["overall"])]
    for m in METHODS:
        md += [f"\n### Retrieval by question type, {m}", markdown_table(ret["by_type"][m])]
    answers = {}
    if a.answers:
        call = _llm(a, llm)
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
    return _gate(current, a)


def _gate(current: dict, a) -> int:
    base_path = Path(a.baseline)
    if a.update_baseline:
        old = json.loads(base_path.read_text()) if base_path.exists() else {}
        base_path.parent.mkdir(parents=True, exist_ok=True)
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


def _llm(a, llm):
    return cached(llm or claude_llm, Path(a.cache) / "llm.jsonl", MODEL, offline=a.offline)


def _read_jsonl(path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def cmd_genquestions(a, embed, llm) -> int:
    chunks = load_corpus(a.corpus, exclude=tuple(x for x in a.exclude.split(",") if x))
    qs = generate_questions(chunks, _llm(a, llm), a.n, a.seed, a.workers)
    Path(a.questions).parent.mkdir(parents=True, exist_ok=True)
    Path(a.questions).write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
    print(f"wrote {len(qs)} model-written questions to {a.questions}")
    return 0


def cmd_labelset(a, embed, llm) -> int:
    rows = build_labelset(_retriever(a.corpus, embed, a.exclude), load_questions(a.questions), _llm(a, llm),
                          a.n_pairs, a.n_weak, a.seed, a.workers)
    Path(a.labelset).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} answers to {a.labelset}; label them in {a.labels} before running judge-check")
    return 0


def _position_rows(pos: dict, padded: dict) -> list[str]:
    row = lambda name, p: (f"| {name} | {p['n']} | {p['pick_first_rate']:.3f} | {p['acc_good_first']:.3f} | "
                           f"{p['acc_good_second']:.3f} | {p['flip_rate']:.3f} |")
    return ["| | pairs | picks first | right when good is first | right when good is second | verdict flips on swap |",
            "|---|---:|---:|---:|---:|---:|", row("plain", pos), row("bad answer padded", padded)]


def judge_markdown(o: dict) -> str:
    v, pos, padded, ln = o["validation"], o["position"], o["position_padded_bad"], o["length"]
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    within = ", ".join(f"label {k}: {f(x)}" for k, x in ln["spearman_within_label"].items())
    return "\n".join([
        f"### Judge vs reference labels (n={v['n']})",
        f"- exact agreement {v['agreement']:.3f}, Cohen's kappa (3 classes) {v['kappa_3class']:.3f}",
        f"- binary (fully correct or not): agreement {v['binary_agreement']:.3f}, kappa {v['kappa_binary']:.3f}",
        f"- 95% Wilson interval on exact agreement: {v['agreement_ci95'][0]:.3f} to {v['agreement_ci95'][1]:.3f}",
        f"- confusion, label to judge score: {json.dumps(v['confusion'], sort_keys=True)}",
        *[f"- {k} rows only (n={x['n']}): agreement {x['agreement']:.3f}, kappa 3-class {x['kappa_3class']:.3f}, "
          f"binary kappa {x['kappa_binary']:.3f}" for k, x in o["validation_by_origin"].items()],
        f"\n### Position bias ({o['n_pairs']} good/bad pairs, each judged in both orders)",
        *(_position_rows(pos, padded) if pos["n"] else ["no pair has both a fully correct and a wrong answer"]),
        f"\n### Length bias (n={ln['n']}, each answer judged again with about 45 words of filler appended)",
        f"- mean score change {ln['mean_delta']:+.3f}, raised {ln['pct_raised']:.3f}, lowered {ln['pct_lowered']:.3f}, "
        f"wrong or partial answers raised {f(ln['wrong_promoted'])}",
        f"- Spearman(answer length, judge score) {f(ln['spearman_len_score'])}; within a label: {within}"])


def cmd_judge_check(a, embed, llm) -> int:
    rows = _read_jsonl(a.labelset)
    labels = {r["id"]: r["label"] for r in _read_jsonl(a.labels)} if Path(a.labels).exists() else {}
    missing = [r["id"] for r in rows if r["id"] not in labels]
    if missing:
        print(f"unlabelled answers: {missing}", file=sys.stderr)
        return 2
    out = run_check(rows, labels, _llm(a, llm), a.workers)
    table = judge_markdown(out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "judge.md").write_text(table + "\n")
    (Path(a.out) / "judge.json").write_text(json.dumps(out, indent=2) + "\n")
    print(table)
    v = out["validation"]
    return _gate({"judge/validation": {"agreement": v["agreement"], "kappa_binary": v["kappa_binary"]}}, a)


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", default="demo/corpus")
    common.add_argument("--questions", default="demo/questions.jsonl")
    common.add_argument("--exclude", default="", help="comma-separated directory names to skip")
    common.add_argument("--baseline", default="results/baseline.json")
    common.add_argument("--out", default="results")
    common.add_argument("--cache", default="demo/cache")
    common.add_argument("--offline", action="store_true", help="fail on an LLM cache miss instead of calling claude")
    common.add_argument("--update-baseline", action="store_true")
    common.add_argument("--tol", type=float, default=0.005)
    common.add_argument("--workers", type=int, default=8)
    common.add_argument("--labelset", default="demo/labelset.jsonl")
    common.add_argument("--labels", default="demo/labels.jsonl")
    p = argparse.ArgumentParser(prog="vault-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", parents=[common], help="retrieval and answer metrics, results table, baseline gate")
    e.add_argument("--answers", action="store_true", help="also generate and judge answers")
    e.add_argument("-k", type=int, default=5)
    g = sub.add_parser("genquestions", parents=[common], help="have claude write one question per sampled chunk")
    g.add_argument("--n", type=int, default=60)
    g.add_argument("--seed", type=int, default=7)
    ls = sub.add_parser("labelset", parents=[common], help="build the pool of answers to label by hand")
    ls.add_argument("--n-pairs", type=int, default=16)
    ls.add_argument("--n-weak", type=int, default=8)
    ls.add_argument("--seed", type=int, default=7)
    sub.add_parser("judge-check", parents=[common], help="judge agreement with labels, position and length bias")
    return p


def main(argv=None, embed=None, llm=None) -> int:
    a = parser().parse_args(argv)
    return {"eval": cmd_eval, "genquestions": cmd_genquestions, "labelset": cmd_labelset, "judge-check": cmd_judge_check}[a.cmd](a, embed, llm)


if __name__ == "__main__":
    sys.exit(main())
