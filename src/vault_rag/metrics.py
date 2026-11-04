from collections import Counter


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    return next((1 / (i + 1) for i, n in enumerate(ranked) if n in gold), 0.0)


def mean_metrics(runs, ks=(1, 3, 5)) -> dict[str, float]:
    out = {f"recall@{k}": sum(recall_at_k(r, g, k) for r, g in runs) / len(runs) for k in ks}
    out["mrr"] = sum(reciprocal_rank(r, g) for r, g in runs) / len(runs)
    return out


def cohen_kappa(a, b) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in ca.keys() | cb.keys()) / n**2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def regressions(current: dict, baseline: dict, tol: float) -> list[str]:
    out = []
    for group, metrics in baseline.items():
        for name, base in metrics.items():
            cur = current.get(group, {}).get(name)
            if cur is None:
                out.append(f"{group}/{name}: missing, baseline {base:.4f}")
            elif cur < base - tol:
                out.append(f"{group}/{name}: {cur:.4f} < baseline {base:.4f}")
    return out
