"""Measure the two terms of the selection score as a function of pool size.

The scale argument is that the coverage gain is extensive in the pool, since it
sums a contribution over all N members, while relevance is intensive, a single
bounded cosine. If that holds, the ratio between the two terms at the first
selection step should grow roughly linearly in N, so a fixed weight becomes
progressively more wrong as the pool grows. This measures both terms directly on
the multi-gold benchmark at several pool sizes, so the prediction can be checked
against the same data the main comparison uses.

Runs on cached embeddings. No model, no API key.
"""
from __future__ import annotations

import json
from pathlib import Path
from sys import path as _syspath

import numpy as np

_syspath.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E

POOL_SIZES = (15, 30, 50, 100, 200)
K = 5
ALPHA = 0.5


def trace_one(sim: np.ndarray, rel: np.ndarray, k: int, alpha: float) -> list[dict]:
    """
    Greedy selection, recording both score terms for the candidate chosen.

    The marginal coverage gain is computed for every candidate at once: since
    max(m_j, s_ij) - m_j equals relu(s_ij - m_j), the whole gain vector is one
    rectified subtraction summed along the pool axis.
    """
    n = len(rel)
    cur = np.zeros(n, dtype=np.float64)
    taken = np.zeros(n, dtype=bool)
    steps: list[dict] = []
    for _ in range(k):
        gains = np.maximum(sim - cur[None, :], 0.0).sum(axis=1)
        scores = (1 - alpha) * gains + alpha * rel
        scores[taken] = -np.inf
        best = int(np.argmax(scores))
        taken[best] = True
        cur = np.maximum(cur, sim[best])
        r = float(rel[best])
        g = float(gains[best])
        steps.append({"coverage_gain": g, "relevance": r,
                      "ratio": g / r if r else float("inf")})
    return steps


def main() -> None:
    passages, questions, pe, qe = E.load()
    out: list[dict] = []

    print(f"Two score terms of the selection rule, alpha = {ALPHA}, k = {K}", flush=True)
    print(f"\n{'N':>5} {'step':>5} {'coverage gain':>22} {'relevance':>18} "
          f"{'ratio (med)':>12} {'rel>cov':>8}", flush=True)

    for N in POOL_SIZES:
        E.POOL_N = N
        ctx = E.build_context(passages, questions, pe, qe)
        traces = [trace_one(c["sim"].astype(np.float64), c["rel"].astype(np.float64),
                            K, ALPHA) for c in ctx]
        for st in range(K):
            cov = np.array([t[st]["coverage_gain"] for t in traces])
            rel = np.array([t[st]["relevance"] for t in traces])
            ratio = np.array([t[st]["ratio"] for t in traces])
            row = {
                "pool_n": N, "step": st + 1,
                "coverage_p05": float(np.percentile(cov, 5)),
                "coverage_p95": float(np.percentile(cov, 95)),
                "coverage_median": float(np.median(cov)),
                "relevance_p05": float(np.percentile(rel, 5)),
                "relevance_p95": float(np.percentile(rel, 95)),
                "relevance_median": float(np.median(rel)),
                "ratio_median": float(np.median(ratio)),
                "frac_rel_exceeds_cov": float(np.mean(rel > cov)),
                "n_questions": len(ctx),
            }
            out.append(row)
            print(f"{N:>5} {st+1:>5} "
                  f"{row['coverage_p05']:>9.2f} to {row['coverage_p95']:<9.2f} "
                  f"{row['relevance_p05']:>7.2f} to {row['relevance_p95']:<7.2f} "
                  f"{row['ratio_median']:>12.1f} {row['frac_rel_exceeds_cov']:>8.2f}",
                  flush=True)
        print(flush=True)

    first = [r for r in out if r["step"] == 1]
    print("Prediction: the first-step ratio should scale with N.", flush=True)
    print(f"{'N':>6} {'median ratio':>14} {'ratio / N':>11}", flush=True)
    for r in first:
        print(f"{r['pool_n']:>6} {r['ratio_median']:>14.1f} "
              f"{r['ratio_median'] / r['pool_n']:>11.3f}", flush=True)
    ns = np.array([r["pool_n"] for r in first], dtype=float)
    rs = np.array([r["ratio_median"] for r in first])
    print(f"\nPearson r(N, ratio) = {float(np.corrcoef(ns, rs)[0, 1]):.4f}", flush=True)

    E.RESULTS.mkdir(exist_ok=True)
    (E.RESULTS / "multigold_instrumentation.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {E.RESULTS / 'multigold_instrumentation.json'}", flush=True)


if __name__ == "__main__":
    main()
