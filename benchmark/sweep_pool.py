"""Does the verdict depend on the pool size?

At N=30 the pool ceiling on this benchmark is recall 0.51, so most gold passages are
not reachable by any selection rule and the comparison is retrieval-bound rather than
selection-bound. The single-repo benchmark hid this: with one gold per question the
ceiling was 1.00 at N=15, so pool size never bound there.

This sweeps the pool size so the methods are compared in the regime where selection
actually decides the outcome -- a high ceiling, with far more reachable gold than the
budget can hold. Weights are tuned on the training repositories and evaluated once on
the held-out ones, independently at each N.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sys import path as _syspath
_syspath.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E

POOL_SIZES = (15, 30, 50, 100, 200)
BUDGETS = (5, 10)


def main() -> None:
    passages, questions, pe, qe = E.load()
    rows = []

    for N in POOL_SIZES:
        E.POOL_N = N
        ctx = E.build_context(passages, questions, pe, qe)
        train = [c for c in ctx if c["repo"] not in E.TEST_REPOS]
        test = [c for c in ctx if c["repo"] in E.TEST_REPOS]
        ceil = E.pool_ceiling(test)

        for k in BUDGETS:
            if k >= N:
                continue
            base = E.per_question(test, E.select_topk, 0.0, k)
            # tune alpha and lambda on train, then evaluate once on test
            a_best = max(E.ALPHAS, key=lambda a: E.score(train, E.select_fl, a, k)["recall"])
            l_best = max(E.LAMBDAS, key=lambda l: E.score(train, E.select_mmr, l, k)["recall"])

            entry = {
                "pool_n": N, "k": k,
                "ceiling_recall": ceil["recall"], "ceiling_compl": ceil["recall_compl"],
                "top_k": E.score(test, E.select_topk, 0.0, k),
                "random": E.score_random(test, k),
                "alpha_tuned": a_best, "lambda_tuned": l_best,
            }
            for name, f, w in (("fl_alpha0.5", E.select_fl, 0.5),
                               ("fl_tuned", E.select_fl, a_best),
                               ("mmr_lambda0.5", E.select_mmr, 0.5),
                               ("mmr_tuned", E.select_mmr, l_best)):
                entry[name] = E.score(test, f, w, k)
                cand = E.per_question(test, f, w, k)
                entry[name + "_vs_topk"] = {
                    m: E.wilcoxon(cand[m], base[m]) for m in ("recall", "recall_compl")
                }
            rows.append(entry)
            print(f"N={N:>3} k={k:>2}  ceiling r={ceil['recall']:.3f} c={ceil['recall_compl']:.3f}"
                  f"  | tuned alpha={a_best} lambda={l_best}")

    E.RESULTS.mkdir(exist_ok=True)
    (E.RESULTS / "multigold_pool_sweep.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'='*104}")
    print("HELD-OUT REPOS (n=84). d = method minus top-k; negative means worse than top-k.")
    print(f"{'='*104}")
    print(f"{'N':>4} {'k':>3} {'ceil_r':>7} {'topk_r':>7} {'rand_r':>7} | "
          f"{'a*':>4} {'d_rec':>8} {'p':>7} {'d_cmpl':>8} {'p':>7} | "
          f"{'L*':>4} {'d_rec':>8} {'p':>7}")
    for e in rows:
        fa = e["fl_tuned_vs_topk"]; mm = e["mmr_tuned_vs_topk"]
        print(f"{e['pool_n']:>4} {e['k']:>3} {e['ceiling_recall']:>7.3f} "
              f"{e['top_k']['recall']:>7.3f} {e['random']['recall']:>7.3f} | "
              f"{e['alpha_tuned']:>4} {fa['recall']['mean_diff']:>+8.4f} "
              f"{fa['recall']['p']:>7.4f} {fa['recall_compl']['mean_diff']:>+8.4f} "
              f"{fa['recall_compl']['p']:>7.4f} | "
              f"{e['lambda_tuned']:>4} {mm['recall']['mean_diff']:>+8.4f} "
              f"{mm['recall']['p']:>7.4f}")

    print(f"\nAt the paper's default weight (alpha = lambda = 0.5):")
    print(f"{'N':>4} {'k':>3} {'d_rec(FL)':>10} {'p':>7} {'d_cmpl(FL)':>11} {'p':>7} "
          f"{'d_rec(MMR)':>11} {'p':>7}")
    for e in rows:
        f5 = e["fl_alpha0.5_vs_topk"]; m5 = e["mmr_lambda0.5_vs_topk"]
        print(f"{e['pool_n']:>4} {e['k']:>3} {f5['recall']['mean_diff']:>+10.4f} "
              f"{f5['recall']['p']:>7.4f} {f5['recall_compl']['mean_diff']:>+11.4f} "
              f"{f5['recall_compl']['p']:>7.4f} {m5['recall']['mean_diff']:>+11.4f} "
              f"{m5['recall']['p']:>7.4f}")


if __name__ == "__main__":
    main()
