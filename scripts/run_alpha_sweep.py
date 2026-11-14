"""Sweep the relevance weight alpha, and the MMR weight lambda, over the benchmark.

Both selection rules are reported in the paper at a single hand-set value (alpha = 0.5,
lambda = 0.5), which leaves open whether the result is a property of the method or of
that one setting. Retrieval and selection are deterministic and need no generator, so
the sweep costs nothing but CPU.

Each question's pool is embedded once and reused across every alpha and lambda, so the
whole sweep costs about the same as a single comparison run.

Also reports two reference points the main comparison lacks:
  - pool recall, the ceiling for any method that selects from the pool
  - the expected recall of selecting k of N uniformly at random, which is the level a
    selection rule has to beat to be doing anything at all
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path
from sys import path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in path:
    path.insert(0, str(PROJECT_ROOT))

from retriever.retrieve import retrieve_candidates
from selection.facility_location import relevance_weighted_facility_location_select

EMBEDDING_MODEL = "all-mpnet-base-v2"
TOP_K = 5
NUM_CANDIDATES = 15
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
LAMBDAS = [round(0.1 * i, 1) for i in range(11)]

INPUT_PATH = PROJECT_ROOT / "data" / "code_questions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "alpha_sweep.json"


def mmr_indices(sim: np.ndarray, rel: np.ndarray, k: int, lam: float) -> list[int]:
    """MMR over a precomputed pool, identical in rule to selection.mmr.mmr_select."""
    selected: list[int] = []
    candidates = list(range(len(rel)))
    while len(selected) < k and candidates:
        if not selected:
            best = max(candidates, key=lambda i: rel[i])
        else:
            best = max(
                candidates,
                key=lambda i: lam * rel[i] - (1 - lam) * max(sim[i][j] for j in selected),
            )
        selected.append(best)
        candidates.remove(best)
    return selected


def binom_tail_ge(x: int, n: int, p: float) -> float:
    """P(X >= x) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x, n + 1))


def main() -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)
    with INPUT_PATH.open(encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Embed each question's pool exactly once.
    pools = []
    for record in data:
        question: str = record["question"]
        gold = set(record["gold_passage_ids"])
        candidates = retrieve_candidates(question, n=NUM_CANDIDATES)
        pool_ids = [p["passage_id"] for p in candidates]

        emb = np.array(model.encode([p["text"] for p in candidates]))
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        q_emb = np.array(model.encode([question])[0])
        q_emb = q_emb / np.linalg.norm(q_emb)

        pools.append({
            "qid": record["qid"],
            "pool_ids": pool_ids,
            "gold": gold,
            "sim": emb @ emb.T,
            "rel": emb @ q_emb,
            "gold_in_pool": bool(gold & set(pool_ids)),
        })

    n = len(pools)

    def score(select) -> tuple[float, float, int]:
        """Macro-averaged precision@k, recall@k, and hit count for a selection rule."""
        precisions, recalls, hits = [], [], 0
        for p in pools:
            idx = select(p)
            ids = {p["pool_ids"][i] for i in idx}
            overlap = ids & p["gold"]
            precisions.append(len(overlap) / len(ids) if ids else 0.0)
            recalls.append(len(overlap) / len(p["gold"]) if p["gold"] else 0.0)
            hits += bool(overlap)
        return float(np.mean(precisions)), float(np.mean(recalls)), hits

    alpha_rows = []
    for a in ALPHAS:
        pr, rc, h = score(lambda p, a=a: relevance_weighted_facility_location_select(
            p["sim"], p["rel"], k=TOP_K, alpha=a))
        alpha_rows.append({"alpha": a, "precision@k": pr, "recall@k": rc, "hits": h})

    lambda_rows = []
    for lam in LAMBDAS:
        pr, rc, h = score(lambda p, lam=lam: mmr_indices(
            p["sim"], p["rel"], TOP_K, lam))
        lambda_rows.append({"lambda": lam, "precision@k": pr, "recall@k": rc, "hits": h})

    pool_hits = sum(p["gold_in_pool"] for p in pools)
    p_random = TOP_K / NUM_CANDIDATES
    expected_random_hits = pool_hits * p_random

    best_alpha = max(alpha_rows, key=lambda r: (r["hits"], r["precision@k"]))
    best_lambda = max(lambda_rows, key=lambda r: (r["hits"], r["precision@k"]))
    cov_only = next(r for r in alpha_rows if r["alpha"] == 0.0)

    output = {
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "num_candidates": NUM_CANDIDATES,
            "k": TOP_K,
            "questions": n,
        },
        "pool_recall": pool_hits / n,
        "pool_recall_hits": pool_hits,
        "random_baseline": {
            "p_select_gold_given_in_pool": p_random,
            "expected_recall": expected_random_hits / n,
            "expected_hits": expected_random_hits,
            "p_coverage_only_ge_observed":
                binom_tail_ge(cov_only["hits"], pool_hits, p_random),
        },
        "alpha_sweep": alpha_rows,
        "lambda_sweep": lambda_rows,
        "best_alpha": best_alpha,
        "best_lambda": best_lambda,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"{n} questions, pool N = {NUM_CANDIDATES}, budget k = {TOP_K}")
    print(f"gold in pool: {pool_hits}/{n} = {pool_hits / n:.4f}  (ceiling for pool methods)")
    r = output["random_baseline"]
    print(f"random {TOP_K}-of-{NUM_CANDIDATES}: expected recall {r['expected_recall']:.4f} "
          f"({r['expected_hits']:.1f} hits)\n")

    print("Facility location, sweep over alpha")
    print(f"{'alpha':>6}  {'P@5':>8}  {'R@5':>8}  hits")
    for row in alpha_rows:
        print(f"{row['alpha']:>6.1f}  {row['precision@k']:>8.4f}  "
              f"{row['recall@k']:>8.4f}  {row['hits']:>4}")

    print("\nMMR, sweep over lambda")
    print(f"{'lambda':>6}  {'P@5':>8}  {'R@5':>8}  hits")
    for row in lambda_rows:
        print(f"{row['lambda']:>6.1f}  {row['precision@k']:>8.4f}  "
              f"{row['recall@k']:>8.4f}  {row['hits']:>4}")

    print(f"\nbest alpha:  {best_alpha}")
    print(f"best lambda: {best_lambda}")
    print(f"coverage-only (alpha = 0) vs random: {cov_only['hits']} hits against "
          f"{r['expected_hits']:.1f} expected, one-sided binomial "
          f"p = {r['p_coverage_only_ge_observed']:.3f}")
    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
