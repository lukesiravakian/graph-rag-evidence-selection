"""Section VI-C, first proposed analysis: was each missed gold passage even available?

For every question this records whether the gold passage reached the N = 15 candidate
pool and whether it reached the top 5. The distinction matters because it decides what
a miss means:

  gold not in pool  -> the retriever never made it available; no selection rule could
                       have recovered it, and the pool size is the binding constraint.
  gold in pool but
  not selected      -> the selection rule discarded an available answer.

The pool-recall figure is also the ceiling for every method that selects from the pool
(MMR, facility location, reranking), so it is the number against which their recall@5
should be read.

Needs the FAISS index; makes no Gemini calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from sys import path
from typing import Any

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in path:
    path.insert(0, str(PROJECT_ROOT))

from retriever.retrieve import retrieve, retrieve_candidates
from selection.graph_builder import build_similarity_graph, compute_query_similarities
from selection.facility_location import (
    facility_location_select,
    relevance_weighted_facility_location_select,
)
from selection.mmr import mmr_select
from selection.reranker import rerank_select

EMBEDDING_MODEL = "all-mpnet-base-v2"
TOP_K = 5
NUM_CANDIDATES = 15
ALPHA = 0.5

INPUT_PATH = PROJECT_ROOT / "data" / "code_questions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "miss_analysis.json"

POOL_METHODS = ("mmr", "facility_location_weighted", "facility_location_coverage_only",
                "reranking")


def main() -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)
    with INPUT_PATH.open(encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    rows: list[dict[str, Any]] = []

    for record in data:
        question: str = record["question"]
        gold = set(record["gold_passage_ids"])

        candidates = retrieve_candidates(question, n=NUM_CANDIDATES)
        pool_ids = [p["passage_id"] for p in candidates]
        top5_ids = [p["passage_id"] for p in retrieve(question, k=TOP_K)]

        sim_matrix, _ = build_similarity_graph(candidates, model)
        query_sims = compute_query_similarities(question, candidates, model)

        selections = {
            "top_k": top5_ids,
            "mmr": [p["passage_id"] for p in mmr_select(model, question, candidates, k=TOP_K)],
            "facility_location_weighted": [
                pool_ids[i] for i in relevance_weighted_facility_location_select(
                    sim_matrix, query_sims, k=TOP_K, alpha=ALPHA)
            ],
            "facility_location_coverage_only": [
                pool_ids[i] for i in facility_location_select(sim_matrix, k=TOP_K)
            ],
            "reranking": [p["passage_id"] for p in rerank_select(question, candidates, k=TOP_K)],
        }

        in_pool = bool(gold & set(pool_ids))
        rows.append({
            "qid": record["qid"],
            "gold_in_pool": in_pool,
            "gold_rank_in_pool": next(
                (i + 1 for i, pid in enumerate(pool_ids) if pid in gold), None),
            "distinct_ids_in_top_k": len(set(top5_ids)),
            "hits": {m: bool(gold & set(ids)) for m, ids in selections.items()},
        })

    n = len(rows)
    pool_recall = sum(r["gold_in_pool"] for r in rows) / n

    per_method = {}
    for method in ("top_k",) + POOL_METHODS:
        misses = [r for r in rows if not r["hits"][method]]
        # A miss is a selection error only if the gold passage was in the pool. Top-k
        # does not read the pool, so its misses are retrieval-ranking errors by
        # definition; they are counted here as "recoverable from the pool" instead.
        recoverable = [r["qid"] for r in misses if r["gold_in_pool"]]
        per_method[method] = {
            "hits": n - len(misses),
            "recall": (n - len(misses)) / n,
            "missed_qids": [r["qid"] for r in misses],
            "misses_with_gold_in_pool": recoverable,
            "misses_with_gold_outside_pool":
                [r["qid"] for r in misses if not r["gold_in_pool"]],
        }

    output = {
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "num_candidates": NUM_CANDIDATES,
            "k": TOP_K,
            "alpha": ALPHA,
            "questions": n,
        },
        "pool_recall": pool_recall,
        "pool_recall_hits": sum(r["gold_in_pool"] for r in rows),
        "per_method": per_method,
        "per_question": rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Gold passage in the pool of {NUM_CANDIDATES}: "
          f"{output['pool_recall_hits']}/{n} = {pool_recall:.4f}")
    print(f"This is the recall@{TOP_K} ceiling for every method that selects from the pool.\n")
    print(f"{'method':34s} {'recall':>7}  misses  of which recoverable from the pool")
    for method, v in per_method.items():
        print(f"  {method:32s} {v['recall']:.4f}  "
              f"{len(v['missed_qids']):>6}  "
              f"{len(v['misses_with_gold_in_pool'])} "
              f"{sorted(v['misses_with_gold_in_pool'])}")

    outside = [r["qid"] for r in rows if not r["gold_in_pool"]]
    if outside:
        print(f"\nGold outside the pool for: {outside} — for these the pool size, not the "
              f"selection rule, is the binding constraint.")
    else:
        print(f"\nNo question has its gold passage outside the pool, so on this benchmark "
              f"every miss by a pool-based method is a selection error, not a retrieval "
              f"failure, and the pool-size caveat does not bind.")

    dup = [(r["qid"], r["distinct_ids_in_top_k"]) for r in rows
           if r["distinct_ids_in_top_k"] < TOP_K]
    if dup:
        print(f"\nTop-k sets with fewer than {TOP_K} distinct IDs (duplicate passage IDs): {dup}")
    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
