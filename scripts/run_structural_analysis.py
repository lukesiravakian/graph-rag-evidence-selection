"""Section V-B: structure of the evidence sets picked by weighted facility location.

For every question this records the density, clustering coefficient, and average
pairwise similarity of the selected five-passage subgraph (edges at cosine >= 0.5)
together with whether the gold passage was among the five.
"""
from __future__ import annotations

import json
from pathlib import Path
from sys import path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to Python path so the script runs from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in path:
    path.insert(0, str(PROJECT_ROOT))

from retriever.retrieve import retrieve_candidates
from selection.graph_builder import build_similarity_graph, compute_query_similarities
from selection.facility_location import relevance_weighted_facility_location_select
from selection.structural_analysis import compute_subgraph_properties

EMBEDDING_MODEL = "all-mpnet-base-v2"
TOP_K = 5
NUM_CANDIDATES = 15
ALPHA = 0.5

INPUT_PATH = PROJECT_ROOT / "data" / "code_questions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "structural_analysis.json"


def main() -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)
    with INPUT_PATH.open(encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    rows: list[dict[str, Any]] = []

    for record in data:
        question: str = record["question"]
        gold_ids: list[str] = record["gold_passage_ids"]

        candidates = retrieve_candidates(question, n=NUM_CANDIDATES)
        sim_matrix, _ = build_similarity_graph(candidates, model)
        query_sims = compute_query_similarities(question, candidates, model)

        selected_idx = relevance_weighted_facility_location_select(
            sim_matrix, query_sims, k=TOP_K, alpha=ALPHA
        )
        selected_ids = [candidates[i]["passage_id"] for i in selected_idx]

        hit = 1 if any(g in selected_ids for g in gold_ids) else 0
        props = compute_subgraph_properties(sim_matrix, selected_idx)

        rows.append({
            "qid": record["qid"],
            "density": props["density"],
            "clustering_coefficient": props["clustering_coefficient"],
            "avg_pairwise_similarity": props["avg_pairwise_similarity"],
            "hit": hit,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote structural analysis for {len(rows)} questions.")

    hits = [r for r in rows if r["hit"] == 1]
    misses = [r for r in rows if r["hit"] == 0]
    if hits:
        print(f"Hits ({len(hits)}): avg pairwise similarity = "
              f"{np.mean([r['avg_pairwise_similarity'] for r in hits]):.3f}")
    if misses:
        print(f"Misses ({len(misses)}): avg pairwise similarity = "
              f"{np.mean([r['avg_pairwise_similarity'] for r in misses]):.3f}")


if __name__ == "__main__":
    main()
