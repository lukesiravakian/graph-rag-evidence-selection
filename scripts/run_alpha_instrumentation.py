"""Section III-E / VI-B and Fig. 3a: how the two score terms of Algorithm 1 compare.

The paper claims the nominal alpha = 0.5 split is not realized in practice, because
the coverage gain sums a contribution over all N = 15 pool members while rel(c) is a
single bounded cosine. This script instruments Algorithm 1 on the benchmark and
reports, for the candidate actually chosen at each step, the value of delta_c, the
value of rel(c), and their ratio.

Selection is untouched: the trace comes from the same greedy loop that produces
every result in the paper, so the selected sets here are identical to those in
results/week3_full_comparison.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from sys import path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in path:
    path.insert(0, str(PROJECT_ROOT))

from retriever.retrieve import retrieve_candidates
from selection.graph_builder import build_similarity_graph, compute_query_similarities
from selection.facility_location import relevance_weighted_facility_location_trace

EMBEDDING_MODEL = "all-mpnet-base-v2"
TOP_K = 5
NUM_CANDIDATES = 15
ALPHA = 0.5

INPUT_PATH = PROJECT_ROOT / "data" / "code_questions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "alpha_instrumentation.json"


def summarize_step(traces: list[list[dict[str, float]]], step: int) -> dict[str, float]:
    """Range of both terms, and of their ratio, across questions at one step."""
    coverage = [t[step - 1]["coverage_gain"] for t in traces]
    relevance = [t[step - 1]["relevance"] for t in traces]
    ratios = [t[step - 1]["ratio"] for t in traces]
    return {
        "step": step,
        "coverage_gain_min": min(coverage),
        "coverage_gain_max": max(coverage),
        "relevance_min": min(relevance),
        "relevance_max": max(relevance),
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "questions_where_relevance_exceeds_coverage":
            sum(1 for c, r in zip(coverage, relevance) if r > c),
        "questions_within_factor_two":
            sum(1 for c, r in zip(coverage, relevance) if r > 0 and c / r <= 2.0),
    }


def main() -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)
    with INPUT_PATH.open(encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    per_question: list[dict[str, Any]] = []
    traces: list[list[dict[str, float]]] = []

    for record in data:
        question: str = record["question"]

        candidates = retrieve_candidates(question, n=NUM_CANDIDATES)
        sim_matrix, _ = build_similarity_graph(candidates, model)
        query_sims = compute_query_similarities(question, candidates, model)

        selected_idx, trace = relevance_weighted_facility_location_trace(
            sim_matrix, query_sims, k=TOP_K, alpha=ALPHA
        )
        traces.append(trace)
        per_question.append({
            "qid": record["qid"],
            "selected_ids": [candidates[i]["passage_id"] for i in selected_idx],
            "trace": trace,
        })

    by_step = [summarize_step(traces, s) for s in range(1, TOP_K + 1)]

    output = {
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "num_candidates": NUM_CANDIDATES,
            "k": TOP_K,
            "alpha": ALPHA,
            "questions": len(data),
        },
        "by_step": by_step,
        "per_question": per_question,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    first = by_step[0]
    print(f"Instrumented Algorithm 1 on {len(data)} questions "
          f"(alpha = {ALPHA}, N = {NUM_CANDIDATES}, k = {TOP_K}).\n")
    print(f"{'step':>4}  {'coverage gain':>20}  {'relevance':>18}  {'ratio':>14}  rel > cov")
    for s in by_step:
        print(f"{s['step']:>4}  "
              f"{s['coverage_gain_min']:8.2f} to {s['coverage_gain_max']:<8.2f}  "
              f"{s['relevance_min']:7.2f} to {s['relevance_max']:<7.2f}  "
              f"{s['ratio_min']:5.1f} to {s['ratio_max']:<5.1f}  "
              f"{s['questions_where_relevance_exceeds_coverage']:>2}/{len(data)}")
    print(f"\nFirst pick: coverage gain {first['coverage_gain_min']:.1f} to "
          f"{first['coverage_gain_max']:.1f} against relevance "
          f"{first['relevance_min']:.2f} to {first['relevance_max']:.2f}, "
          f"a ratio of {first['ratio_min']:.0f} to {first['ratio_max']:.0f}.")
    third = by_step[2]
    print(f"By the third step {third['questions_within_factor_two']}/{len(data)} questions "
          f"have the two terms within a factor of two, and relevance exceeds coverage "
          f"for {third['questions_where_relevance_exceeds_coverage']}.")
    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
