from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
from sys import path

from sentence_transformers import SentenceTransformer

# Add project root to Python path
# If after this has errors, please delete this! 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in path:
    path.insert(0, str(PROJECT_ROOT))


from retriever.retrieve import retrieve, retrieve_candidates
from selection.graph_builder import build_similarity_graph, compute_query_similarities
from selection.mmr import mmr_select
from selection.facility_location import relevance_weighted_facility_location_select
from selection.reranker import rerank_select
from generator.generate import generate_answer_with_citations
from eval.scorer import evaluate, compare_methods, citation_precision, check_hallucination

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DATA_PATH = DATA_DIR / "code_questions.jsonl"
OUTPUT_PATH = RESULTS_DIR / "week3_full_comparison.json"

EMBEDDING_MODEL = "all-mpnet-base-v2"
TOP_K = 5
NUM_CANDIDATES = 15
METHODS = ["top_k", "mmr", "facility_location_weighted", "reranking"]

# A passage is a dict with at least a "passage_id" key.
Passage = dict[str, Any]
PassageId = str
IdLists = list[list[PassageId]]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def select_passages(model: SentenceTransformer, question: str) -> dict[str, list[Passage]]:
    """Run every selection strategy on the same question, in the order of METHODS."""
    candidates: list[Passage] = retrieve_candidates(question, n=NUM_CANDIDATES)

    selections: dict[str, list[Passage]] = {}
    selections["top_k"] = retrieve(question, k=TOP_K)
    selections["mmr"] = mmr_select(model, question, candidates, k=TOP_K)

    sim_matrix, _ = build_similarity_graph(candidates, model)
    query_sims = compute_query_similarities(question, candidates, model)
    fl_idx: list[int] = relevance_weighted_facility_location_select(sim_matrix, query_sims, k=TOP_K)
    selections["facility_location_weighted"] = [candidates[i] for i in fl_idx]

    selections["reranking"] = rerank_select(question, candidates, k=TOP_K)
    return selections


def main() -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)
    data = load_jsonl(DATA_PATH)

    results_by_method: dict[str, tuple[IdLists, IdLists]] = {m: ([], []) for m in METHODS}
    predictions_by_method: dict[str, list[str]] = {m: [] for m in METHODS}
    citation_scores: dict[str, list[float]] = {m: [] for m in METHODS}
    hallucination_flags: dict[str, list[bool]] = {m: [] for m in METHODS}

    for record in data:
        question: str = record["question"]
        gold_ids: list[PassageId] = record["gold_passage_ids"]

        for method, passages in select_passages(model, question).items():
            provided_ids = [p["passage_id"] for p in passages]
            answer, cited_ids = generate_answer_with_citations(question, passages)

            provided_list, gold_list = results_by_method[method]
            provided_list.append(provided_ids)
            gold_list.append(gold_ids)

            predictions_by_method[method].append(answer)
            citation_scores[method].append(citation_precision(cited_ids, provided_ids))
            hallucination_flags[method].append(check_hallucination(question, answer, passages))

    comparison: dict[str, dict[str, Any]] = compare_methods(results_by_method)
    for method in METHODS:
        comparison[method]["citation_precision"] = mean(citation_scores[method])
        comparison[method]["hallucination_rate"] = mean(hallucination_flags[method])

    print(comparison)

    with OUTPUT_PATH.open("w") as f:
        json.dump({
                "comparison": comparison, 
                "predictions": predictions_by_method
            }, f, indent=2
        )


if __name__ == "__main__":
    main()