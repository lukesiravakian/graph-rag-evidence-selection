import json
from pathlib import Path
from sys import path

from sentence_transformers import SentenceTransformer


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in path:
    path.insert(0, str(PROJECT_ROOT))


from retriever.retrieve import retrieve, retrieve_candidates
from selection.graph_builder import build_similarity_graph
from selection.mmr import mmr_select
from selection.facility_location import facility_location_select
from generator.generate import generate_answer
from eval.scorer import compare_methods


# Configuration
MODEL_NAME = "all-mpnet-base-v2"
K = 5
CANDIDATE_SIZE = 15

INPUT_PATH = PROJECT_ROOT / "data" / "code_questions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "week2_comparison.json"


def load_data(path: Path) -> list[dict]:
    """Load questions from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_results(
    path: Path,
    comparison: dict,
    predictions: dict,
) -> None:
    """Save comparison results and generated predictions."""
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "comparison": comparison,
        "predictions": predictions,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main() -> None:
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    data = load_data(INPUT_PATH)
    print(f"Loaded {len(data)} questions.")

    results_by_method = {
        "top_k": ([], []),
        "mmr": ([], []),
        "facility_location": ([], []),
    }

    predictions_by_method = {
        "top_k": [],
        "mmr": [],
        "facility_location": [],
    }

    for i, record in enumerate(data, start=1):
        question = record["question"]
        gold_ids = record["gold_passage_ids"]

        print(f"\n[{i}/{len(data)}] Processing question...")

        # Method 1: Plain Top-K retrieval
        top_k_passages = retrieve(question, k=K)
        top_k_ids = [p["passage_id"] for p in top_k_passages]

        # Shared candidate pool for MMR and Facility Location
        candidates = retrieve_candidates(
            question,
            n=CANDIDATE_SIZE,
        )

        # Method 2: MMR selection
        mmr_passages = mmr_select(
            model,
            question,
            candidates,
            k=K,
        )
        mmr_ids = [p["passage_id"] for p in mmr_passages]

        # Method 3: Facility Location selection
        sim_matrix, _ = build_similarity_graph(
            candidates,
            model,
        )

        fl_indices = facility_location_select(
            sim_matrix,
            k=K,
        )

        fl_passages = [candidates[i] for i in fl_indices]
        fl_ids = [p["passage_id"] for p in fl_passages]

        # Store retrieval results and generate answers
        methods = [
            ("top_k", top_k_passages, top_k_ids),
            ("mmr", mmr_passages, mmr_ids),
            ("facility_location", fl_passages, fl_ids),
        ]

        for method, passages, retrieved_ids in methods:
            results_by_method[method][0].append(retrieved_ids)
            results_by_method[method][1].append(gold_ids)

            answer = generate_answer(question, passages)
            predictions_by_method[method].append(answer)

        print("  Finished.")


    # Compare retrieval performance
    comparison = compare_methods(results_by_method)

    print("\n=== Comparison Results ===")
    print(json.dumps(comparison, indent=2))

    # Save results
    save_results(
        OUTPUT_PATH,
        comparison,
        predictions_by_method,
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()