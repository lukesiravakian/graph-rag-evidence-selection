import json
from pathlib import Path
from sys import path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in path:
    path.insert(0, str(PROJECT_ROOT))

from retriever.retrieve import retrieve
from generator.generate import generate_answer
from eval.scorer import evaluate


# INPUT_PATH & OUTPUT_PATH
INPUT_PATH = PROJECT_ROOT / "data" / "code_questions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "results" / "code_baseline_week1.json"
MODE_READ = "r"
MODE_WRITE = "w"

K = 5


def load_data(path: Path) -> list[dict[str, object]]:
    """Load questions from a JSONL file."""
    with open(path, MODE_READ, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    data = load_data(INPUT_PATH)

    all_retrieved_ids = []
    all_gold_ids = []
    predictions = []

    print(f"Loaded {len(data)} questions.")

    for i, record in enumerate(data, start=1):
        question = record["question"]

        print(f"[{i}/{len(data)}] Processing question...")

        # Retrieve top-k passages
        passages = retrieve(question, k=K)

        retrieved_ids = [
            passage["passage_id"]
            for passage in passages
        ]

        # Generate an answer using the retrieved passages
        answer = generate_answer(question, passages)

        all_retrieved_ids.append(retrieved_ids)
        all_gold_ids.append(record["gold_passage_ids"])
        predictions.append(answer)

    # Evaluate retrieval performance
    results = evaluate(
        all_retrieved_ids,
        all_gold_ids,
    )

    print("\nResults:")
    print(json.dumps(results, indent=2))

    # Make sure the output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save results and generated answers
    output = {
        "results": results,
        "predictions": predictions,
    }

    with open(OUTPUT_PATH, MODE_WRITE, encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()