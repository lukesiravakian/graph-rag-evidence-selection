"""
Evaluate a RAG (Retrieval-Augmented Generation) pipeline on a QA dataset.

Usage:
    python run_baseline.py

Output format:
    {"results": <metrics dict>, "predictions": [<answers>]}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retriever.retrieve import retrieve
from generator.generate import generate_answer
from eval.scorer import evaluate

logger = logging.getLogger("rag_eval")


@dataclass
class EvalConfig:
    data_path: Path
    output_path: Path
    top_k: int = 5
    checkpoint_every: int = 50
    log_level: str = "INFO"


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file, raising an error if it's missing or malformed."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_num} of {path}: {e}") from e

    if not records:
        raise ValueError(f"No records found in {path}")

    logger.info("Loaded %d records from %s", len(records), path)
    return records


def run_pipeline(
    records: list[dict[str, Any]],
    top_k: int,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 50,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """
    Run retrieval + generation for every record.

    Returns (predictions, golds, failures). A failure on one record is logged
    and skipped instead of crashing the entire run. predictions and golds
    always stay aligned 1:1 (a failed record contributes to neither list).
    """
    predictions: list[str] = []
    golds: list[str] = []
    failures: list[dict[str, Any]] = []

    for i, record in enumerate(tqdm(records, desc="Running RAG pipeline", unit="q")):
        question = record.get("question")
        gold = record.get("answer")

        if not question or gold is None:
            logger.warning("Skipping record %d: missing 'question' or 'answer' field", i)
            failures.append({"index": i, "record": record, "error": "missing_fields"})
            continue

        try:
            passages = retrieve(question, k=top_k)
            answer = generate_answer(question, passages)
        except Exception as e:  # noqa: BLE001 - deliberately broad so one bad record doesn't kill the run
            logger.error("Record %d failed (%s): %s", i, question[:60], e)
            failures.append({"index": i, "question": question, "error": str(e)})
            continue

        predictions.append(answer)
        golds.append(gold)

        if checkpoint_path and checkpoint_every and (i + 1) % checkpoint_every == 0:
            _save_output(checkpoint_path, results=None, predictions=predictions)
            logger.debug("Checkpoint saved at record %d", i + 1)

    if failures:
        logger.warning("%d/%d records failed and were skipped", len(failures), len(records))

    return predictions, golds, failures


def _save_output(output_path: Path, results: dict[str, Any] | None, predictions: list[str]) -> None:
    """
    Write results to disk with the SAME shape as the original script:
        {"results": ..., "predictions": [...]}
    Uses an atomic write (write to .tmp then rename) so the file is never
    left half-written if the process is interrupted mid-save.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": results, "predictions": predictions}
    tmp_path = output_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(output_path)


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate a RAG pipeline on a QA dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/hotpotqa_dev_300.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline_week1.json"))
    parser.add_argument("--k", type=int, default=5, help="Number of passages to retrieve per question.")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save a checkpoint every N questions.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    return EvalConfig(
        data_path=args.data,
        output_path=args.output,
        top_k=args.k,
        checkpoint_every=args.checkpoint_every,
        log_level=args.log_level,
    )


def main() -> None:
    config = parse_args()
    setup_logging(config.log_level)

    logger.info("Starting RAG evaluation | data=%s | k=%d", config.data_path, config.top_k)
    start = time.time()

    records = load_jsonl(config.data_path)
    predictions, golds, failures = run_pipeline(
        records,
        top_k=config.top_k,
        checkpoint_path=config.output_path,
        checkpoint_every=config.checkpoint_every,
    )

    if not predictions:
        logger.error("No predictions were generated - aborting before scoring.")
        sys.exit(1)

    results = evaluate(predictions, golds)
    elapsed = time.time() - start

    print(results)
    logger.info(
        "Finished %d/%d questions in %.1fs (%.2fs/question, %d skipped)",
        len(predictions), len(records), elapsed, elapsed / max(len(predictions), 1), len(failures),
    )

    _save_output(config.output_path, results=results, predictions=predictions)
    logger.info("Saved results to %s", config.output_path)


if __name__ == "__main__":
    main()
