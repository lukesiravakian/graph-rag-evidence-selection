"""Encode the benchmark corpus and queries once, and cache the vectors.

Everything downstream -- retrieval, every selection rule, every value of alpha and
lambda -- is then a numpy operation over these matrices. The single-repo pipeline
re-encoded the same passages three times per question and once per sweep setting,
which is why a sweep there cost tens of minutes; here it costs seconds.

Note the batch size. ``retriever/build_index.py`` encodes with ``batch_size=1``,
which is one forward pass per passage and leaves the hardware idle; 3500 passages
at batch 32 is the same arithmetic in a fraction of the time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BENCH = Path("benchmark")
MODEL_NAME = "all-mpnet-base-v2"
BATCH = 32


def main() -> None:
    passages = [json.loads(l) for l in (BENCH / "passages.jsonl").open()]
    questions = [json.loads(l) for l in (BENCH / "questions.jsonl").open()]
    print(f"{len(passages)} passages, {len(questions)} questions")

    model = SentenceTransformer(MODEL_NAME)

    print("encoding passages...")
    pe = model.encode([p["text"] for p in passages], batch_size=BATCH,
                      show_progress_bar=True, convert_to_numpy=True)
    print("encoding queries...")
    qe = model.encode([q["question"] for q in questions], batch_size=BATCH,
                      show_progress_bar=True, convert_to_numpy=True)

    pe = np.asarray(pe, dtype=np.float32)
    qe = np.asarray(qe, dtype=np.float32)
    pe /= np.linalg.norm(pe, axis=1, keepdims=True)
    qe /= np.linalg.norm(qe, axis=1, keepdims=True)

    np.save(BENCH / "passage_emb.npy", pe)
    np.save(BENCH / "query_emb.npy", qe)
    (BENCH / "passage_ids.json").write_text(
        json.dumps([p["passage_id"] for p in passages]))
    (BENCH / "question_ids.json").write_text(
        json.dumps([q["qid"] for q in questions]))

    print(f"wrote passage_emb {pe.shape}, query_emb {qe.shape}")


if __name__ == "__main__":
    main()
