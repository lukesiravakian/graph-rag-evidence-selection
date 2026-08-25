# Results

Evaluation run outputs. One JSON file per run, written by `scripts/run_baseline.py`.

## File format

```json
{"results": {"precision@k": 0.0, "recall@k": 0.0}, "predictions": ["..."]}
```

- `results` — metrics dict returned by `eval.scorer.evaluate`. `null` in checkpoint
  files, which are written every `--checkpoint-every` questions before scoring runs.
- `predictions` — generated answer strings in dataset order, one per question that
  completed. Questions that error out are logged and skipped, so this list can be
  shorter than the dataset.

## Files

- `baseline_week1.json` — first end-to-end RAG run (dense retrieval → Gemini).
  Reports `precision@k` 0.233 / `recall@k` 0.936 over 21 answers.

## How to run

```bash
python retriever/build_index.py
```
or
```bash
python scripts/run_baseline.py --data data/hotpotqa_dev_300.jsonl \
                               --output results/baseline_week1.json --k 5
```

Generation calls Gemini, so `GOOGLE_API_KEY` must be set in `.env` (see `.env.example`).
