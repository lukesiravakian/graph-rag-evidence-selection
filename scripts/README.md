# Scripts

Runnable entry points for the pipeline — one file per experiment. These scripts own
no logic of their own: they wire together `retriever/`, `selection/`, `generator/`,
and `eval/`, then write a JSON file into `results/`.

| Script | Writes | What it does |
| --- | --- | --- |
| `run_baseline.py` | `results/code_baseline_week1.json` | Dense top-5 retrieval → Gemini → retrieval metrics |
| `run_comparison.py` | `results/week2_comparison.json` | Same, for three selection methods side by side |

See [`results/README.md`](../results/README.md) for the numbers these produce and how
to read them.

## Before you run

1. **Install deps** — `pip install -r requirements.txt`.
2. **Build the FAISS index once** — both scripts import `retriever.retrieve`, which
   loads the index at import time and raises a `RuntimeError` with build instructions
   if it is missing:

   ```bash
   python retriever/build_index.py
   ```

3. **Set `GOOGLE_API_KEY` in `.env`** — every question triggers a real Gemini call
   (`generator/generate.py`, model `gemini-3.6-flash`). See `.env.example`.
4. **Run from the repository root.** The scripts add the project root to `sys.path`
   themselves, so imports resolve from anywhere, but `retriever/retrieve.py` opens
   `data/code_passages.jsonl` and `retriever/passage_index.faiss` as *relative* paths.
   Launching from inside `scripts/` fails on those.

## How to run

```bash
python scripts/run_baseline.py
```

```bash
python scripts/run_comparison.py
```

Both also work as `python -m scripts.run_baseline` / `python -m scripts.run_comparison`
from the repository root.

## `run_baseline.py`

The week 1 baseline: plain dense retrieval, no evidence selection.

For each of the 18 questions in `data/code_questions.jsonl` it calls
`retrieve(question, k=5)`, passes those five passages to `generate_answer`, and
collects the retrieved ids alongside the record's `gold_passage_ids`. After the loop,
`eval.scorer.evaluate` macro-averages `precision@k` and `recall@k` over all questions,
and the script writes:

```json
{"results": {"precision@k": 0.0, "recall@k": 0.0}, "predictions": ["..."]}
```

`predictions` is in dataset order (`q1` … `q18`). Progress is printed per question and
the final metrics are echoed to stdout before saving.

Configuration lives in module constants at the top of the file — `INPUT_PATH`,
`OUTPUT_PATH`, `K = 5`. There are no command-line flags; change the constants to
change the run.

## `run_comparison.py`

The week 2 comparison. Same corpus, questions, and budget (`k = 5`) for all three
methods, so only the *selection* strategy differs:

- **`top_k`** — `retrieve(question, k=5)`. Identical to the baseline above.
- **`mmr`** — `retrieve_candidates(question, n=15)`, then `selection.mmr.mmr_select`
  picks 5 (default `lambda_param = 0.5`, set in `selection/mmr.py`).
- **`facility_location`** — the *same* 15 candidates go to
  `selection.graph_builder.build_similarity_graph` for a passage × passage cosine
  matrix, then `selection.facility_location.facility_location_select` greedily picks 5.

Each method's five passages get their own generated answer and their own score.
`eval.scorer.compare_methods` produces one metrics dict per method:

```json
{
  "comparison": {"top_k": {"precision@k": 0.0, "recall@k": 0.0}, "mmr": {}, "facility_location": {}},
  "predictions": {"top_k": ["..."], "mmr": ["..."], "facility_location": ["..."]}
}
```

Unlike `run_baseline.py`, this script is written as top-level module code with no
`main()` and no `if __name__ == "__main__"` guard — **importing it runs the whole
experiment**. Its `k=5` and `n=15` are inline literals in the loop, not constants.

## Behaviour worth knowing

- **All-or-nothing output.** Neither script checkpoints, and neither wraps the Gemini
  call in `try`/`except`. The output file is written only after the final question, so
  a single API error partway through loses the entire run. An earlier version of
  `run_baseline.py` did have `argparse` flags (`--data`, `--output`, `--k`,
  `--checkpoint-every`, `--log-level`) and mid-run checkpointing; the current rewrite
  dropped all of it in favour of module constants, so docs mentioning those flags are
  stale.
- **Cost scales with methods.** `run_baseline.py` makes 18 generation calls;
  `run_comparison.py` makes 18 × 3 = **54**, because every method gets its own answer
  for every question.
- **Duplicated embedding work in `run_comparison.py`.** It creates its own
  `SentenceTransformer("all-mpnet-base-v2")` while `retriever/retrieve.py` already
  holds one at module level, so the model sits in memory twice. Per question, the
  15 candidate texts are encoded twice (once by `mmr_select`, once by
  `build_similarity_graph`) and the query is encoded twice (once for `retrieve`, once
  for `retrieve_candidates`). Correct, just slower than it needs to be.
- **Retrieved ids are not saved.** Only aggregate metrics and answer text reach the
  JSON, so per-question hit/miss analysis means re-running. `python
  retriever/retrieve.py` prints exactly that breakdown for the top-k baseline without
  spending any Gemini calls.
- **`__init__.py` is out of date.** Its docstring still describes `run_baseline` as
  scoring "the 300-question HotpotQA dev subset" into `results/baseline_week1.json`.
  The code reads the 18 code questions and writes `code_baseline_week1.json`; trust
  the code.
