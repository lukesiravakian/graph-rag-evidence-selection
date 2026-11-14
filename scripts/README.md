# Scripts

Runnable entry points for the pipeline — one file per experiment. These scripts own
no logic of their own: they wire together `retriever/`, `selection/`, `generator/`,
and `eval/`, then write a JSON file into `results/`.

| Script | Writes | What it does | Gemini calls |
| --- | --- | --- | --- |
| `run_baseline.py` | `results/code_baseline_week1.json` | Dense top-5 retrieval → Gemini → retrieval metrics | 18 |
| `run_comparison.py` | `results/week2_comparison.json` | Same, for top-k vs. MMR vs. coverage-only facility location (α = 0) | 54 |
| `run_week3_comparison.py` | `results/week3_full_comparison.json` | **The paper's main comparison:** top-k vs. MMR vs. weighted facility location (α = 0.5) vs. reranking, on all four metrics | 144 |
| `run_structural_analysis.py` | `results/structural_analysis.json` | Density / clustering / average similarity of each selected set, against hit-miss | 0 |
| `run_alpha_instrumentation.py` | `results/alpha_instrumentation.json` | Per-step values of the two terms of Algorithm 1 | 0 |
| `run_miss_analysis.py` | `results/miss_analysis.json` | Whether each missed gold passage was even in the pool — retrieval failure vs. selection failure | 0 |
| `run_alpha_sweep.py` | `results/alpha_sweep.json` | Sweeps the FL weight α and the MMR weight λ over [0,1], plus the pool-recall ceiling and the random-selection reference | 0 |
| `make_figures.py` | `paper/figures/*.pdf` | Regenerates every paper figure from the result JSONs | 0 |
| `run_significance_tests.py` | `results/significance_tests.json` | Wilson intervals, McNemar bounds, Fisher exact test, from the saved JSON | 0 |

See [`results/README.md`](../results/README.md) for the numbers these produce and how
to read them.

## Before you run

1. **Install deps** — `pip install -r requirements.txt`.
2. **Build the FAISS index once** — every script except `run_significance_tests.py`
   imports `retriever.retrieve`, which loads the index at import time and raises a
   `RuntimeError` with build instructions if it is missing:

   ```bash
   python retriever/build_index.py
   ```

3. **Set `GOOGLE_API_KEY` in `.env`** — for the first three scripts, every question
   triggers a real Gemini call (`generator/generate.py`, model `gemini-3.6-flash`),
   and `run_week3_comparison.py` makes a second call per answer for the hallucination
   judge. See `.env.example`. The last three scripts in the table need no key:
   importing `eval.scorer` would construct the client, so they do not import it.
4. **Run from the repository root.** The scripts add the project root to `sys.path`
   themselves, so imports resolve from anywhere, but `retriever/retrieve.py` opens
   `data/code_passages.jsonl` and `retriever/passage_index.faiss` as *relative* paths.
   Launching from inside `scripts/` fails on those.

## How to run

```bash
python scripts/run_baseline.py              # week 1 baseline
python scripts/run_comparison.py            # week 2: + MMR, coverage-only FL
python scripts/run_week3_comparison.py      # week 3: + weighted FL, reranking
python scripts/run_structural_analysis.py   # selected-set structure
python scripts/run_alpha_instrumentation.py # the two terms of Algorithm 1
python scripts/run_miss_analysis.py         # retrieval failure vs. selection failure
python scripts/run_alpha_sweep.py           # sweep alpha and lambda
python scripts/run_significance_tests.py    # the paper's inferential statistics
python scripts/make_figures.py              # rebuild paper/figures from results/
```

Each also works as `python -m scripts.<name>` from the repository root.

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
`OUTPUT_PATH`, `K = 5`. There are no command-line flags.

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

Every script shares the same shape: the work lives in `main()` behind an
`if __name__ == "__main__"` guard, so importing one does not run the experiment —
though the module-level `from retriever.retrieve import ...` still loads the embedding
model and the FAISS index at import time.

Configuration is module constants at the top of each file — here `MODEL_NAME`, `K = 5`,
`CANDIDATE_SIZE = 15`, `INPUT_PATH`, `OUTPUT_PATH` — and writing is factored out into
`save_results()`. No script takes command-line flags; change the constants to change
the run.

## `run_week3_comparison.py`

The comparison behind Table II of the paper. Four methods, one shared pool, identical
budget:

- **`top_k`** — `retrieve(question, k=5)`, straight off the index.
- **`mmr`** — `mmr_select` over the 15-candidate pool, `lambda_param = 0.5`.
- **`facility_location_weighted`** — `relevance_weighted_facility_location_select`
  over the same pool with `alpha = 0.5`: Algorithm 1 of the paper, combining the
  marginal coverage gain with each candidate's cosine similarity to the question.
- **`reranking`** — `rerank_select` scores all 15 `(query, passage)` pairs with
  `cross-encoder/ms-marco-MiniLM-L-6-v2` and keeps the best 5.

Each method's five passages get their own answer from `generate_answer_with_citations`,
their own citation-precision score, and their own judge call. Output:

```json
{
  "comparison": {"top_k": {"precision@k": 0.0, "recall@k": 0.0, "citation_precision": 0.0, "hallucination_rate": 0.0}, "...": {}},
  "predictions": {"top_k": ["..."], "mmr": ["..."], "facility_location_weighted": ["..."], "reranking": ["..."]}
}
```

## `run_structural_analysis.py` and `run_alpha_instrumentation.py`

Both run weighted facility location on all 18 questions and write one row per
question. Neither calls the generator, so they are cheap to re-run.

`run_structural_analysis.py` builds a graph over the five selected passages with an
edge wherever pairwise cosine ≥ 0.5, then records `density`,
`clustering_coefficient`, `avg_pairwise_similarity`, and whether the gold passage was
selected.

`run_alpha_instrumentation.py` uses `relevance_weighted_facility_location_trace`,
which runs the identical greedy loop but also reports, for the candidate chosen at
each step, the coverage gain, the relevance term, and their ratio. This is what backs
the paper's claim that the nominal `alpha = 0.5` does not split influence evenly: the
coverage term sums over all 15 pool members while relevance is a single bounded
cosine, so coverage dominates the first pick or two.

## `run_miss_analysis.py`

Runs all five selection rules on every question and records, per question, whether the
gold passage reached the `n = 15` pool, its rank within the pool, how many distinct IDs
the top-k set held, and which methods selected it.

The point is to attribute each miss. A method that selects from the pool can only fail
in two ways: the gold passage was never in the pool (a retrieval failure, bounded by
`n`), or it was there and the rule passed it over (a selection failure). On this
benchmark pool recall is **18/18**, so the ceiling for MMR, facility location, and
reranking is 1.00 and every one of their misses is the second kind. That also means the
pool-size caveat often attached to diversity-aware methods does not bind here.

Needs the FAISS index and the cross-encoder; makes no Gemini calls.

## `run_significance_tests.py`

Needs no model, no index, and no API key — it reads `week3_full_comparison.json`,
`week2_comparison.json`, and `structural_analysis.json` and computes Wilson intervals,
exact McNemar p-values, and a two-sided Fisher exact test with `math.comb`. Because
the comparison scripts do not persist per-question hit/miss lists, the McNemar numbers
are reported as bounds over the discordance patterns consistent with the hit counts,
not as a single paired p-value.

## Behaviour worth knowing

- **All-or-nothing output.** No script checkpoints, and none wraps the Gemini
  call in `try`/`except`. The output file is written only after the final question, so
  a single API error partway through loses the entire run — worth knowing for
  `run_week3_comparison.py`, which is 144 calls deep. An earlier version of
  `run_baseline.py` did have `argparse` flags (`--data`, `--output`, `--k`,
  `--checkpoint-every`, `--log-level`) and mid-run checkpointing; the current rewrite
  dropped all of it in favour of module constants, so docs mentioning those flags are
  stale.
- **Cost scales with methods.** `run_baseline.py` makes 18 generation calls;
  `run_comparison.py` makes 18 × 3 = **54**, because every method gets its own answer
  for every question; `run_week3_comparison.py` makes 18 × 4 = 72 generation calls plus
  72 judge calls = **144**, and a citation retry adds one more per affected answer.
- **Duplicated embedding work in `run_comparison.py`.** `main()` builds its own
  `SentenceTransformer(MODEL_NAME)`, but importing `retriever.retrieve` has already
  loaded one at module level, so the same model sits in memory twice. Per question,
  the 15 candidate texts are encoded twice (once by `mmr_select`, once by
  `build_similarity_graph`) and the query three times (once each for `retrieve`,
  `retrieve_candidates`, and `mmr_select`). Correct, just slower than it needs to be.
- **Retrieved ids are not saved by the comparison scripts.** Only aggregate metrics
  and answer text reach their JSON, so per-question hit/miss analysis means re-running.
  `python retriever/retrieve.py` prints exactly that breakdown for the top-k baseline
  without spending any Gemini calls, and `run_structural_analysis.py` /
  `run_alpha_instrumentation.py` persist it for weighted facility location.
- **The analysis scripts re-run selection rather than reading it back.**
  `run_structural_analysis.py` and `run_alpha_instrumentation.py` each redo retrieval
  and weighted facility location from scratch, because the comparison scripts do not
  persist the selected IDs. Retrieval and selection are deterministic, so they
  reproduce the same sets `run_week3_comparison.py` used — but a change to the index
  or the embedding model silently decouples them from the saved comparison file.

## `run_alpha_sweep.py`

Sweeps the relevance weight α for facility location and the weight λ for MMR over
`{0.0, 0.1, …, 1.0}`, and reports two reference points the main comparison lacks: the
pool-recall ceiling and the expected recall of selecting `k` of `N` uniformly at random.

Both rules appear in the paper at a single hand-set value of 0.5, which leaves open
whether a result is a property of the method or of that one setting. The sweep closes
that gap at no API cost, since retrieval and selection are deterministic and no
generator is involved.

Each question's pool is embedded **once** and the similarity matrix and relevance vector
are reused across all 22 settings, so the whole sweep costs about the same as a single
comparison run. For speed it reimplements MMR against a precomputed matrix rather than
calling `selection.mmr.mmr_select`, which re-encodes; the rule is identical, including
the first-pick special case and the first-maximum tie-breaking. Three outputs act as
built-in checks on that: α = 0.5 must reproduce 13 hits, α = 0 must reproduce 8, and
λ = 0.5 must reproduce 13, matching `week3_full_comparison.json` and
`week2_comparison.json`.

## `make_figures.py`

Rebuilds `paper/figures/*.pdf` from the JSON files in `results/`. Needs `matplotlib`.
Every plotted number is read from a result file, so a figure cannot disagree with the
run that produced it — see [`../paper/README.md`](../paper/README.md) for why that
matters here.
