# Scripts

Entry points for the experiments run so far, one file per protocol. Nothing in this
folder implements retrieval, selection, generation or scoring. Each script composes
those packages (`retriever/`, `selection/`, `generator/`, `eval/`) for a single
experimental design and writes one JSON file to `results/`.

| Script | Protocol | Writes |
| --- | --- | --- |
| `run_baseline.py` | Week 1. Dense top-k, no selection step | `results/code_baseline_week1.json` |
| `run_comparison.py` | Week 2. Top-k vs. MMR vs. facility location | `results/week2_comparison.json` |
| `run_week3_comparison.py` | Week 3. Adds relevance-weighted facility location and cross-encoder reranking; scores citations and hallucination | `results/week3_full_comparison.json` (not produced yet) |

`results/README.md` documents the output files themselves. This page covers the
experimental setup the scripts encode, the numbers they have produced, and what those
numbers can and cannot support.

## Experimental setup

The following is held fixed across all three scripts, which is what makes the
week-to-week numbers comparable.

**Corpus.** 320 function- and class-level chunks cut from the `requests` source tree
with `ast` (`data/build_code_dataset.py`), stored as `data/code_passages.jsonl`. A
chunk's `passage_id` is `<file_path>::<node name>`. That naming is not unique: 23 ids
are shared by 63 rows (`auth.py::__init__` alone appears six times), leaving 280
distinct ids for 320 vectors. The effect on scoring is taken up under *Threats to
validity*.

**Questions.** 18 hand-written questions about that code (`data/code_questions.jsonl`),
each annotated with exactly one gold chunk.

**Retriever.** `all-mpnet-base-v2` sentence embeddings, L2-normalized, in a FAISS
`IndexFlatIP`. Inner product over unit vectors, so cosine similarity, exact search.
The index is built once by `python retriever/build_index.py`.

**Candidate pool.** Every selection method works on the 15 nearest neighbours of the
question. `retrieve_candidates(question, n=15)` is literally `retrieve(question, k=15)`,
so the top-5 baseline is a prefix of the pool every other method draws from, and no
method can surface a gold chunk ranked below 15.

**Budget.** k = 5 passages per question, for every method, every week.

**Generator.** `gemini-3.6-flash`, temperature 0.1, `max_output_tokens` 1000. Weeks 1
and 2 call `generate_answer` and get a free-form one- or two-sentence answer. Week 3
calls `generate_answer_with_citations`, which numbers the passages in the prompt,
demands a trailing `CITATIONS:` line containing exact passage ids, and re-prompts once
if no valid id comes back.

## Methods under comparison

- `top_k`: the five nearest neighbours. Serves as the baseline in every week.
- `mmr` (`selection/mmr.py`): greedy maximal marginal relevance with λ = 0.5. The
  first pick is the most query-similar chunk; each later pick maximises
  λ·cos(q, p) − (1 − λ)·max<sub>s ∈ S</sub> cos(p, s).
- `facility_location` (week 2 only): `build_similarity_graph` produces the 15 × 15
  cosine matrix; `facility_location_select` greedily adds the chunk with the largest
  increase in Σ<sub>j</sub> max<sub>s ∈ S</sub> sim(j, s). The query is not part of the
  objective at all; it only chose the pool.
- `facility_location_weighted` (week 3): `relevance_weighted_facility_location_select`
  scores each candidate by (1 − α)·coverage gain + α·cos(q, p), with α = 0.5. The
  script relies on the function default and does not set α itself.
- `reranking` (week 3, `selection/reranker.py`): `cross-encoder/ms-marco-MiniLM-L-6-v2`
  scores each (question, chunk) pair over the pool; the top five by cross-encoder score
  are kept. This is the standard rerank baseline the project set out to beat.

## Metrics

Retrieval quality is scored per question on the *set* of retrieved ids R against the
gold set G: precision = |R ∩ G| / |R|, recall = |R ∩ G| / |G|. Both are macro-averaged
over the 18 questions (`eval.scorer.evaluate`; `compare_methods` maps it over methods).
Because |G| = 1 and k = 5, recall@5 is simply the hit rate (recall × 18 = questions
hit), and precision@5 cannot exceed 0.2.

Week 3 adds two answer-level quantities, each averaged over questions:

- `citation_precision`: of the ids the model cites, the fraction that belong to the
  five passages it was actually shown.
- `hallucination_rate`: a second `gemini-3.6-flash` call is given the context, the
  question and the answer, and asked YES or NO whether the answer makes claims the
  context does not support. The rate is the share of YES.

No week scores answer correctness. The generated text is saved under `predictions` and
nothing downstream reads it.

## Results

### Week 1: baseline

`code_baseline_week1.json`: precision@5 = 0.1630, recall@5 = 0.7778. The gold chunk
landed in the top five for 14 of 18 questions.

### Week 2: selection strategies

| Method | precision@5 | recall@5 | Questions hit |
| --- | --- | --- | --- |
| `top_k` | 0.1630 | 0.7778 | 14 / 18 |
| `mmr` | 0.1444 | 0.7222 | 13 / 18 |
| `facility_location` | 0.0889 | 0.4444 | 8 / 18 |

Top-k reproduces week 1 to the last digit, which it must: same index, same query
encoder, same k. MMR gives up one question. Facility location gives up six, and that
loss is not sampling noise. The objective rewards chunks that are central to the pool,
and the pool holds fourteen chunks that are not the answer; optimising coverage of them
is optimising away from the gold. With a 320-chunk corpus and one gold per question,
a query-blind diversity criterion was never going to match top-k on recall. What week
2 pins down is the size of the penalty: roughly half the hits.

The MMR gap is a single question, 5.6 recall points. At n = 18 that is the granularity
of the measurement, not a finding.

### Week 3: relevance-weighted selection, reranking, citation scoring

`run_week3_comparison.py` is written but has not been executed on this branch, and
`results/week3_full_comparison.json` does not exist. The script imports three
functions that live on sibling branches and have not been merged here yet:

| Symbol | Where it lives |
| --- | --- |
| `selection.reranker.rerank_select` | `sukanya/week3-reranker` |
| `generator.generate.generate_answer_with_citations` | `sasha/citation-generation` |
| `eval.scorer.citation_precision`, `eval.scorer.check_hallucination` | `derek/week3-eval` |

Until those land the import block fails. The protocol itself is settled: same 18
questions, same 15-candidate pool, four methods, and per method precision@5, recall@5,
`citation_precision` and `hallucination_rate`.

Two caveats are worth writing down before the numbers exist, so that nobody
over-reads them afterwards.

*`citation_precision` is close to degenerate.* `generate_answer_with_citations` already
drops every cited id that is not in the provided set before it returns. So
`citation_precision(cited_ids, provided_ids)` evaluates to 1.0 whenever at least one
valid citation survived and to 0.0 otherwise. The column will report the fraction of
answers that came back with a usable `CITATIONS:` line. That is a legitimate quantity.
It is not precision.

*α = 0.5 does not split the weighted objective in half.* The coverage gain is a sum
over all 15 candidates; on the first pick it is the row sum of the similarity matrix,
several units. The relevance term is one cosine, below 1. At α = 0.5 the early picks
are still decided by coverage, and relevance mostly breaks ties late in the greedy
loop. Rescaling coverage by 1/n, or sweeping α, is needed before any claim about a
"balanced" trade-off.

## Threats to validity

- **n = 18.** One question is 5.6 recall points. A difference of one question is not
  evidence of anything.
- **Duplicate ids inflate precision.** The index stores 320 vectors, but `retrieve`
  resolves each hit to a passage by id, and 23 ids are shared. A top-5 can therefore
  contain the same id twice, |R| falls below 5, and set precision goes up. In week 2
  the `top_k` precision sum is 2.9333 against 14 × 0.2 = 2.8, which pins it to exactly
  one hit question returning three distinct ids instead of five. The `mmr` and
  `facility_location` sums match their hit counts exactly.
- **One gold per question.** Precision saturates at 0.2 by construction. A method that
  pads the gold chunk with four genuinely useful neighbours scores the same as one that
  pads it with noise. Under this annotation precision@5 is recall in disguise.
- **Judge equals generator.** Hallucination is `gemini-3.6-flash` grading
  `gemini-3.6-flash`. Expect leniency toward its own phrasing, and expect the rate to be
  a lower bound.
- **Generation is not reproducible.** Retrieval is deterministic across runs; answer
  text is not. Each run overwrites the result file in place.

## Running

In order:

1. `pip install -r requirements.txt`
2. `python retriever/build_index.py`. Writes `retriever/passage_index.faiss` and
   `retriever/id_map.json`. `retriever.retrieve` loads both at import and raises with
   this instruction if they are missing.
3. Put `GOOGLE_API_KEY` in `.env` (template in `.env.example`). Every question costs at
   least one Gemini call.
4. Launch from the repository root. `retriever/retrieve.py` opens
   `data/code_passages.jsonl` and the index by relative path, and
   `run_week3_comparison.py` does the same for its own input and output.

```bash
python scripts/run_baseline.py
```

```bash
python scripts/run_comparison.py
```

```bash
python -m scripts.run_week3_comparison
```

The first two also run as `python -m scripts.run_baseline` and
`python -m scripts.run_comparison`. The week 3 script runs *only* in the `-m` form:
unlike the other two it does not insert the project root into `sys.path`, so
`python scripts/run_week3_comparison.py` dies on the first `from retriever ...` line.

**API cost.** Week 1: 18 generation calls. Week 2: 54, three answers per question.
Week 3: 72 generation calls, up to 72 more when the citation line is missing and the
generator retries, and 72 hallucination-judge calls; 144 to 216 requests per run.

**No checkpointing.** None of the scripts wrap the Gemini call in `try`/`except` or
write partial output. A failure on question 17 loses questions 1 through 16.
`run_baseline.py` once had `argparse` flags (`--data`, `--output`, `--k`,
`--checkpoint-every`, `--log-level`) and mid-run checkpoints; all of it was removed in
favour of module constants, and any document that still mentions those flags is
describing a version that no longer exists.

**Configuration** is module-level constants: `K` / `TOP_K`,
`CANDIDATE_SIZE` / `NUM_CANDIDATES`, `MODEL_NAME` / `EMBEDDING_MODEL`, the input and
output paths. No script accepts command-line arguments.

## Output format

`code_baseline_week1.json`:

```json
{"results": {"precision@k": 0.0, "recall@k": 0.0}, "predictions": ["..."]}
```

`predictions` holds 18 strings in dataset order (`q1` … `q18`).

`week2_comparison.json` adds a `config` block (embedding model, k, candidate size,
question count, method list) so the file describes its own run. The script types it
with `ConfigDict` / `ExperimentResultDict` and writes through `save_results()`:

```json
{
  "config": {"model": "all-mpnet-base-v2", "k": 5, "candidate_size": 15, "num_questions": 18, "methods": ["top_k", "mmr", "facility_location"]},
  "comparison": {"top_k": {"precision@k": 0.0, "recall@k": 0.0}, "mmr": {}, "facility_location": {}},
  "predictions": {"top_k": ["..."], "mmr": ["..."], "facility_location": ["..."]}
}
```

`week3_full_comparison.json` (planned) carries `comparison` and `predictions` only, no
`config` block; each method's metrics dict gains `citation_precision` and
`hallucination_rate`. None of the three files persist the retrieved ids, and week 3
does not persist the cited ids either, so per-question analysis means a re-run.
`python retriever/retrieve.py` prints the per-question hit/miss breakdown for the top-k
baseline without spending a single Gemini call.

## Implementation notes

- `run_comparison.py` and `run_week3_comparison.py` each construct their own
  `SentenceTransformer`, while importing `retriever.retrieve` has already loaded one at
  module level. Two copies of the encoder sit in memory. Per question the 15 candidate
  texts are encoded by `mmr_select`, again by `build_similarity_graph`, and in week 3 a
  third time by `compute_query_similarities`. Correct, just wasteful.
- The cross-encoder in `selection/reranker.py` is loaded lazily on the first
  `rerank_select` call, so week 3 pays that start-up cost inside the loop, on question
  1.
- `run_week3_comparison.py` reads its JSONL without skipping blank lines, unlike the
  `load_data` helpers in the other two scripts. Fine for the current file; a trailing
  empty line would break it.
- The docstring in this folder's `__init__.py` describes weeks 1 and 2 only.
  `run_week3_comparison` is not listed there yet.
