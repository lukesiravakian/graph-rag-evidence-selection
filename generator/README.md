# Generator

Answer generation with Gemini. Takes a question plus the `k = 5` passages a selection
method chose and returns an answer — optionally with the passage IDs the model claims
to have used.

## Two prompts, two eras

| Function | Returns | Used by |
| --- | --- | --- |
| `generate_answer(question, passages)` | answer string | `run_baseline.py`, `run_comparison.py` |
| `generate_answer_with_citations(question, passages)` | `(answer, cited_ids)` | `run_week3_comparison.py` |

Both call `gemini-3.6-flash` at `temperature=0.1` with `max_output_tokens=1000`.
`generate_answer` adds a system instruction asking for a 1–2 sentence answer that
names the relevant functions or classes. `generate_answer_with_citations` has no length
instruction; instead it numbers the snippets, supplies a **passage ID key** mapping each
number to its full `passage_id`, and requires a trailing `CITATIONS:` line.

## Citation parsing, and what it means for the metric

`generate_answer_with_citations` splits the response on `CITATIONS:`, takes the first
line after it, splits on commas, and **keeps only IDs that are in the provided set**.
If that filter leaves nothing, the prompt is extended with an explicit request for a
valid ID and the model is called once more; whatever comes back on that second attempt
is used regardless of outcome (at most two calls per answer).

The consequence is worth stating plainly: **the scorer never sees an invalid ID**, so
`eval.scorer.citation_precision` cannot register one. In practice it reports whether an
answer included at least one accepted citation, not whether the model was capable of
fabricating an ID. Measuring fabrication would mean returning the unfiltered list.

Note also that the snippet headers show each passage's `title` (`cookies.py::foo`)
while the ID key shows the full `passage_id`
(`data/source_repo/src/requests/cookies.py::foo`) — the model must cite the latter.

## Setup

```bash
cp .env.example .env   # then set GOOGLE_API_KEY
```

`generate.py` calls `load_dotenv()` and constructs `genai.Client()` **at import time**,
so importing this module without a key fails. That is why `generator/__init__.py`
re-exports lazily, and why `eval.scorer` — which imports the same client — is kept out
of the analysis-only scripts.

## Behaviour worth knowing

- **No retry on API errors.** Only the citation retry exists; a transport error
  propagates and loses the run, since no script checkpoints.
- **`generate_answer` has a fallback path.** If `response.text` is empty it walks
  `response.candidates[0].content.parts` for text, and returns the literal string
  `"No answer generated."` if that also fails. `generate_answer_with_citations` has no
  such fallback.
- **Answers are not deterministic** even at `temperature=0.1`, so re-runs are not
  byte-identical and overwrite `results/` in place.

## Standalone check

```bash
python generator/generate.py
```

Runs `generate_answer_with_citations` on the first three questions using their *gold*
passages (so retrieval is out of the loop), or on a single dummy passage if the data
files are missing. Costs 1–3 real API calls.
