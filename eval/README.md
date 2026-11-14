# Eval

The four metrics the paper reports, one function each. Nothing here grades answer
*correctness* — these measure retrieval and grounding only.

> **Name collision.** This package shadows the built-in `eval` function inside any
> module that does `import eval`. Prefer `from eval.scorer import evaluate`.

## Metrics

### `retrieval_metrics(retrieved_ids, gold_ids)` → `{precision, recall}`

Per-question precision and recall over **sets** of passage IDs:

```
precision = |retrieved ∩ gold| / |retrieved|
recall    = |retrieved ∩ gold| / |gold|
```

Either measure is `0` when its denominator is `0`. Because the inputs are converted to
sets, a repeated ID counts once — which matters, since only 280 of the corpus's 320
`passage_id`s are distinct. A retrieval that returns the same ID twice has
`|retrieved| = 4`, and its precision is correspondingly higher. See `results/README.md`
for the effect this has on the reported top-k precision.

With exactly one gold passage per question and `k = 5`, per-question recall is 0 or 1,
so `recall@k × 18` is the number of questions hit, and precision is capped at 0.2.

### `evaluate(all_retrieved_ids, all_gold_ids)` → `{precision@k, recall@k}`

Arithmetic mean of the per-question metrics — macro-averaged, not pooled. Returns
zeros rather than raising when given nothing to score.

### `citation_precision(cited_ids, provided_ids)` → `float`

Share of an answer's parsed citations that name a passage the generator was actually
given; `0.0` when there are no citations.

**Read this one with care.** `generator.generate_answer_with_citations` already drops
IDs that do not match a provided passage, so by the time the scorer sees a list, every
entry is valid by construction and the result can only be `0.0` or `1.0`. In practice
this reports *whether an answer cited anything acceptable at all*. Measuring whether
the model would fabricate an ID requires the unfiltered list from the generator.

### `check_hallucination(question, answer, passages)` → `bool`

LLM-as-judge. Builds a context block from the provided passages, asks
`gemini-3.6-flash` whether the answer contains claims the context does not support,
and returns `True` when the reply begins with `YES` (after `.strip().upper()`).

Caveats, all of which the paper states as limitations:

- **Not validated against human judgments.** This is a rate of *flagged* answers, not
  a measured rate of factual errors.
- **Not pinned to a temperature.** Unlike the generator, the judge uses
  `client.chats.create(...)` with the API's default sampling, so it is less
  reproducible than everything else in the pipeline.
- **One call per answer**, which is why `run_week3_comparison.py` costs 144 Gemini
  calls rather than 72.

## Import cost

`eval/scorer.py` imports the Gemini `client` from `generator.generate`, so importing
it **requires `GOOGLE_API_KEY`** even if you only want `retrieval_metrics`.
`eval/__init__.py` re-exports lazily so `import eval` stays free, and the
analysis-only scripts (`run_structural_analysis.py`, `run_alpha_instrumentation.py`,
`run_significance_tests.py`) avoid importing the scorer for this reason.

## Standalone check

```bash
python eval/scorer.py
```

Runs `check_hallucination` on one hand-written question/answer/passage triple where
the answer *is* supported, so the expected output is `Hallucination detected: False`.
Costs one real API call.
