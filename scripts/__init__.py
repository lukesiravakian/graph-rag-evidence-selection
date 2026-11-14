"""Runnable entry points for the pipeline.

Modules here are executables, not a library: each one is meant to be run as
``python -m scripts.<name>`` (or ``python scripts/<name>.py``) from the
repository root. Running from anywhere else breaks retrieval — the scripts
resolve their own inputs and outputs against the project root, but
``retriever.retrieve`` opens ``data/code_passages.jsonl`` and
``retriever/passage_index.faiss`` relative to the working directory.

Nothing is exported, so ``import scripts`` has no side effects. Importing a
submodule is a different matter: every one except ``run_significance_tests``
pulls in ``retriever.retrieve`` at module level, which loads the embedding model
and the FAISS index right away, and so needs ``python retriever/build_index.py``
to have been run once.

All entry points read the 18 code questions in ``data/code_questions.jsonl`` and
retrieve under a budget of ``k = 5`` (pool ``n = 15`` for everything but plain
top-k). Answer *correctness* is never scored; the generated answers are saved
but never graded. Configuration is module constants; none takes command-line
flags.

Available entry points
----------------------
run_baseline
    The week 1 baseline: plain dense top-``k`` retrieval, no evidence selection.
    Scores ``precision@k`` / ``recall@k``.
    Writes ``results/code_baseline_week1.json``. 18 Gemini calls.
run_comparison
    The week 2 comparison: top-``k`` vs. MMR vs. coverage-only facility location
    (``alpha = 0``) over a shared 15-passage candidate pool, with its own answer
    set and score per method.
    Writes ``results/week2_comparison.json``. 54 Gemini calls.
run_week3_comparison
    The main comparison of the paper: top-``k`` vs. MMR vs. relevance-weighted
    facility location (``alpha = 0.5``) vs. cross-encoder reranking, scored on
    all four metrics — ``precision@k``, ``recall@k``, citation precision, and the
    LLM-judged hallucination rate.
    Writes ``results/week3_full_comparison.json``. 144 Gemini calls (18 questions
    x 4 methods, x2 because every answer is also judged).
run_structural_analysis
    Density, clustering coefficient, and average pairwise similarity of each set
    picked by weighted facility location, against hit/miss.
    Writes ``results/structural_analysis.json``. No Gemini calls.
run_alpha_instrumentation
    Per-step values of the two terms of Algorithm 1, showing that the coverage
    term dominates the early picks despite the nominal ``alpha = 0.5``.
    Writes ``results/alpha_instrumentation.json``. No Gemini calls.
run_miss_analysis
    Per question, whether the gold passage reached the ``n = 15`` pool and which
    methods selected it — which separates retrieval failures from selection
    failures and gives the recall ceiling for the pool-based methods.
    Writes ``results/miss_analysis.json``. No Gemini calls.
run_alpha_sweep
    Sweeps the facility-location weight ``alpha`` and the MMR weight ``lambda``
    over [0, 1], and reports the pool-recall ceiling and the expected recall of
    uniform random selection.
    Writes ``results/alpha_sweep.json``. No Gemini calls.
run_significance_tests
    Wilson intervals, exact McNemar bounds, and the Fisher exact test on the
    density table, computed from the saved result files.
    Writes ``results/significance_tests.json``. No model, no index, no API key.
make_figures
    Rebuilds the paper figures from the files in ``results/``. Writes into
    ``paper/figures/``; needs matplotlib, no model and no API key.
"""

from __future__ import annotations

__all__: list[str] = []
