"""Runnable entry points for the pipeline.

Modules here are executables, not a library: each one is meant to be run as
``python -m scripts.<name>`` (or ``python scripts/<name>.py``) from the
repository root. Running from anywhere else breaks retrieval — the scripts
resolve their own inputs and outputs against the project root, but
``retriever.retrieve`` opens ``data/code_passages.jsonl`` and
``retriever/passage_index.faiss`` relative to the working directory.

Nothing is exported, so ``import scripts`` has no side effects. Importing a
submodule is a different matter: each pulls in ``retriever.retrieve`` at module
level, which loads the embedding model and the FAISS index right away, and so
needs ``python retriever/build_index.py`` to have been run once.

Both entry points read the 18 code questions in ``data/code_questions.jsonl``,
retrieve under a budget of ``k = 5``, answer each question with Gemini, and
score *retrieval* only (``precision@k`` / ``recall@k``) — the generated answers
are saved but never graded. Configuration is module constants; neither takes
command-line flags.

Available entry points
----------------------
run_baseline
    The week 1 baseline: plain dense top-``k`` retrieval, no evidence selection.
    Writes ``results/code_baseline_week1.json``.
run_comparison
    The week 2 comparison: top-``k`` vs. MMR vs. facility location over a shared
    15-passage candidate pool, with its own answer set and score per method.
    Writes ``results/week2_comparison.json``.
"""

from __future__ import annotations

__all__: list[str] = []
