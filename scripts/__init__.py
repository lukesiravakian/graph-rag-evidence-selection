"""Runnable entry points for the pipeline.

Modules here are executables, not a library: each one is meant to be run from
the repository root. ``run_baseline`` and ``run_comparison`` insert the project
root into ``sys.path`` themselves and so accept both ``python -m scripts.<name>``
and ``python scripts/<name>.py``; ``run_week3_comparison`` does not, and only the
``-m`` form works for it. Running from any other directory breaks retrieval in
every case, because ``retriever.retrieve`` opens ``data/code_passages.jsonl`` and
``retriever/passage_index.faiss`` relative to the working directory (and the week
3 script resolves its own input and output the same way).

Nothing is exported, so ``import scripts`` has no side effects. Importing a
submodule is a different matter: each pulls in ``retriever.retrieve`` at module
level, which loads the embedding model and the FAISS index right away, and so
needs ``python retriever/build_index.py`` to have been run once.

All three entry points read the 18 code questions in ``data/code_questions.jsonl``,
hand ``k = 5`` passages to Gemini for every question, and macro-average
``precision@k`` / ``recall@k`` over the retrieved ids. The generated answers are
saved but never graded for correctness. Configuration is module constants; no
script takes command-line flags.

Available entry points
----------------------
run_baseline
    The week 1 baseline: plain dense top-``k`` retrieval, no evidence selection.
    Writes ``results/code_baseline_week1.json``.
run_comparison
    The week 2 comparison: top-``k`` vs. MMR vs. facility location over a shared
    15-passage candidate pool, with its own answer set and score per method.
    Writes ``results/week2_comparison.json``, including a ``config`` block that
    records the run parameters.
run_week3_comparison
    The week 3 comparison: top-``k``, MMR, relevance-weighted facility location
    (``alpha = 0.5``) and cross-encoder reranking over the same pool. Answers are
    generated with ``generate_answer_with_citations``; besides the retrieval
    metrics, each method is scored for ``citation_precision`` and an LLM-judged
    ``hallucination_rate``. Depends on ``selection.reranker`` and on the citation
    and hallucination helpers in ``generator.generate`` / ``eval.scorer``.
    Writes ``results/week3_full_comparison.json``.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__: list[str] = []

# The runnable submodules, in the order they were added. 
# ``__dir__`` lists them and ``__getattr__`` imports them on demand; 
# nothing here is loaded eagerly.
ENTRY_POINTS: tuple[str, ...] = (
    "run_baseline",
    "run_comparison",
    "run_week3_comparison",
)

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from scripts import run_baseline, run_comparison, run_week3_comparison


def __getattr__(name: str) -> ModuleType:
    """Resolve ``scripts.<entry point>`` lazily (PEP 562).

    Only the names in ``ENTRY_POINTS`` are honoured. Importing one of them pulls
    in ``retriever.retrieve`` and therefore the embedding model and FAISS index,
    exactly as ``import scripts.<name>`` would; the import machinery binds the
    submodule onto this package, so this hook runs at most once per name.
    """
    if name not in ENTRY_POINTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return import_module(f".{name}", __name__)


def __dir__() -> list[str]:
    return sorted([*__all__, *ENTRY_POINTS])
