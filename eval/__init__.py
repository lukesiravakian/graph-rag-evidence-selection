"""Scoring: the four metrics the paper reports for each selection method.

Public API
----------
retrieval_metrics(retrieved_ids, gold_ids) -> dict
    Set-based precision and recall for one question.
evaluate(all_retrieved_ids, all_gold_ids) -> dict
    ``precision@k`` and ``recall@k``, macro-averaged over questions. Both inputs
    are lists of passage-ID lists, one per question — this compares *retrieved
    passage IDs* against gold passage IDs, not answer text against gold answers.
compare_methods(results_by_method) -> dict
    ``evaluate`` applied per selection method.
citation_precision(cited_ids, provided_ids) -> float
    Share of an answer's parsed citations that name a passage the generator was
    actually given. Note the generator already drops invalid IDs, so in practice
    this mostly reflects whether an answer cited anything acceptable at all.
check_hallucination(question, answer, passages) -> bool
    LLM-judged flag: True when the judge says the answer makes claims the
    provided passages do not support. Costs one Gemini call.

Note: this package shadows the built-in ``eval`` function inside any module
that does ``import eval``; prefer ``from eval.scorer import evaluate``.

Names are re-exported lazily (PEP 562), so ``import eval`` stays cheap and
side-effect free; submodules load on first attribute access. ``eval.scorer``
constructs the Gemini client at import time (it shares one with ``generator``)
and so needs ``GOOGLE_API_KEY``, which is the other reason not to import it
eagerly here.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "check_hallucination",
    "citation_precision",
    "compare_methods",
    "evaluate",
    "retrieval_metrics",
]

_LAZY_ATTRS = {
    "check_hallucination": "scorer",
    "citation_precision": "scorer",
    "compare_methods": "scorer",
    "evaluate": "scorer",
    "retrieval_metrics": "scorer",
}

if TYPE_CHECKING:
    from eval.scorer import (
        check_hallucination,
        citation_precision,
        compare_methods,
        evaluate,
        retrieval_metrics,
    )


def __getattr__(name: str) -> Any:
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
