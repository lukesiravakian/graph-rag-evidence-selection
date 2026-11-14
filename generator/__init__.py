"""Answer generation: turn a question plus its evidence into an answer.

Public API
----------
build_prompt(question, passages) -> str
    Render the plain grounded-QA prompt (week 1 and week 2 runs).
generate_answer(question, passages) -> str
    Answer ``question`` using only ``passages``. No citations.
build_prompt_with_citations(question, passages) -> str
    Render the citation-aware prompt: numbered snippets, a passage-ID key, and a
    required trailing ``CITATIONS:`` line.
generate_answer_with_citations(question, passages) -> (str, list[str])
    Answer ``question`` and parse the cited passage IDs. IDs that do not name one
    of the provided passages are discarded; if that leaves no citation, the model
    is prompted once more and whatever comes back is used. This is what produces
    every week 3 result.

Names are re-exported lazily (PEP 562). This matters here: ``generate``
constructs the Gemini client at import time and needs ``GOOGLE_API_KEY``,
so plain ``import generator`` must not pull it in.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "build_prompt",
    "build_prompt_with_citations",
    "generate_answer",
    "generate_answer_with_citations",
]

_LAZY_ATTRS = {
    "build_prompt": "generate",
    "build_prompt_with_citations": "generate",
    "generate_answer": "generate",
    "generate_answer_with_citations": "generate",
}

if TYPE_CHECKING:
    from generator.generate import (
        build_prompt,
        build_prompt_with_citations,
        generate_answer,
        generate_answer_with_citations,
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
