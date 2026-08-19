"""Answer generation: turn a question plus its evidence into an answer.

Public API
----------
build_prompt(question, passages) -> str
    Render the grounded-QA prompt sent to the model.
generate_answer(question, passages) -> str
    Answer ``question`` using only ``passages``.

Names are re-exported lazily (PEP 562). This matters here: ``generate``
constructs the Gemini client at import time and needs ``GOOGLE_API_KEY``,
so plain ``import generator`` must not pull it in.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["build_prompt", "generate_answer"]

# public name -> submodule that defines it
_LAZY_ATTRS = {"build_prompt": "generate", "generate_answer": "generate"}

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from generator.generate import build_prompt, generate_answer


def __getattr__(name: str) -> Any:
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value  # cache: this hook runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
