"""Scoring: compare predicted answers against HotpotQA gold answers.

Public API
----------
evaluate(predictions, golds) -> dict
    Aggregate metrics over a full run.

Note: this package shadows the built-in ``eval`` function inside any module
that does ``import eval``; prefer ``from eval.scorer import evaluate``.

Names are re-exported lazily (PEP 562), so ``import eval`` stays cheap and
side-effect free; submodules load on first attribute access.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["evaluate"]

# public name -> submodule that defines it
_LAZY_ATTRS = {"evaluate": "scorer"}

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from eval.scorer import evaluate


def __getattr__(name: str) -> Any:
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value  # cache: this hook runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
