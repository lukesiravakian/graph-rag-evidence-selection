"""Evidence retrieval: select candidate passages for a question.

Public API
----------
retrieve(question, k=5) -> list[dict]
    Return the top-``k`` passages for ``question``.

Names are re-exported lazily (PEP 562), so ``import retriever`` stays cheap
and side-effect free; submodules load on first attribute access.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["retrieve"]

_LAZY_ATTRS = {"retrieve": "retrieve"}

if TYPE_CHECKING:
    from retriever.retrieve import retrieve


def __getattr__(name: str) -> Any:
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
