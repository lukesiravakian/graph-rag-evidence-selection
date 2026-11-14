"""Dense retrieval: the FAISS index and the two ways the pipeline reads from it.

Public API
----------
retrieve(question, k=5) -> list[dict]
    The top-``k`` passages for ``question`` by cosine similarity. With ``k = 5``
    this is the plain top-k baseline.
retrieve_candidates(question, n=15) -> list[dict]
    The ``n``-passage candidate pool that MMR, facility location, and reranking
    all select from. A thin alias for ``retrieve(question, k=n)``, kept separate
    so the pool size and the final budget stay distinguishable at call sites.

Names are re-exported lazily (PEP 562), so ``import retriever`` stays cheap
and side-effect free; submodules load on first attribute access. ``retriever.retrieve``
loads the embedding model and the FAISS index at import time, so
``python retriever/build_index.py`` must have been run once.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["retrieve", "retrieve_candidates"]

_LAZY_ATTRS = {"retrieve": "retrieve", "retrieve_candidates": "retrieve"}

if TYPE_CHECKING:
    from retriever.retrieve import retrieve, retrieve_candidates


def __getattr__(name: str) -> Any:
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
