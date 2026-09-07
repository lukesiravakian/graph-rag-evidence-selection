"""Evidence selection: pick a diverse, non-redundant subset of candidates.

Each strategy takes the candidate pool a retriever produced and returns the
``k`` passages to hand to the generator. Unlike plain top-``k``, they weigh how
much a passage adds on top of what is already selected.

Public API
----------
build_similarity_graph(passages, model) -> (similarity_matrix, embeddings)
    Full passage x passage cosine similarity matrix over the candidates.
mmr_select(model, query, passages, k=5, lambda_param=0.5) -> list[dict]
    Maximal marginal relevance: trade query relevance off against the largest
    similarity to anything already picked. Returns passage dicts.
facility_location_select(similarity_matrix, k=5) -> list[int]
    Greedy facility location over a similarity graph built by
    ``build_similarity_graph``. Returns indices into the candidate list, not
    passages, and never looks at the query.

Names are re-exported lazily (PEP 562). That matters here: ``graph_builder``
and ``mmr`` import ``sentence_transformers`` (and so torch) at module level,
while ``facility_location`` needs only numpy — so ``import selection`` stays
cheap and pulls in a model stack only for the strategy actually used.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["build_similarity_graph", "facility_location_select", "mmr_select"]

# public name -> submodule that defines it
_LAZY_ATTRS = {
    "build_similarity_graph": "graph_builder",
    "facility_location_select": "facility_location",
    "mmr_select": "mmr",
}

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from selection.facility_location import facility_location_select
    from selection.graph_builder import build_similarity_graph
    from selection.mmr import mmr_select


def __getattr__(name: str) -> Any:
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value  # cache: this hook runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
