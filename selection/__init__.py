"""Evidence selection: pick a jointly informative subset of the candidate pool.

Each strategy takes the candidate pool a retriever produced and returns the
``k`` passages to hand to the generator. Unlike plain top-``k``, they weigh how
much a passage adds on top of what is already selected.

Public API
----------
build_similarity_graph(passages, model) -> (similarity_matrix, embeddings)
    Full passage x passage cosine similarity matrix over the candidates.
compute_query_similarities(query, passages, model) -> ndarray
    Cosine similarity of every candidate to the query, i.e. the ``rel`` vector
    that the weighted selector below needs.
mmr_select(model, query, passages, k=5, lambda_param=0.5) -> list[dict]
    Maximal marginal relevance: trade query relevance off against the largest
    similarity to anything already picked. Returns passage dicts.
relevance_weighted_facility_location_select(similarity_matrix, query_similarities, k=5, alpha=0.5) -> list[int]
    The method proposed in the paper (Algorithm 1): greedy facility location
    with an explicit query-relevance term,
    ``score(c | S) = (1 - alpha) * coverage_gain(c) + alpha * rel(c)``.
    Returns indices into the candidate list, not passages.
relevance_weighted_facility_location_trace(similarity_matrix, query_similarities, k=5, alpha=0.5) -> (list[int], list[dict])
    Same selection, plus the per-step values of the two score terms. Used for the
    coverage-vs-relevance scale analysis in the paper.
facility_location_select(similarity_matrix, k=5) -> list[int]
    The coverage-only ablation: the ``alpha = 0`` case of the selector above, so
    it never looks at the query once the pool has been retrieved.
rerank_select(query, passages, k=5) -> list[dict]
    Cross-encoder reranking baseline: score every ``(query, passage)`` pair
    jointly and keep the best ``k``.
compute_subgraph_properties(similarity_matrix, selected_idx, threshold=0.5) -> dict
    Density, clustering coefficient, and average pairwise similarity of a
    selected set, for the structural analysis.

Names are re-exported lazily (PEP 562). That matters here: ``graph_builder``,
``mmr`` and ``reranker`` import ``sentence_transformers`` (and so torch) at
module level, while ``facility_location`` needs only numpy — so
``import selection`` stays cheap and pulls in a model stack only for the
strategy actually used.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "build_similarity_graph",
    "compute_query_similarities",
    "compute_subgraph_properties",
    "facility_location_select",
    "mmr_select",
    "relevance_weighted_facility_location_select",
    "relevance_weighted_facility_location_trace",
    "rerank_select",
]

_LAZY_ATTRS = {
    "build_similarity_graph": "graph_builder",
    "compute_query_similarities": "graph_builder",
    "compute_subgraph_properties": "structural_analysis",
    "facility_location_select": "facility_location",
    "mmr_select": "mmr",
    "relevance_weighted_facility_location_select": "facility_location",
    "relevance_weighted_facility_location_trace": "facility_location",
    "rerank_select": "reranker",
}

if TYPE_CHECKING:
    from selection.facility_location import (
        facility_location_select,
        relevance_weighted_facility_location_select,
        relevance_weighted_facility_location_trace,
    )
    from selection.graph_builder import build_similarity_graph, compute_query_similarities
    from selection.mmr import mmr_select
    from selection.reranker import rerank_select
    from selection.structural_analysis import compute_subgraph_properties


def __getattr__(name: str) -> Any:
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
