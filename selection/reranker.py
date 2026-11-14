from sentence_transformers import CrossEncoder
from typing import Any

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Loads the cross-encoder once and reuses it across calls."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank_select(
    query: str,
    passages: list[dict[str, Any]],
    k: int = 5,
) -> list[dict[str, Any]]:
    """
    Scores every (query, passage) pair jointly with the cross-encoder and keeps
    the k highest-scoring passages.

    query: the question
    passages: candidate pool, each a dict with at least a 'text' key
    k: number of passages to keep
    Returns: the k highest-scoring passages, best first
    """
    reranker = get_reranker()
    pairs = [(query, p["text"]) for p in passages]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:k]]
