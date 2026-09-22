from sentence_transformers import CrossEncoder
from numpy import float32
from numpy.typing import NDArray
from typing import Any

_reranker = None


def get_reranker() -> NDArray[float32]:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank_select(
    query: str, 
    passages: list[dict[str, Any]], 
    k: int = 5
) -> list[dict[str, Any]]:
    reranker = get_reranker()
    pairs = [(query, p["text"]) for p in passages]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [p for p, _ in ranked[:k]]