import numpy as np
from sentence_transformers import SentenceTransformer

def mmr_select(
    model: SentenceTransformer,
    query: str,
    passages: list[dict[str, str]],
    k: int = 5,
    lambda_param: float = 0.5,
) -> list[dict[str, str]]:
    texts = [p["text"] for p in passages]
    embeddings = np.array(model.encode(texts))
    query_embedding = np.array(model.encode([query])[0])

    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    query_sims = [cos_sim(e, query_embedding) for e in embeddings]

    selected_idx = []
    candidate_idx = list(range(len(passages)))

    while len(selected_idx) < k and candidate_idx:
        if not selected_idx:
            best = max(candidate_idx, key=lambda i: query_sims[i])
        else:
            def mmr_score(i: int) -> float:
                redundancy = max(cos_sim(embeddings[i], embeddings[j]) for j in selected_idx)
                return lambda_param * query_sims[i] - (1 - lambda_param) * redundancy
            best = max(candidate_idx, key=mmr_score)

        selected_idx.append(best)
        candidate_idx.remove(best)

    return [passages[i] for i in selected_idx]