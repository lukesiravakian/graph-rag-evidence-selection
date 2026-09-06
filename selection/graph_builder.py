import numpy as np
from sentence_transformers import SentenceTransformer

def build_similarity_graph(
    passages: list[dict[str, str]],
    model: SentenceTransformer,
) -> tuple[np.ndarray, np.ndarray]:
       """
       passages: list of passage dicts (each with a 'text' field)
       model: a loaded SentenceTransformer model
       Returns: (similarity_matrix, embeddings)
       similarity_matrix[i][j] = cosine similarity between passage i and passage j
       """
       texts = [p["text"] for p in passages]
       embeddings = np.array(model.encode(texts))
       norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
       normalized = embeddings / norms
       similarity_matrix = normalized @ normalized.T
       return similarity_matrix, embeddings


if __name__ == "__main__":
       import json

       model = SentenceTransformer("all-mpnet-base-v2")
       passages = [json.loads(l) for l in open("data/code_passages.jsonl")][:10]
       sim_matrix, embeddings = build_similarity_graph(passages, model)

       print("Matrix shape:", sim_matrix.shape)
       print("Diagonal values (should all be ~1.0):", sim_matrix.diagonal())
       print("Symmetric check (should be True):", np.allclose(sim_matrix, sim_matrix.T))