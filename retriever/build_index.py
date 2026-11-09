import numpy as np
import faiss

from json import loads, dump
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PASSAGES_PATH = PROJECT_ROOT / "data" / "code_passages.jsonl"
INDEX_PATH = PROJECT_ROOT / "retriever" / "passage_index.faiss"
ID_MAP_PATH = PROJECT_ROOT / "retriever" / "id_map.json"

def build(MODEL_NAME: str) -> None:
    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        passages = [loads(line) for line in f]
    texts = [p["text"] for p in passages]
    embeddings = SentenceTransformer(MODEL_NAME).encode(texts, show_progress_bar=True, batch_size=1)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        dump([p["passage_id"] for p in passages], f)
    
    print(f"Indexed {len(passages)} passages -> {INDEX_PATH}")
        
if __name__ == "__main__":
    build(
        "all-mpnet-base-v2"
    )