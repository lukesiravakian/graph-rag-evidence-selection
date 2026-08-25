import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

PASSAGES_PATH = "data/code_passages.jsonl"
INDEX_PATH = "retriever/passage_index.faiss"
ID_MAP_PATH = "retriever/id_map.json"
MODEL_NAME = "all-mpnet-base-v2"

def build():
    model=SentenceTransformer(MODEL_NAME)

    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]
    texts = [p["text"] for p in passages]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=1)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index=faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump([p["passage_id"] for p in passages], f)
    
    print(f"Indexed {len(passages)} passages -> {INDEX_PATH}")
        
if __name__ == "__main__":
    build()
        
    
    