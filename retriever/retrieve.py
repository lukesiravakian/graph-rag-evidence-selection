import json
from sentence_transformers import SentenceTransformer
import faiss

PASSAGES_PATH = "data/code_passages.jsonl"
INDEX_PATH = "retriever/passage_index.faiss"
ID_MAP_PATH = "retriever/id_map.json"
MODEL_NAME = "all-mpnet-base-v2"


_BUILD_INDEX_MSG = (
    "\n\nretriever/retrieve.py couldn't find the FAISS index / id map / "
    "passages file.\nYou need to build the index once before importing "
    "retrieve():\n\n    python retriever/build_index.py\n"
)
try:
    _model = SentenceTransformer(MODEL_NAME)
    _index = faiss.read_index(INDEX_PATH)  
except RuntimeError as e:
    raise RuntimeError(_BUILD_INDEX_MSG + f"\n(details: {e})") from e

try:
    with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
        _id_map = json.load(f)

    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        _passage_lookup = {
            p["passage_id"]: p
            for p in (json.loads(line) for line in f if line.strip())
        }
        
except FileNotFoundError as e:
    raise RuntimeError(_BUILD_INDEX_MSG + f"\n(missing file: {e.filename})") from e


def retrieve(query, k=5):
    k = min(k, _index.ntotal)
    q_emb = _model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    _, idxs = _index.search(q_emb, k)
    return [_passage_lookup[_id_map[i]] for i in idxs[0] if i != -1]

def retrieve_candidates(query, n=15):
    return retrieve(query, k=n)


if __name__ == "__main__":
    QUESTIONS_PATH = "data/code_questions.jsonl"
    K = 5

    questions = [json.loads(l) for l in open(QUESTIONS_PATH)]
    hits = 0

    for q in questions:
        results = retrieve(q["question"], k=K)
        retrieved_ids = {r["passage_id"] for r in results}
        gold_ids = set(q["gold_passage_ids"])
        hit = bool(retrieved_ids & gold_ids)
        hits += hit

        status = "HIT " if hit else "MISS"
        print(f"[{status}] {q['qid']}: {q['question']}")
        if not hit:
            print(f"         gold:      {sorted(gold_ids)}")
            print(f"         retrieved: {sorted(retrieved_ids)}")

    print(f"\n{hits}/{len(questions)} questions had a gold passage in the top-{K}")
