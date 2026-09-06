import json
from sentence_transformers import SentenceTransformer

## If having errors on this, please comment/delete these 5 lines.
from pathlib import Path
from sys import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in path:
    path.insert(0, str(PROJECT_ROOT))


from selection.graph_builder import build_similarity_graph
from selection.mmr import mmr_select
from selection.facility_location import facility_location_select
from generator.generate import generate_answer
from eval.scorer import evaluate, compare_methods

from retriever.retrieve import retrieve, retrieve_candidates


MODEL_NAME = "all-mpnet-base-v2"
ROOT_FILE = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_FILE / "results" / "week2_comparison.json"
INPUT_PATH = ROOT_FILE / "data" / "code_questions.jsonl"
MODE_OUTPUT = "w"

model = SentenceTransformer(MODEL_NAME)

data = [json.loads(line) for line in open(INPUT_PATH)]
results_by_method = {
    "top_k": ([], []),
    "mmr": ([], []),
    "facility_location": ([], []),
}

predictions_by_method = {"top_k": [], "mmr": [], "facility_location": []}

for record in data[:5]:
    question = record["question"]
    gold_ids = record["gold_passage_ids"]

    # Method 1: plain top-k baseline
    top_k_passages = retrieve(question, k=5)
    top_k_ids = [p["passage_id"] for p in top_k_passages]

    # Get a shared larger candidate pool for the other two methods
    candidates = retrieve_candidates(question, n=15)    # Method 2: MMR
    mmr_passages = mmr_select(model, question, candidates, k=5)
    mmr_ids = [p["passage_id"] for p in mmr_passages]

    # Method 3: facility location
    sim_matrix, _ = build_similarity_graph(candidates, model)
    fl_idx = facility_location_select(sim_matrix, k=5)
    fl_passages = [candidates[i] for i in fl_idx]
    fl_ids = [p["passage_id"] for p in fl_passages]
    for method, passages, ids in [
        ("top_k", top_k_passages, top_k_ids),
        ("mmr", mmr_passages, mmr_ids),        
        ("facility_location", fl_passages, fl_ids),    
    ]:
        results_by_method[method][0].append(ids)
        results_by_method[method][1].append(gold_ids)
        answer = generate_answer(question, passages)
        predictions_by_method[method].append(answer)

comparison = compare_methods(results_by_method)

print(comparison)

with open(OUTPUT_PATH, MODE_OUTPUT) as f:
    json.dump(
        {"comparison": comparison, "predictions": predictions_by_method}, f, indent=2
    )
