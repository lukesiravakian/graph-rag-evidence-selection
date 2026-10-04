import json
import numpy as np
from sentence_transformers import SentenceTransformer
from retriever.retrieve import retrieve_candidates
from selection.graph_builder import build_similarity_graph, compute_query_similarities
from selection.facility_location import relevance_weighted_facility_location_select
from selection.structural_analysis import compute_subgraph_properties

model = SentenceTransformer("all-mpnet-base-v2")
data = [json.loads(l) for l in open("data/code_questions.jsonl")]

rows = []

for record in data:
    question = record["question"]
    gold_ids = record["gold_passage_ids"]

    candidates = retrieve_candidates(question, n=15)
    sim_matrix, _ = build_similarity_graph(candidates, model)
    query_sims = compute_query_similarities(question, candidates, model)

    selected_idx = relevance_weighted_facility_location_select(sim_matrix, query_sims, k=5)
    selected_ids = [candidates[i]["passage_id"] for i in selected_idx]

    hit = 1 if any(g in selected_ids for g in gold_ids) else 0
    props = compute_subgraph_properties(sim_matrix, selected_idx)

    rows.append({
        "qid": record["qid"],
        "density": props["density"],
        "clustering_coefficient": props["clustering_coefficient"],
        "avg_pairwise_similarity": props["avg_pairwise_similarity"],
        "hit": hit,
    })

with open("results/structural_analysis.json", "w") as f:
    json.dump(rows, f, indent=2)

print(f"Wrote structural analysis for {len(rows)} questions.")

hits = [r for r in rows if r["hit"] == 1]
misses = [r for r in rows if r["hit"] == 0]
if hits:
    print(f"Hits ({len(hits)}): avg pairwise similarity = {np.mean([r['avg_pairwise_similarity'] for r in hits]):.3f}")
if misses:
    print(f"Misses ({len(misses)}): avg pairwise similarity = {np.mean([r['avg_pairwise_similarity'] for r in misses]):.3f}")