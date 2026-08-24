from datasets import load_dataset
import json

ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
subset = ds.select(range(300))

def flatten_passages(example, qid):
    passages = []
    titles = example["context"]["title"]
    sentences_list = example["context"]["sentences"]
    for i, (title, sents) in enumerate(zip(titles, sentences_list)):
        passages.append({
            "passage_id": f"{qid}_{i}",
            "title": title,
            "text": " ".join(sents)
        })
    return passages

records = []
for ex in subset:
    qid = ex["id"]
    records.append({
        "qid": qid,
        "question": ex["question"],
        "answer": ex["answer"],
        "supporting_titles": ex["supporting_facts"]["title"],
        "passages": flatten_passages(ex, qid)
    })

with open("data/hotpotqa_dev_300.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

all_passages = {}
for r in records:
    for p in r["passages"]:
        all_passages[p["passage_id"]] = p

with open("data/all_passages.jsonl", "w") as f:
    for p in all_passages.values():
        f.write(json.dumps(p) + "\n")

print(f"Wrote {len(records)} questions and {len(all_passages)} unique passages.")