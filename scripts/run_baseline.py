import json

from retriever.retrieve import retrieve
from generator.generate import generate_answer
from eval.scorer import evaluate


data = [json.loads(l) for l in open("data/hotpotqa_dev_300.jsonl")]

predictions, golds = [], []

for record in data:
    passages = retrieve(record["question"], k=5) 
    answer = generate_answer(record["question"], passages) 
    predictions.append(answer)
    golds.append(record["answer"])


results = evaluate(predictions, golds)
print(results)

with open("results/baseline_week1.json", "w") as f:
    json.dump({
        "results": results, 
        "predictions": predictions
    }, f, indent=2)