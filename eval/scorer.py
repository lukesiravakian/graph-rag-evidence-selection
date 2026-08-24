





def retrieval_metrics(retrieved_ids, gold_ids):
    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)
    hits = retrieved_set & gold_set

    precision = len(hits) / len(retrieved_set) if retrieved_set else 0
    recall = len(hits) / len(gold_set) if gold_set else 0

    return {"precision": precision, "recall": recall}


def evaluate(all_retrieved_ids, all_gold_ids):
    results = [
        retrieval_metrics(r, g)
        for r, g in zip(all_retrieved_ids, all_gold_ids) if r is not None and g is not None and len(all_retrieved_ids) > 0 and len(all_gold_ids) > 0
    ]

    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
   
    return {"precision@k": avg_precision, "recall@k": avg_recall}



