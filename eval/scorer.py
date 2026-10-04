from generator.generate import client

def retrieval_metrics(retrieved_ids: list[str], gold_ids: list[str]) -> dict[str, float]:
    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)
    hits = retrieved_set & gold_set

    precision = len(hits) / len(retrieved_set) if retrieved_set else 0
    recall = len(hits) / len(gold_set) if gold_set else 0

    return {"precision": precision, "recall": recall}


def evaluate(
    all_retrieved_ids: list[list[str]], 
    all_gold_ids: list[list[str]]
) -> dict[str, float]:
    results = [
        retrieval_metrics(r, g)
        for r, g in zip(all_retrieved_ids, all_gold_ids) if r is not None and g is not None and len(all_retrieved_ids) > 0 and len(all_gold_ids) > 0
    ]

    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
   
    return {"precision@k": avg_precision, "recall@k": avg_recall}


def compare_methods(
    results_by_method: dict[
        str,
        tuple[list[list[str]], list[list[str]]]
    ]
) -> dict[str, dict[str, float]]:   
    """    results_by_method: dict like {"top_k": (all_retrieved_ids, all_gold_ids), "mmr": (...), "facility_location": (...)}    Returns a dict of method_name -> {precision@k, recall@k}    """  
    comparison = {}    
    for method_name, (retrieved, golds) in results_by_method.items():  
        comparison[method_name] = evaluate(retrieved, golds)
    return comparison


def citation_precision(cited_ids, provided_ids):
    """Of the passages the model claimed to use, how many were actually real, provided passages?"""
    if not cited_ids:
        return 0.0
    valid = [c for c in cited_ids if c in provided_ids]
    return len(valid) / len(cited_ids)


def check_hallucination(question, answer, passages):
    context = "\n\n".join(
        f"# {passage.get('title', 'Passage')}\n{passage['text']}"
        for passage in passages
    )
    prompt = f"""Given this context:
{context}

Answer the question "{question}":
{answer}

Does the answer contain any claims not supported by the context above? Reply with only YES or NO."""

    chat = client.chats.create(model="gemini-3.6-flash")
    response = chat.send_message(prompt)
    return response.text.strip().upper().startswith("YES")

if __name__ == "__main__":
    question = "How does requests decide whether to verify SSL certificates?"
    answer = "When verify is None, requests uses the session's verify setting."
    passages = [
        {
            "title": "sessions.py::Session.merge_environment_settings",
            "text": """def merge_environment_settings(self, url, proxies, stream, verify, cert):
    if verify is None:
        verify = self.verify
    return {'verify': verify}""",
        }
    ]

    result = check_hallucination(question, answer, passages)
    print(f"Hallucination detected: {result}")