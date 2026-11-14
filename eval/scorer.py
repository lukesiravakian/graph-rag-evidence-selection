from typing import Any

from generator.generate import client

JUDGE_MODEL = "gemini-3.6-flash"


def retrieval_metrics(retrieved_ids: list[str], gold_ids: list[str]) -> dict[str, float]:
    """
    Precision@k and recall@k for one question, over sets of passage IDs, so a
    repeated ID counts once. Either measure is 0 when its denominator is 0.
    """
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
    """Arithmetic mean of the per-question metrics over every scored question."""
    results = [
        retrieval_metrics(retrieved, gold)
        for retrieved, gold in zip(all_retrieved_ids, all_gold_ids)
        if retrieved is not None and gold is not None
    ]

    if not results:
        return {"precision@k": 0.0, "recall@k": 0.0}

    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)

    return {"precision@k": avg_precision, "recall@k": avg_recall}


def compare_methods(
    results_by_method: dict[
        str,
        tuple[list[list[str]], list[list[str]]]
    ]
) -> dict[str, dict[str, float]]:
    """
    results_by_method: dict like {"top_k": (all_retrieved_ids, all_gold_ids), "mmr": (...), ...}
    Returns: a dict of method_name -> {precision@k, recall@k}
    """
    comparison = {}
    for method_name, (retrieved, golds) in results_by_method.items():
        comparison[method_name] = evaluate(retrieved, golds)
    return comparison


def citation_precision(cited_ids: list[str], provided_ids: list[str]) -> float:
    """Of the passages the model claimed to use, how many were actually real, provided passages?"""
    if not cited_ids:
        return 0.0
    valid = [c for c in cited_ids if c in provided_ids]
    return len(valid) / len(cited_ids)


def check_hallucination(
    question: str,
    answer: str,
    passages: list[dict[str, Any]],
) -> bool:
    """
    LLM-judged hallucination flag: True when the judge's reply begins with "YES",
    meaning the answer makes claims the provided passages do not support.
    """
    context = "\n\n".join(
        f"# {passage.get('title', 'Passage')}\n{passage['text']}"
        for passage in passages
    )
    prompt = f"""Given this context:
{context}

Answer the question "{question}":
{answer}

Does the answer contain any claims not supported by the context above? Reply with only YES or NO."""

    chat = client.chats.create(model=JUDGE_MODEL)
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
