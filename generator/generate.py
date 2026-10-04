import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

def build_prompt_with_citations(question, passages):
    context = "\n\n".join(f"[{i+1}] {p['title']}\n{p['text']}" for i, p in enumerate(passages))
    id_key = "\n".join(f"[{i+1}] = {p['passage_id']}" for i, p in enumerate(passages))
    return f"""You are answering a question about a codebase using the numbered code snippets below.

Code context:
{context}

Passage ID key:
{id_key}

Question: {question}

Answer concisely. After your answer, on a new line, write "CITATIONS:" followed by the passage ID(s) you actually used, separated by commas.

You must include the CITATIONS line in every response. Use only exact passage IDs from the Passage ID key; do not use the snippet numbers, brackets, or any other text."""


def generate_answer_with_citations(question, passages):
    prompt = build_prompt_with_citations(question, passages)
    valid_ids = {passage["passage_id"] for passage in passages}

    for attempt in range(2):
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1000,
            ),
        )
        text = response.text.strip()

        if "CITATIONS:" in text:
            answer_part, citation_part = text.split("CITATIONS:", 1)
            citation_line = citation_part.strip().splitlines()[0] if citation_part.strip() else ""
            cited_ids = [
                citation.strip()
                for citation in citation_line.split(",")
                if citation.strip() in valid_ids
            ]
        else:
            answer_part = text
            cited_ids = []

        if cited_ids or attempt == 1:
            return answer_part.strip(), cited_ids

        prompt += (
            "\n\nYour previous response did not contain a valid citation. "
            "Rewrite the answer and end with `CITATIONS: ` followed by at least one "
            "exact passage ID from the Passage ID key."
        )

def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(f"# {p['title']}\n{p['text']}" for p in passages)
    return f"Code Context:\n{context}\n\nQuestion: {question}"

def generate_answer(question: str, passages: list[dict[str, str]]) -> str:
    prompt = build_prompt(question, passages)
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful coding assistant. Answer the user's question in 1-2 concise sentences based on the provided code context. Mention relevant function or class names.",
            temperature=0.1,
            max_output_tokens=1000,
        )
    )
    
    if getattr(response, "text", None):
        return response.text.strip()
    
    try:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if getattr(part, "text", None):
                    return part.text.strip()
    except (AttributeError, IndexError):
        pass
        
    return "No answer generated."

if __name__ == "__main__":
    passages_file = "data/code_passages.jsonl"
    questions_file = "data/code_questions.jsonl"
    
    if os.path.exists(passages_file) and os.path.exists(questions_file):
        all_passages = {}
        with open(passages_file, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                all_passages[p["passage_id"]] = p
                
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [json.loads(line) for line in f]
            
        print(f"Testing generator on {min(3, len(questions))} real code questions...\n")

        for q in questions[:3]:
            gold_passages = [all_passages[pid] for pid in q["gold_passage_ids"] if pid in all_passages]
            answer, cited_ids = generate_answer_with_citations(q["question"], gold_passages)
            print(f"Q: {q['question']}")
            print(f"A: {answer}")
            print(f"CITATIONS: {', '.join(cited_ids) if cited_ids else 'None'}\n" + "-" * 50)
    else:
        dummy_passages = [
            {
                "passage_id": "sessions.py::Session.merge_environment_settings",
                "title": "sessions.py::Session.merge_environment_settings",
                "text": "def merge_environment_settings(self, url, proxies, stream, verify, cert):\n    if verify is None:\n        verify = self.verify\n    return {'verify': verify}",
            }
        ]
        dummy_question = "How does requests decide whether to verify SSL certificates?"
        print("Testing Code-RAG Generator Standalone...")
        answer, cited_ids = generate_answer_with_citations(dummy_question, dummy_passages)
        print(f"Question: {dummy_question}")
        print(f"A: {answer}")
        print(f"CITATIONS: {', '.join(cited_ids) if cited_ids else 'None'}")
