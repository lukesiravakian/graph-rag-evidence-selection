<<<<<<< HEAD
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n\n".join(f"# {p['title']}\n{p['text']}" for p in passages)
    return f"""You are answering a question about a codebase using the code snippets below.

Code context:
{context}

Question: {question}
Answer in 1-2 concise sentences, referencing relevant functions/classes by name where applicable.
Answer:"""

def generate_answer(question: str, passages: list[dict]) -> str:
    """Generates a code-grounded answer using Gemini."""
    prompt = build_prompt(question, passages)
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
        system_instruction="You are a helpful coding assistant. Answer the user's question concisely in 1-2 sentences based on the provided code context. Mention relevant function or class names.",
            max_output_tokens=100,
            temperature=0.0
        )
    )
    
    # Safely handle response text
def generate_answer(question: str, passages: list[dict]) -> str:
    prompt = build_prompt(question, passages)
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful coding assistant. Answer the user's question concisely in 1-2 sentences based on the provided code context. Mention relevant function or class names.",
            max_output_tokens=500,
            temperature=0.1,
        )
    )
    
    # Direct check on top-level response.text
    if getattr(response, "text", None):
        return response.text.strip()
    
    # Safe fallback parsing without assuming parts is a list
    try:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if getattr(part, "text", None):
                    return part.text.strip()
    except (AttributeError, IndexError):
        pass
        
    return "No answer generated."
    
    try:
        return response.candidates[0].content.parts[0].text.strip()
    except (AttributeError, IndexError):
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
            
        print(f"Testing generator on {min(3, len(questions))} real questions...\n")
        
        for q in questions[:3]:
            gold_passages = [all_passages[pid] for pid in q["gold_passage_ids"] if pid in all_passages]
            answer = generate_answer(q["question"], gold_passages)
            print(f"Q: {q['question']}")
            print(f"A: {answer}\n" + "-" * 50)
    else:
        # Fallback dummy test if files don't exist yet
        dummy_passages = [
            {"title": "sessions.py::Session.merge_environment_settings", "text": "def merge_environment_settings(self, url, proxies, stream, verify, cert):\n    if verify is None:\n        verify = self.verify\n    return {'verify': verify}"}
        ]
        dummy_question = "How does requests decide whether to verify SSL certificates?"
        print("Testing Code-RAG Generator Standalone...")
        result = generate_answer(dummy_question, dummy_passages)
        print(f"Question: {dummy_question}")
        print(f"Generated Answer: {result}")
=======
from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def build_prompt(question, passages):
    context = "\n\n".join(f"[{p['title']}] {p['text']}" for p in passages)
    return f"""Answer the question using only the passages below. Be concise — answer in as few words as possible.

Passages:
{context}

Question: {question}
Answer:"""

def generate_answer(question, passages):
    prompt = build_prompt(question, passages)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip()
>>>>>>> upstream/main
