import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

def build_prompt(question: str, passages: list[dict]) -> str:
    """Formats prompt with codebase context instead of prose passages."""
    context = "\n\n".join(f"# {p['title']}\n{p['text']}" for p in passages)
    return f"""You are answering a question about a codebase using the code snippets below.

Code context:
{context}

Question: {question}
Answer concisely, referencing the relevant function/class by name where helpful.
Answer:"""

def generate_answer(question: str, passages: list[dict]) -> str:
    """Generates a code-grounded answer using Gemini."""
    prompt = build_prompt(question, passages)
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=50,
            temperature=0.0
        )
    )
    
    return response.text.strip()

if __name__ == "__main__":
    dummy_passages = [
        {"title": "sessions.py::Session.merge_environment_settings", "text": "def merge_environment_settings(self, url, proxies, stream, verify, cert):\n    if verify is None:\n        verify = self.verify\n    return {'verify': verify}"}
    ]
    dummy_question = "How does requests decide whether to verify SSL certificates?"
    
    print("Testing Code-RAG Generator Standalone...")
    result = generate_answer(dummy_question, dummy_passages)
    print(f"Question: {dummy_question}")
    print(f"Generated Answer: {result}")