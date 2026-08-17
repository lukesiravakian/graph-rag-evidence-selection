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