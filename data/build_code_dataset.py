import ast, json, os

SOURCE_DIR = "data/source_repo/src/requests"

def extract_chunks(filepath):
       with open(filepath, "r", encoding="utf-8") as f:
           source = f.read()
       tree = ast.parse(source)
       chunks = []
       for node in ast.walk(tree):
           if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
               text = ast.get_source_segment(source, node)
               if text:
                   chunks.append({
                       "passage_id": f"{filepath}::{node.name}",
                       "title": f"{os.path.basename(filepath)}::{node.name}",
                       "text": text,
                       "file_path": filepath
                   })
       return chunks

all_passages = []
for root, _, files in os.walk(SOURCE_DIR):
       for fname in files:
           if fname.endswith(".py"):
               fpath = os.path.join(root, fname)
               all_passages.extend(extract_chunks(fpath))

with open("data/code_passages.jsonl", "w") as f:
       for p in all_passages:
           f.write(json.dumps(p) + "\n")

print(f"Extracted {len(all_passages)} code chunks.")