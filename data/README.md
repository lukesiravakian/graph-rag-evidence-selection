# Data files

   - `hotpotqa_dev_300.jsonl` — 300 HotpotQA dev questions. Each line: qid, question, answer, supporting_titles, passages (list of {passage_id, title, text}).
   - `all_passages.jsonl` — every unique passage across all 300 questions, deduplicated by passage_id.

   Regenerate both by running: `python3.14 data/build_dataset.py`