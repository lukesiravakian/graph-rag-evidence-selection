# Retriever

Dense retrieval over the code corpus. Everything downstream — every selection method,
every metric — reads its candidates from here, so this module fixes the ceiling on
what any selection rule can achieve: a gold passage that never enters the pool cannot
be recovered later.

## Files

| File | Purpose |
| --- | --- |
| `build_index.py` | Embeds `data/code_passages.jsonl` and writes the FAISS index + ID map |
| `retrieve.py` | `retrieve(question, k)` and `retrieve_candidates(question, n)` |

Neither `passage_index.faiss` nor `id_map.json` is checked in (`.gitignore`), so the
index must be built once before anything else runs.

## Building the index

```bash
python retriever/build_index.py
```

This encodes all 320 passage texts with `all-mpnet-base-v2`, casts to `float32`,
L2-normalizes with `faiss.normalize_L2`, and adds them to an `IndexFlatIP`. Because
the vectors are unit-length, inner product *is* cosine similarity, and `IndexFlatIP`
is exhaustive rather than approximate — so retrieval is exact and deterministic. The
ID map is a plain list of `passage_id` strings in index order.

`build_index.py` encodes with `batch_size=1`, which is slow (minutes for 320 passages)
but has no effect on the resulting vectors.

## Retrieving

```python
from retriever.retrieve import retrieve, retrieve_candidates

retrieve(question, k=5)             # the plain top-k baseline
retrieve_candidates(question, n=15) # the pool every other method selects from
```

`retrieve_candidates` is a thin alias for `retrieve(question, k=n)`; the two names
exist so the pool size (`N = 15`) and the final budget (`k = 5`) stay distinguishable
at call sites.

## Behaviour worth knowing

- **Import loads the model and the index.** `retriever/retrieve.py` builds a
  `SentenceTransformer` and reads the FAISS index at module level, so `from
  retriever.retrieve import retrieve` is expensive and raises a `RuntimeError` with
  build instructions if the index is missing. `import retriever` is lazy and cheap.
- **Paths are relative to the working directory.** `PASSAGES_PATH`, `INDEX_PATH`, and
  `ID_MAP_PATH` are bare relative strings, so everything must run from the repository
  root even though the scripts fix up `sys.path` themselves.
- **`passage_id` is not unique, and lookup is by ID.** `retrieve` maps each FAISS
  index position through `id_map` and then into a `passage_id → passage` dict built
  from the JSONL. The index holds all 320 vectors, but only 280 IDs are distinct, so
  two different chunks sharing an ID resolve to the *same* passage dict — and a single
  retrieval can return that dict more than once, dropping the number of distinct IDs
  below `k`. This is the source of the precision artifact documented in
  `results/README.md`.

## Standalone check

```bash
python retriever/retrieve.py
```

Prints a per-question HIT/MISS breakdown of the top-5 baseline over
`data/code_questions.jsonl`, with the gold and retrieved IDs for each miss. This is
the cheapest way to see per-question retrieval outcomes, since the comparison scripts
do not persist them.
