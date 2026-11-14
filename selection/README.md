# Selection

The core of the project: given the `N = 15` candidates retrieval returned for one
question, choose the `k = 5` passages to hand the generator. Every strategy here sees
the same pool, so downstream differences are attributable to the selection rule rather
than to retrieval.

## Strategies

| File | Function | Rule |
| --- | --- | --- |
| `facility_location.py` | `relevance_weighted_facility_location_select` | **The proposed method.** Greedy facility location plus an explicit query-relevance term |
| `facility_location.py` | `facility_location_select` | The coverage-only ablation (`α = 0`) |
| `mmr.py` | `mmr_select` | Maximal Marginal Relevance, `λ = 0.5` |
| `reranker.py` | `rerank_select` | Cross-encoder reranking baseline |
| `graph_builder.py` | `build_similarity_graph`, `compute_query_similarities` | The similarity graph and the `rel` vector the selectors consume |
| `structural_analysis.py` | `compute_subgraph_properties` | Density / clustering / average similarity of a chosen set |

Plain top-k is not here — it is just `retriever.retrieve(question, k=5)`, read straight
off the index without a pool.

## The proposed objective

Facility location scores a subset `S` by how well it represents every node of the
candidate graph:

```
f(S) = Σ_i max_{j ∈ S} sim(i, j)
```

`f` is monotone and submodular, so greedily adding the candidate with the largest
marginal gain stays within `1 − 1/e` of the optimal `k`-subset. The proposed rule adds
a relevance term, with `rel(c) = sim(c, query)`:

```
score(c | S) = (1 − α) · Δc + α · rel(c),     Δc = f(S ∪ {c}) − f(S),   α = 0.5
```

`α = 0` recovers the coverage-only rule exactly — `facility_location_select` is
implemented as that call, so the two cannot drift apart.

### The two terms are not on the same scale

`Δc` sums a contribution over all 15 pool members; `rel(c)` is a single cosine bounded
in `[−1, 1]`. At the first selection step, when nothing is covered yet, `Δc` for the
chosen passage runs **9.39–11.98** across the 18 questions against a `rel(c)` of
**0.44–0.74** — a ratio of **16.2 to 25.0** that the nominal `α = 0.5` does nothing to
close. The gap narrows as the pool saturates.

These are the measured values from `scripts/run_alpha_instrumentation.py` over all 18
questions. They are wider than the ranges quoted in the paper draft (Δc 11.0–12.0,
rel 0.57–0.74, ratio "roughly 16 to 20"), which appear to come from a subset of
questions; both upper bounds on the terms match, but the lower bounds and the ratio
ceiling do not. The qualitative conclusion is unaffected and in fact strengthens — the
coverage term can outweigh relevance by 25x, not 20x.

So `α = 0.5` does **not** split influence evenly across the five picks. Coverage
effectively controls the first pick or two, and relevance matters mainly for the picks
after that. `relevance_weighted_facility_location_trace` returns this per-step
breakdown alongside the selection — it runs the identical greedy loop, so the traced
run picks the same set as the untraced one. `scripts/run_alpha_instrumentation.py`
reports it over the benchmark.

## Shapes, and the index/passage split

The facility-location functions take a similarity **matrix** and return **indices**
into the candidate list; `mmr_select` and `rerank_select` take and return **passage
dicts**. Callers bridge the two with `[candidates[i] for i in idx]`. Worth knowing
before wiring a new strategy in.

## Behaviour worth knowing

- **Ties go to the lower index.** All greedy loops compare with strict `>` while
  scanning candidates in ascending order, so an exact tie keeps the earlier (more
  similar, since the pool arrives ranked) candidate.
- **Selection is deterministic.** Given the same index, the same five passages come
  back every time — which is why the analysis scripts can re-derive the sets the
  comparison script used instead of reading them back.
- **Duplicated embedding work.** `build_similarity_graph`, `compute_query_similarities`,
  and `mmr_select` each encode the 15 candidate texts themselves, so a single question
  encodes the same passages three times. Correct, just slower than it needs to be.
- **`facility_location.py` needs only numpy**; `graph_builder`, `mmr`, and `reranker`
  pull in `sentence_transformers` (and torch) at module level, which is why
  `selection/__init__.py` re-exports lazily.
- **`structural_analysis.py` needs networkx** for `nx.density` and
  `nx.average_clustering`. Edges are drawn at pairwise cosine `>= 0.5`, and density is
  the fraction of the ten possible pairs connected.

## Standalone checks

```bash
python selection/facility_location.py   # greedy picks on a 4-passage toy matrix
python selection/graph_builder.py       # sanity-checks the matrix on 10 real passages
python selection/test_relevance_fix.py  # the relevance term pulls in the relevant item
```

Only `graph_builder.py` needs the embedding model; the other two are pure numpy and
run instantly.
