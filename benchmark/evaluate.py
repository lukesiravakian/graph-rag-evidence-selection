"""Evaluate every selection rule on the multi-gold, multi-repository benchmark.

Retrieval is per repository: a question about ``requests`` is answered from the
``requests`` corpus, which is how codebase QA actually works and keeps the
repository-level train/test split clean.

Three recall targets are reported, and the third is the one the single-repo
benchmark could not measure at all:

  recall           over the full gold set
  recall_anchor    over the anchor alone, the passage the query was derived from
  recall_compl     over gold minus the anchor, i.e. the complementary passages

``recall_compl`` is the circularity-free measure. The query is the anchor's
docstring, so lexical overlap helps a pointwise retriever find the anchor and gives
it no signal at all about the complementary passages. A set-level selection rule
that genuinely finds jointly informative evidence should separate from pointwise
ranking on ``recall_compl`` even if the two tie on ``recall``.

All operations run on the cached embedding matrices, so the full sweep is seconds.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Resolve against this file, not the working directory, so the module works when
# imported from anywhere -- the same relative-path trap the single-repo scripts had.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH = PROJECT_ROOT / "benchmark"
RESULTS = PROJECT_ROOT / "results"

POOL_N = 30
BUDGETS = (5, 10)
ALPHAS = [round(0.1 * i, 1) for i in range(11)]
LAMBDAS = [round(0.1 * i, 1) for i in range(11)]

# Held-out repositories. Chosen before any result was computed, to span sizes and
# domains: one HTTP client, one template engine, one small utility.
TEST_REPOS = ("httpx", "jinja2", "yarl", "multidict")


# --------------------------------------------------------------------- selectors
def select_topk(sim: np.ndarray, rel: np.ndarray, k: int, w: float) -> list[int]:
    return list(np.argsort(-rel)[:k])


def select_mmr(sim: np.ndarray, rel: np.ndarray, k: int, lam: float) -> list[int]:
    chosen: list[int] = []
    remaining = list(range(len(rel)))
    while len(chosen) < k and remaining:
        if not chosen:
            best = max(remaining, key=lambda i: rel[i])
        else:
            best = max(remaining,
                       key=lambda i: lam * rel[i] - (1 - lam) * max(sim[i][j] for j in chosen))
        chosen.append(best)
        remaining.remove(best)
    return chosen


def select_fl(sim: np.ndarray, rel: np.ndarray, k: int, alpha: float) -> list[int]:
    """Algorithm 1: greedy relevance-weighted facility location."""
    n = len(rel)
    chosen: list[int] = []
    cur = np.zeros(n)
    for _ in range(k):
        best, best_score = -1, -np.inf
        for i in range(n):
            if i in chosen:
                continue
            gain = np.maximum(cur, sim[i]).sum() - cur.sum()
            score = (1 - alpha) * gain + alpha * rel[i]
            if score > best_score:
                best_score, best = score, i
        chosen.append(best)
        cur = np.maximum(cur, sim[best])
    return chosen


RANDOM_DRAWS = 25


def score_random(ctx: list[dict], k: int, draws: int = RANDOM_DRAWS) -> dict[str, float]:
    """
    Uniform random selection of k of N pool members, averaged over ``draws``.

    A fixed seed shared across questions would pick the same pool *ranks* every
    time, and since the pool is ordered by relevance that is not a uniform draw
    over gold locations -- it is one arbitrary sample. Each question therefore gets
    its own seed, and the whole thing is averaged over several draws so the number
    is an expectation rather than a single realisation.
    """
    acc = {"precision": 0.0, "recall": 0.0, "recall_anchor": 0.0, "recall_compl": 0.0}
    for d in range(draws):
        P, R, RA, RC = [], [], [], []
        for qi, c in enumerate(ctx):
            rng = np.random.default_rng(1000 * d + qi)
            n = len(c["rel"])
            idx = rng.choice(n, size=min(k, n), replace=False)
            picked = {c["pool_ids"][i] for i in idx}
            P.append(len(picked & c["gold"]) / len(picked) if picked else 0.0)
            R.append(len(picked & c["gold"]) / len(c["gold"]))
            RA.append(1.0 if c["anchor"] in picked else 0.0)
            RC.append(len(picked & c["compl"]) / len(c["compl"]) if c["compl"] else 0.0)
        acc["precision"] += float(np.mean(P)) / draws
        acc["recall"] += float(np.mean(R)) / draws
        acc["recall_anchor"] += float(np.mean(RA)) / draws
        acc["recall_compl"] += float(np.mean(RC)) / draws
    acc["n"] = len(ctx)
    return acc


# --------------------------------------------------------------------- harness
def load() -> tuple[list[dict], list[dict], np.ndarray, np.ndarray]:
    passages = [json.loads(l) for l in (BENCH / "passages.jsonl").open()]
    questions = [json.loads(l) for l in (BENCH / "questions.jsonl").open()]
    pe = np.load(BENCH / "passage_emb.npy")
    qe = np.load(BENCH / "query_emb.npy")
    return passages, questions, pe, qe


def build_context(passages, questions, pe, qe) -> list[dict[str, Any]]:
    """Per question: its repo's pool of POOL_N candidates, with sim matrix and rel."""
    idx_by_repo: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(passages):
        idx_by_repo[p["repo"]].append(i)
    pid = [p["passage_id"] for p in passages]

    ctx = []
    for qi, q in enumerate(questions):
        rows = np.array(idx_by_repo[q["repo"]])
        scores = pe[rows] @ qe[qi]
        top = rows[np.argsort(-scores)[:POOL_N]]
        sub = pe[top]
        ctx.append({
            "qid": q["qid"],
            "repo": q["repo"],
            "pool_ids": [pid[i] for i in top],
            "sim": sub @ sub.T,
            "rel": sub @ qe[qi],
            "gold": set(q["gold_passage_ids"]),
            "anchor": q["anchor_id"],
            "compl": set(q["complementary_ids"]),
        })
    return ctx


def per_question(ctx: list[dict], sel: Callable, w: float, k: int) -> dict[str, np.ndarray]:
    """Per-question metric vectors, so methods can be compared with a paired test."""
    P, R, RA, RC = [], [], [], []
    for c in ctx:
        picked = {c["pool_ids"][i] for i in sel(c["sim"], c["rel"], k, w)}
        P.append(len(picked & c["gold"]) / len(picked) if picked else 0.0)
        R.append(len(picked & c["gold"]) / len(c["gold"]))
        RA.append(1.0 if c["anchor"] in picked else 0.0)
        RC.append(len(picked & c["compl"]) / len(c["compl"]) if c["compl"] else 0.0)
    return {"precision": np.array(P), "recall": np.array(R),
            "recall_anchor": np.array(RA), "recall_compl": np.array(RC)}


def score(ctx: list[dict], sel: Callable, w: float, k: int) -> dict[str, float]:
    pq = per_question(ctx, sel, w, k)
    out = {m: float(v.mean()) for m, v in pq.items()}
    out["n"] = len(ctx)
    return out


def wilcoxon(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """
    Two-sided Wilcoxon signed-rank test on the paired differences.

    Recall here is a fraction of a small gold set, so it is neither normal nor
    continuous; a signed-rank test on the paired differences is the appropriate
    choice and does not assume either.
    """
    from scipy.stats import wilcoxon as _w
    d = x - y
    nz = int(np.sum(d != 0))
    if nz == 0:
        return {"n_nonzero": 0, "p": 1.0, "mean_diff": 0.0}
    stat, p = _w(x, y, zero_method="wilcox", alternative="two-sided")
    return {"n_nonzero": nz, "p": float(p), "mean_diff": float(d.mean())}


def pool_ceiling(ctx: list[dict]) -> dict[str, float]:
    """Best achievable by any rule selecting from the pool, at unlimited budget."""
    R, RC = [], []
    for c in ctx:
        inpool = set(c["pool_ids"])
        R.append(len(inpool & c["gold"]) / len(c["gold"]))
        RC.append(len(inpool & c["compl"]) / len(c["compl"]) if c["compl"] else 0.0)
    return {"recall": float(np.mean(R)), "recall_compl": float(np.mean(RC))}


def main() -> None:
    passages, questions, pe, qe = load()
    ctx = build_context(passages, questions, pe, qe)
    train = [c for c in ctx if c["repo"] not in TEST_REPOS]
    test = [c for c in ctx if c["repo"] in TEST_REPOS]
    print(f"{len(ctx)} questions over {len({c['repo'] for c in ctx})} repos "
          f"| train {len(train)} ({len({c['repo'] for c in train})} repos) "
          f"| test {len(test)} ({len({c['repo'] for c in test})} repos)")
    print(f"pool N = {POOL_N}, mean |G| = "
          f"{np.mean([len(c['gold']) for c in ctx]):.2f}\n")

    out: dict[str, Any] = {
        "config": {"pool_n": POOL_N, "budgets": list(BUDGETS),
                   "test_repos": list(TEST_REPOS), "questions": len(ctx),
                   "train_n": len(train), "test_n": len(test),
                   "mean_gold": float(np.mean([len(c["gold"]) for c in ctx]))},
        "pool_ceiling": {"all": pool_ceiling(ctx), "train": pool_ceiling(train),
                         "test": pool_ceiling(test)},
        "sweeps": {}, "selected": {}, "fixed": {},
    }

    for k in BUDGETS:
        # fixed-setting baselines, including the paper's defaults
        fixed = {
            "top_k": (select_topk, 0.0),
            "mmr_lambda0.5": (select_mmr, 0.5),
            "fl_alpha0.5": (select_fl, 0.5),
            "fl_alpha0": (select_fl, 0.0),
        }
        out["fixed"][f"k{k}"] = {
            name: {"train": score(train, f, w, k), "test": score(test, f, w, k)}
            for name, (f, w) in fixed.items()
        }
        out["fixed"][f"k{k}"]["random"] = {
            "train": score_random(train, k), "test": score_random(test, k)}

        # sweeps on TRAIN only
        a_rows = [{"w": a, **score(train, select_fl, a, k)} for a in ALPHAS]
        l_rows = [{"w": l, **score(train, select_mmr, l, k)} for l in LAMBDAS]
        out["sweeps"][f"k{k}"] = {"fl_alpha": a_rows, "mmr_lambda": l_rows}

        # select the weight on train, then evaluate once on test
        best_a = max(a_rows, key=lambda r: r["recall"])["w"]
        best_l = max(l_rows, key=lambda r: r["recall"])["w"]
        out["selected"][f"k{k}"] = {
            "fl_alpha": {"chosen_on_train": best_a,
                         "train": score(train, select_fl, best_a, k),
                         "test": score(test, select_fl, best_a, k)},
            "mmr_lambda": {"chosen_on_train": best_l,
                           "train": score(train, select_mmr, best_l, k),
                           "test": score(test, select_mmr, best_l, k)},
        }

    # Paired tests on the held-out split: does any selection rule beat plain top-k?
    out["paired_tests_test_split"] = {}
    for k in BUDGETS:
        base = per_question(test, select_topk, 0.0, k)
        rows = {}
        for name, (f, w) in {
            "mmr_lambda0.5": (select_mmr, 0.5),
            "fl_alpha0.5": (select_fl, 0.5),
            "fl_alpha0": (select_fl, 0.0),
            f"fl_alpha{out['selected'][f'k{k}']['fl_alpha']['chosen_on_train']}":
                (select_fl, out["selected"][f"k{k}"]["fl_alpha"]["chosen_on_train"]),
        }.items():
            cand = per_question(test, f, w, k)
            rows[name] = {
                m: wilcoxon(cand[m], base[m])
                for m in ("recall", "recall_compl")
            }
        out["paired_tests_test_split"][f"k{k}"] = rows

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "multigold_benchmark.json").write_text(json.dumps(out, indent=2))

    # ---------------------------------------------------------------- report
    for k in BUDGETS:
        print(f"{'='*86}\nBUDGET k = {k}   (pool ceiling: recall "
              f"{out['pool_ceiling']['all']['recall']:.3f}, "
              f"compl {out['pool_ceiling']['all']['recall_compl']:.3f})\n{'='*86}")
        print(f"{'method':22s} {'split':6s} {'P':>7} {'recall':>8} {'anchor':>8} {'compl':>8}")
        for name, v in out["fixed"][f"k{k}"].items():
            for sp in ("train", "test"):
                s = v[sp]
                print(f"{name:22s} {sp:6s} {s['precision']:7.4f} {s['recall']:8.4f} "
                      f"{s['recall_anchor']:8.4f} {s['recall_compl']:8.4f}")
        sel = out["selected"][f"k{k}"]
        print(f"\n{'-- weight tuned on train, evaluated on held-out repos --':<60}")
        for name, v in sel.items():
            print(f"{name:22s} chose w = {v['chosen_on_train']}")
            for sp in ("train", "test"):
                s = v[sp]
                print(f"{'':22s} {sp:6s} {s['precision']:7.4f} {s['recall']:8.4f} "
                      f"{s['recall_anchor']:8.4f} {s['recall_compl']:8.4f}")
        pt = out["paired_tests_test_split"][f"k{k}"]
        print(f"\n-- paired Wilcoxon vs top-k on held-out repos (n={len(test)}) --")
        print(f"{'method':24s} {'d(recall)':>10} {'p':>8}   {'d(compl)':>10} {'p':>8}")
        for name, v in pt.items():
            r, c = v["recall"], v["recall_compl"]
            print(f"{name:24s} {r['mean_diff']:>+10.4f} {r['p']:>8.4f}   "
                  f"{c['mean_diff']:>+10.4f} {c['p']:>8.4f}")
        print(f"\nsweep (train) k={k}:")
        print("  alpha :", " ".join(f"{r['recall']:.3f}" for r in out["sweeps"][f"k{k}"]["fl_alpha"]))
        print("  lambda:", " ".join(f"{r['recall']:.3f}" for r in out["sweeps"][f"k{k}"]["mmr_lambda"]))
        print()

    print(f"wrote {RESULTS / 'multigold_benchmark.json'}")


if __name__ == "__main__":
    main()
