"""Section V: the statistics quoted next to Tables II and III.

Reproduces, from the saved result files, every inferential number the paper reports:
  - 95% Wilson intervals for recall@5 (Fig. 2a error bars)
  - exact McNemar p-values for the recall gaps between methods
  - a two-sided Fisher exact test on the density-versus-hit table (Table III)

Retrieval counts are read from results/week3_full_comparison.json and
results/week2_comparison.json, so this script needs no model and no API key.
"""
from __future__ import annotations

import json
from math import comb, sqrt
from pathlib import Path
from sys import path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in path:
    path.insert(0, str(PROJECT_ROOT))

MISS_PATH = PROJECT_ROOT / "results" / "miss_analysis.json"
WEEK3_PATH = PROJECT_ROOT / "results" / "week3_full_comparison.json"
WEEK2_PATH = PROJECT_ROOT / "results" / "week2_comparison.json"
STRUCTURAL_PATH = PROJECT_ROOT / "results" / "structural_analysis.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "significance_tests.json"

Z = 1.959963984540054  # two-sided 95% normal quantile


def wilson_interval(hits: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def mcnemar_exact(b: int, c: int) -> float:
    """
    Two-sided exact McNemar p-value for b and c discordant pairs.

    Under the null each discordant pair favors either method with probability 1/2,
    so the p-value is the two-sided binomial tail at min(b, c) out of b + c.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """
    Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]],
    summing every table at least as extreme as the observed one by probability.
    """
    row1, row2 = a + b, c + d
    col1, total = a + c, a + b + c + d

    def prob(x: int) -> float:
        return (comb(row1, x) * comb(row2, col1 - x)) / comb(total, col1)

    observed = prob(a)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    # 1e-12 guards against tables that are equiprobable only up to float error.
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= observed * (1 + 1e-12)))


def main() -> None:
    week3 = json.loads(WEEK3_PATH.read_text())["comparison"]
    week2 = json.loads(WEEK2_PATH.read_text())["comparison"]
    rows = json.loads(STRUCTURAL_PATH.read_text())

    n_questions = len(rows)

    hits = {
        "top_k": round(week3["top_k"]["recall@k"] * n_questions),
        "reranking": round(week3["reranking"]["recall@k"] * n_questions),
        "mmr": round(week3["mmr"]["recall@k"] * n_questions),
        "facility_location_weighted":
            round(week3["facility_location_weighted"]["recall@k"] * n_questions),
        "facility_location_coverage_only":
            round(week2["facility_location"]["recall@k"] * n_questions),
    }

    intervals = {
        name: {
            "hits": h,
            "n": n_questions,
            "recall": h / n_questions,
            "wilson_95": list(wilson_interval(h, n_questions)),
        }
        for name, h in hits.items()
    }

    # When miss_analysis.json is present it carries per-question hit/miss for every
    # method, so the paired tests the comparison calls for can be computed directly
    # rather than bounded. The bounds below are kept for the case where it is absent.
    paired: dict[str, Any] = {}
    overlap: dict[str, Any] = {}
    if MISS_PATH.exists():
        miss = json.loads(MISS_PATH.read_text())
        qrows = miss["per_question"]
        qids = [r["qid"] for r in qrows]
        hits_by_q = {m: {r["qid"]: r["hits"][m] for r in qrows}
                     for m in qrows[0]["hits"]}

        def pair(a: str, b: str) -> dict[str, Any]:
            bb = sum(1 for q in qids if hits_by_q[a][q] and not hits_by_q[b][q])
            cc = sum(1 for q in qids if hits_by_q[b][q] and not hits_by_q[a][q])
            return {"b": bb, "c": cc, "p_exact": mcnemar_exact(bb, cc)}

        names = list(hits_by_q)
        for other in names:
            if other != "top_k":
                paired[f"top_k vs {other}"] = pair("top_k", other)
        if {"facility_location_weighted", "facility_location_coverage_only"} <= set(names):
            paired["facility_location_weighted vs coverage_only"] = pair(
                "facility_location_weighted", "facility_location_coverage_only")

        # Equal aggregate recall does not imply equal behavior: these methods can
        # score the same while failing on disjoint questions. Record that directly.
        import itertools
        main4 = [m for m in ("top_k", "reranking", "mmr",
                             "facility_location_weighted") if m in names]
        misses = {m: {q for q in qids if not hits_by_q[m][q]} for m in main4}
        overlap["pairwise_shared_misses"] = {
            f"{a} vs {b}": {"shared": sorted(misses[a] & misses[b]),
                            "union": sorted(misses[a] | misses[b])}
            for a, b in itertools.combinations(main4, 2)
        }
        overlap["union_hits"] = {
            "+".join(c): sum(1 for q in qids if any(hits_by_q[m][q] for m in c))
            for r in (2, 3, 4) for c in itertools.combinations(main4, r)
        }
        overlap["missed_by_every_method"] = [
            q for q in qids if not any(hits_by_q[m][q] for m in main4)]
        overlap["budget_caveat"] = (
            "union_hits are NOT budget-matched: the union of two k=5 selections holds "
            "up to 10 passages, so these counts must be compared against a k=10 "
            "single-method control before any claim is made from them.")

    net_gaps = {
        "top_k vs facility_location_weighted":
            hits["top_k"] - hits["facility_location_weighted"],
        "top_k vs mmr": hits["top_k"] - hits["mmr"],
        "top_k vs facility_location_coverage_only":
            hits["top_k"] - hits["facility_location_coverage_only"],
    }
    mcnemar = {
        label: {
            "net_gap": gap,
            "best_case_discordance": [gap, 0],
            "p_best_case": mcnemar_exact(gap, 0),
            "p_one_discordant_against": mcnemar_exact(gap + 1, 1),
        }
        for label, gap in net_gaps.items()
    }

    sparse = [r for r in rows if r["density"] < 1.0]
    dense = [r for r in rows if r["density"] >= 1.0]
    a, b = sum(r["hit"] for r in sparse), len(sparse) - sum(r["hit"] for r in sparse)
    c, d = sum(r["hit"] for r in dense), len(dense) - sum(r["hit"] for r in dense)
    density_table = {
        "sparse_density_lt_1": {"n": len(sparse), "hits": a, "misses": b,
                                "hit_rate": a / len(sparse) if sparse else 0.0},
        "dense_density_eq_1": {"n": len(dense), "hits": c, "misses": d,
                               "hit_rate": c / len(dense) if dense else 0.0},
        "fisher_exact_two_sided": fisher_exact_two_sided(a, b, c, d),
    }

    output = {
        "n_questions": n_questions,
        "recall_intervals": intervals,
        "mcnemar_paired_exact": paired,
        "mcnemar_bounds": mcnemar,
        "error_profile_overlap": overlap,
        "density_vs_hit": density_table,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"n = {n_questions} questions\n")
    print("Recall@5 with 95% Wilson intervals")
    for name, v in intervals.items():
        lo, hi = v["wilson_95"]
        print(f"  {name:34s} {v['hits']:>2}/{v['n']}  {v['recall']:.4f}  [{lo:.2f}, {hi:.2f}]")

    if paired:
        print("\nPaired exact McNemar (computed from per-question outcomes)")
        for label, v in paired.items():
            star = "" if v["p_exact"] >= 0.05 else "  *"
            print(f"  {label:52s} b={v['b']:>2} c={v['c']:>2}  p = {v['p_exact']:.4f}{star}")
        if overlap:
            print("\nShared misses between methods with near-equal recall")
            for label, v in overlap["pairwise_shared_misses"].items():
                print(f"  {label:58s} {len(v['shared'])}/{len(v['union'])} shared")
            print("\nUnion hit counts (NOT budget-matched; see budget_caveat)")
            for label, v in sorted(overlap["union_hits"].items(),
                                   key=lambda kv: (len(kv[0].split('+')), -kv[1])):
                print(f"  {label:58s} {v}/{n_questions}")

    print("\nExact McNemar bounds on the recall gaps (kept for when per-question data is absent)")
    for label, v in mcnemar.items():
        print(f"  {label:48s} net {v['net_gap']:>2}  "
              f"p = {v['p_best_case']:.3f} best case, "
              f"{v['p_one_discordant_against']:.3f} with one against")

    s = density_table["sparse_density_lt_1"]
    dn = density_table["dense_density_eq_1"]
    print("\nSelected-set density versus retrieval outcome (Table III)")
    print(f"  density < 1.0   n = {s['n']:>2}  gold retrieved {s['hits']:>2}  "
          f"hit rate {s['hit_rate']:.3f}")
    print(f"  density = 1.0   n = {dn['n']:>2}  gold retrieved {dn['hits']:>2}  "
          f"hit rate {dn['hit_rate']:.3f}")
    print(f"  Fisher exact (two-sided) p = {density_table['fisher_exact_two_sided']:.2f}")
    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
