import numpy as np
from numpy.typing import NDArray


def facility_location_select(
    similarity_matrix: NDArray[np.floating],
    k: int = 5,
) -> list[int]:
    """
    Coverage-only facility location: the alpha = 0 case of Algorithm 1 in the paper.

    At each step this picks the passage that most increases total 'coverage'
    (how well every candidate is represented by something selected) and makes
    no reference to the query beyond the pool retrieval already produced.

    similarity_matrix: NxN pool similarity matrix, sim[i][j] = cosine(i, j)
    k: number of passages to select
    Returns: list of indices into the original passages list
    """
    n = similarity_matrix.shape[0]
    return relevance_weighted_facility_location_select(
        similarity_matrix,
        np.zeros(n),
        k=k,
        alpha=0.0,
    )


def relevance_weighted_facility_location_select(
    similarity_matrix: NDArray[np.floating],
    query_similarities: NDArray[np.floating],
    k: int = 5,
    alpha: float = 0.5,
) -> list[int]:
    """
    Algorithm 1 of the paper: greedy relevance-weighted facility location.

    Each step scores every unselected candidate c with
        score(c | S) = (1 - alpha) * coverage_gain(c) + alpha * rel(c)
    where coverage_gain(c) is the marginal gain in f(S) = sum_i max_{j in S} sim(i, j)
    and rel(c) = query_similarities[c].

    alpha controls the balance: 0 = pure coverage (query-agnostic), 1 = pure relevance.
    0.5 is the value used for every result reported in the paper.

    similarity_matrix: NxN pool similarity matrix, sim[i][j] = cosine(i, j)
    query_similarities: length-N array, query_similarities[i] = cosine(passage i, query)
    k: number of passages to select
    alpha: weight on the relevance term
    Returns: list of indices into the original passages list
    """
    return _greedy_select(similarity_matrix, query_similarities, k, alpha)[0]


def relevance_weighted_facility_location_trace(
    similarity_matrix: NDArray[np.floating],
    query_similarities: NDArray[np.floating],
    k: int = 5,
    alpha: float = 0.5,
) -> tuple[list[int], list[dict[str, float]]]:
    """
    Run Algorithm 1 and also report the two score terms at each step.

    Selection is identical to relevance_weighted_facility_location_select; this
    only records what the terms looked like for the candidate actually chosen,
    which is what Section III-E of the paper reports.

    Returns: (selected indices, one trace dict per step) where each trace dict holds
        step            1-based selection step
        index           index of the chosen candidate
        coverage_gain   the delta_c term for that candidate at that step
        relevance       rel(c) for that candidate
        ratio           coverage_gain / relevance (inf when relevance is 0)
        score           the combined (1 - alpha) * delta_c + alpha * rel(c)
    """
    return _greedy_select(similarity_matrix, query_similarities, k, alpha)


def _greedy_select(
    similarity_matrix: NDArray[np.floating],
    query_similarities: NDArray[np.floating],
    k: int,
    alpha: float,
) -> tuple[list[int], list[dict[str, float]]]:
    """Shared greedy loop behind the selectors above, so they cannot drift apart."""
    n = similarity_matrix.shape[0]
    selected: list[int] = []
    trace: list[dict[str, float]] = []
    current_max = np.zeros(n)

    for step in range(1, k + 1):
        best_gain = -1
        best_idx = -1
        best_coverage = 0.0
        best_relevance = 0.0

        for i in range(n):
            if i in selected:
                continue

            new_max = np.maximum(current_max, similarity_matrix[i])
            coverage_gain = new_max.sum() - current_max.sum()
            relevance_bonus = query_similarities[i]
            total_gain = (1 - alpha) * coverage_gain + alpha * relevance_bonus

            if total_gain > best_gain:
                best_gain = total_gain
                best_idx = i
                best_coverage = float(coverage_gain)
                best_relevance = float(relevance_bonus)

        selected.append(best_idx)
        current_max = np.maximum(current_max, similarity_matrix[best_idx])
        trace.append({
            "step": step,
            "index": best_idx,
            "coverage_gain": best_coverage,
            "relevance": best_relevance,
            "ratio": best_coverage / best_relevance if best_relevance else float("inf"),
            "score": float(best_gain),
        })

    return selected, trace


if __name__ == "__main__":
    # Fake similarity matrix test:
    # Passages 0 & 1 are duplicates (sim 0.9), passages 2 & 3 are duplicates (sim 0.8).
    fake_matrix = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.8],
        [0.1, 0.1, 0.8, 1.0],
    ])
    selected = facility_location_select(fake_matrix, k=2)
    print("Standalone test output indices:", selected)
