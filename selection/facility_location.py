import numpy as np

def facility_location_select(
    similarity_matrix: np.ndarray,
    k: int = 5,
) -> list[int]:
    """
    Greedy facility location: at each step, pick the passage that most increases 
    total 'coverage' (how well every candidate is represented by something selected).
    Returns a list of indices into the original passages list.
    """
    n = similarity_matrix.shape[0]
    selected = []
    current_max = np.zeros(n)
    
    for _ in range(k):
        best_gain = -1
        best_idx = -1
        
        for i in range(n):
            if i in selected:
                continue
            
            new_max = np.maximum(current_max, similarity_matrix[i])
            gain = new_max.sum() - current_max.sum()
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        selected.append(best_idx)
        current_max = np.maximum(current_max, similarity_matrix[best_idx])
        
    return selected

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


def relevance_weighted_facility_location_select(
    similarity_matrix: np.ndarray, 
    query_similarities: np.ndarray, 
    k: int = 5, 
    alpha: float = 0.5
) -> list[int]:
    """
    alpha controls the balance: 0 = pure diversity (old behavior), 1 = pure relevance.
    0.5 is a reasonable starting point.
    """
    n = similarity_matrix.shape[0]
    selected = []
    current_max = np.zeros(n)

    for _ in range(k):
        best_gain = -1
        best_idx = -1
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
        selected.append(best_idx)
        current_max = np.maximum(current_max, similarity_matrix[best_idx])

    return selected