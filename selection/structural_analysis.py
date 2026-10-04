import numpy as np
import networkx as nx

def compute_subgraph_properties(similarity_matrix, selected_idx, threshold=0.5):
    """
    similarity_matrix: full NxN candidate similarity matrix for this question
    selected_idx: indices of the passages actually selected by a given method
    threshold: similarity above which two passages count as "connected"
    """
    sub_matrix = similarity_matrix[np.ix_(selected_idx, selected_idx)]
    n = len(selected_idx)

    off_diagonal = sub_matrix[~np.eye(n, dtype=bool)]
    avg_similarity = float(off_diagonal.mean()) if len(off_diagonal) > 0 else 0.0

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if sub_matrix[i][j] >= threshold:
                G.add_edge(i, j)

    return {
        "avg_pairwise_similarity": avg_similarity,
        "density": float(nx.density(G)),
        "clustering_coefficient": float(nx.average_clustering(G)),
    }