## Results

### Performance Comparison Across Selection Methods

| Method | Precision@5 | Recall@5 | Citation Precision | Hallucination Rate |
| --- | --- | --- | --- | --- |
| **Top-$k$ Baseline** | 0.16 | 0.78 | 0.89 | 0.00 |
| **MMR** | 0.14 | 0.72 | 0.83 | 0.06 |
| **Reranking** | 0.16 | 0.78 | 0.67 | 0.00 |
| **Facility Location (Weighted)** | 0.14 | 0.72 | 0.78 | 0.00 |

Across the 4 methods, each evaluating 5 passages, standard Top-k and Reranking achieved the highest retrieval overlap with recall scores of 0.78, while Facility Location and MMR both reached 0.72, showing that enforcing structural diversity is a tradeoff for higher raw recall rates. In terms of retrieval precision, Top-k and Reranking achieved Precision@5 of 0.16, outperforming MMR and Facility Location, which showed 0.14 precision rates. This indicates that focusing on similarity yields somewhat more targeted results, whereas diversity selection returns a wider variety. Citation precision ranged from 0.89 (Top-K) to 0.67 (Reranking), with in-between values of 0.83 for MMR and 0.78 for Facility Location. Hallucination rate was 0 for all methods except MMR, which showed a slight bump to 0.06.    
