# graph-based-evidence
Graph-based evidence selection for Retrieval-Augmented Generation (RAG) to reduce redundancy and improve LLM reasoning on multi-hop QA benchmarks.


Large language models often suffer from performance degradation and hallucinations when presented with redundant or noisy context. This project investigates whole-subset optimization using candidate document subgraphs. Instead of taking the top-k passages by similarity score alone, we construct semantic overlap graphs and apply submodular optimization (facility location) to curate diverse evidence sets for downstream QA tasks.
