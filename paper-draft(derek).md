Methodology subsection:

### Evaluation metrics

For each question, the retrieved passage IDs are compared with the gold passage IDs supplied with that question. Let $S_q$ be the set of distinct IDs selected for question $q$, and let $G_q$ be its gold ID set. The overlap between them is $S_q \cap G_q$. Precision@k measures how many selected IDs are gold IDs, while recall@k measures how many gold IDs were selected:

$$
\mathrm{Precision}_q@k = \frac{|S_q \cap G_q|}{|S_q|},
\qquad
\mathrm{Recall}_q@k = \frac{|S_q \cap G_q|}{|G_q|}.
$$

If either denominator is zero, the scorer assigns that question a value of zero for the corresponding measure. The reported precision@k and recall@k are the arithmetic means of the per-question scores. In this experiment, each question has one gold passage ID and the selection budget is $k=5$. As a result, recall for an individual question is one when its gold passage is selected and zero when it is missed. Since the scores use sets, repeated IDs count once.

Citation precision checks whether the passage IDs cited in an answer were among the passages provided to the generator. For question $q$, let $C_q$ be the list of parsed citation IDs and let $P_q$ be the provided passage IDs. The scorer returns zero when there are no citations. Otherwise, it divides the number of cited IDs found in the provided set by the total number of parsed citations:

$$
\mathrm{CitationPrecision}_q =
\begin{cases}
\frac{\sum_{c \in C_q} \mathbf{1}[c \in P_q]}{|C_q|}, & |C_q| > 0, \\
0, & |C_q| = 0.
\end{cases}
$$

The reported citation precision is the mean of these per-answer scores for each selection method. One detail in the current implementation affects how this result should be read: the generator filters out any citation IDs that do not match the provided passages before returning them to the scorer. The scorer therefore cannot evaluate invalid IDs from the model's original response. In practice, this version of citation precision mostly reflects whether an answer included at least one accepted citation.

Hallucination rate is measured with an LLM judge. For each answer, the judge is given the question, the answer, and the passages provided to the generator. It is asked whether the answer contains claims unsupported by those passages. The code counts an answer as flagged when the judge's response begins with “YES.” If $J_q$ is one for a flagged answer and zero otherwise, the hallucination rate across $N$ answers is:

$$
\mathrm{HallucinationRate} = \frac{1}{N}\sum_{q=1}^{N}J_q.
$$

This is the rate of answers flagged by the judge, not a human-verified rate of factual errors. The current judge uses Gemini 3.6 Flash. These metrics are calculated separately for each selection method and saved in the Week 3 comparison results.


Discussion:

The Week 2 comparison showed that the top-k method performed better than the original facility-location method on the 18-question test set. Top-k reached precision@5 of 0.1630 and recall@5 of 0.7778. MMR scored 0.1444 and 0.7222, while facility location scored 0.0889 and 0.4444. The facility-location method selected passages to cover the candidate pool. After the initial retrieval step, its selection rule did not directly use the question's relevance to each passage. This helps explain why a set could represent several parts of the candidate pool while still leaving out the gold passage for a specific question.

For Week 3, query relevance was added to the facility-location score. At each selection step, the method combines the gain in coverage across the candidate pool with the candidate passage's similarity to the question. The default alpha value is 0.5, so each term has a coefficient of 0.5 in the score. The terms have different scales, however, so this coefficient does not guarantee that relevance and coverage have equal influence. The weighted method reached precision@5 of 0.1444 and recall@5 of 0.7222. This is higher than the Week 2 facility-location result and matches MMR, though top-k remained higher on both retrieval measures. The result suggests that adding question relevance helped this version of facility location on this test set. It does not show that weighted facility location is generally better than the baselines.

The structural analysis looked at the five passages selected by weighted facility location for each question. It connected passage pairs with cosine similarity of at least 0.5 and recorded average pairwise similarity, density, and clustering coefficient. The gold passage appeared in the selected set for 13 questions and was missed for five. The average pairwise similarity was about 0.618 for questions where the gold passage was selected and 0.642 where it was missed. All five miss sets had density of 1.0, which means every pair of selected passages exceeded the similarity threshold. Some successful sets also had density 1.0, so density alone does not separate hits from misses. These results suggest that the method can select a closely related group of passages that still omits the gold passage, although this small analysis cannot establish that similarity caused the misses.

A case-by-case error analysis would help show where those misses happened. For each missed gold passage, it should check whether that passage was in the initial pool of 15 candidates. If it was, the error occurred during selection; if not, the retriever did not make it available to the selector. It should also check how answers behaved when the gold passage was selected.

These results have several limits. The evaluation uses 18 questions about one codebase, Requests. The findings may differ for other repositories and question types. The embedding model, all-mpnet-base-v2, is also the only one tested here, so another model could produce a different candidate pool or ranking. Passage IDs are not unique for every code chunk in the corpus, and the metric converts IDs to sets. As a result, duplicate IDs can affect the precision denominator and make separate chunks indistinguishable during scoring.

The project also changed its generation model from Gemini 2.5 Flash to Gemini 3.6 Flash during development. This limits comparisons of generated answers, citation behavior, and judge-based hallucination rates across runs. The model change does not explain changes in precision@k or recall@k, since those measures compare the selected IDs against the gold IDs. The hallucination check relies on an LLM judge and has not been validated against human judgments. Citation precision also needs cautious interpretation because invalid IDs are filtered before scoring. Taken together, the results are useful for understanding this implementation, though more questions, codebases, and controlled comparisons are needed before drawing broader conclusions.
