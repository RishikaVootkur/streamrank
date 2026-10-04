# 3. Full-catalog evaluation and baseline suite

Date: 2026-10-04

## Status

Accepted

## Context

A two-tower model only earns its complexity if it beats strong simple models under an honest protocol. Sampled-negative metrics can reorder models, and neural recommenders have repeatedly failed to beat well-tuned simple baselines when those were tuned properly.

## Options considered

1. Sampled metrics (rank the positive among 100 random items). Cheap, but inconsistent with full ranking.
2. Full-catalog ranking with per-user metrics and bootstrap intervals over users.
3. Baselines: popularity only, or a suite of popularity, recent popularity, item KNN, EASE, and ALS, each tuned on validation.

## Decision

Option 2 with the full baseline suite.

- Every warm validation user (at least one training interaction and one positive) is ranked against all 65,723 catalog items, with their training items excluded. Relevant items first seen after the cut stay in the denominator even though no ID-based model can retrieve them, so recall is not inflated.
- Metrics per user: Recall@10/50/100/200, NDCG@10, MRR, and ARP@10 (mean training popularity share of recommended items, a popularity-bias measure), plus catalog coverage and Gini of the top-10 lists. Each mean has a 95% percentile bootstrap interval over users (1,000 resamples). Model comparisons use paired bootstrap differences on the same users.
- Cold users (7,368 in validation, more than the 6,463 warm users) cannot be personalized by any of these models. They are counted and reported, and handled by the popularity fallback at serving time.
- Baselines: popularity, recent popularity (30-day window with an all-time tie-break), item KNN with shrunk cosine, EASE on the 20,000 most popular items (closed form, λ tuned), and implicit ALS. Personalized models add a popularity tie-break of at most 1e-6, so users with empty input rows get a popularity list instead of arbitrary index order.
- Tuning: grid search on the 10% user sample's validation partition by Recall@100 (the retrieval stage keeps 200 candidates, and the M4 criterion is Recall@100). Tuning never reads test data, and the runner rejects splits that differ in data hash or cut points. The best setting per family is refit and evaluated on the full split. After a first pass, grids were widened where the winner sat on an edge. EASE's chosen λ = 2000 now has grid values on both sides. The remaining edge choices (ALS with 64 factors, item KNN with 50 neighbors) are on the low-capacity side, and both models trail EASE by more than their intervals.
- Learning from positives only (rating ≥ 4) beat learning from all ratings for every personalized model.

## Results (full split, validation)

| Model | Recall@100 | NDCG@10 |
| --- | ---: | ---: |
| popularity | 0.1411 [0.1368, 0.1455] | 0.0695 [0.0660, 0.0731] |
| recent popularity | 0.2341 [0.2281, 0.2403] | 0.1155 [0.1114, 0.1199] |
| item KNN | 0.2311 [0.2252, 0.2370] | 0.1087 [0.1044, 0.1130] |
| EASE | 0.2854 [0.2795, 0.2912] | 0.1484 [0.1432, 0.1534] |
| ALS | 0.2643 [0.2583, 0.2701] | 0.1281 [0.1234, 0.1325] |

EASE is the best baseline and the bar for the two-tower model. Recent popularity beats all-time popularity by a wide margin, a sign of strong temporal drift that the sequential user tower should exploit. Full table: `docs/baselines.md`.

## Consequences

- `make baselines` reproduces the table in about 12 minutes on the development machine and writes a manifest with the git SHA, grids, input split hashes, and seed.
- Any new model is scored with the same `fit_and_evaluate` path and counts as better only if it beats EASE beyond the interval.

## Sources

- Krichene and Rendle, "On Sampled Metrics for Item Recommendation", KDD 2020: https://dl.acm.org/doi/10.1145/3394486.3403226
- Ferrari Dacrema, Cremonesi, Jannach, "Are We Really Making Much Progress?", RecSys 2019: https://arxiv.org/abs/1907.06902
- Steck, "Embarrassingly Shallow Autoencoders for Sparse Data", WWW 2019: https://arxiv.org/abs/1905.03375
- Hu, Koren, Volinsky, "Collaborative Filtering for Implicit Feedback Datasets", ICDM 2008: https://doi.org/10.1109/ICDM.2008.22
- implicit library: https://github.com/benfred/implicit
- Rendle et al., "Neural Collaborative Filtering vs. Matrix Factorization Revisited", RecSys 2020: https://arxiv.org/abs/2005.09683
