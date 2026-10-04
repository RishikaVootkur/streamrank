# 8. LightGBM LambdaMART ranker

Date: 2026-10-04

## Status

Accepted

## Context

The two-tower model retrieves 200 candidates by one dot product. A second stage can use signals the retrieval vectors cannot: item popularity trends, item quality, how a candidate's genres and era match the user's taste, and how recently the user was active. It has to be cheap per request and trained without leaking the labels it is evaluated on.

## Options considered

1. No ranker (retrieval order).
2. Pointwise gradient boosting (binary relevance).
3. LambdaMART: gradient-boosted trees optimizing NDCG directly over each user's candidate list.
4. A neural ranker (DCN-v2), deferred to an optional stretch.

## Decision

Option 3 with LightGBM.

- Candidates: the trained two-tower model's top 200 unseen items per validation user (exact scoring), labeled by validation positives. `n_relevant` counts all of a user's relevant items, so NDCG stays relative to the full relevant set, like the retrieval evaluation.
- Features (19): retrieval score and rank; user and item statistics from the Feast offline store joined point-in-time at the training cutoff; item year and genre count; genre affinity (cosine between the user's positive-genre profile and the item's genres) and the gap from the user's mean release year, both from training history only. The same formulas run per request in NumPy at serving time, and a parity test pins the two together.
- Users are split so no reported number is selected on its own data. The retrieval model's early-stopping users (its validation users in the 10% sample) are dropped, the rest are split into ranker-train and ranker-eval halves, and 20% of ranker-train users are the inner validation set for early stopping and a 20-trial Optuna search (TPE, 15-minute cap).
- Importance: mean absolute SHAP values from LightGBM's built-in TreeSHAP (`pred_contrib=True`). The `shap` package does not build against NumPy 2.5.
- Serving runs the ranker through ONNX Runtime. Thresholds are rounded down to float32 before conversion so the ONNX model matches LightGBM exactly on float32 features (max difference 9.2e-7).

## Results (ranker-eval users, 2,908)

| Order | NDCG@10 [95% CI] |
| --- | ---: |
| Retrieval (two-tower) | 0.1155 [0.1088, 0.1222] |
| LambdaMART | 0.1504 [0.1435, 0.1576] |
| Lift (paired) | **+0.0349 [+0.0298, +0.0401]** |

The full two-stage system against EASE on the same users (paired): NDCG@10 0.1504 against 0.1471, difference +0.0033 [-0.0044, +0.0108]; Recall@10 0.1492 against 0.1454, difference +0.0038 [-0.0049, +0.0120]. The ranker closes the gap the retrieval stage leaves (ADR 0006); the system is level with the strongest baseline at the top of the list while adding real-time updates, cold-start handling, and millisecond serving.

Top features by mean |SHAP| (`docs/ranker-shap.png`): retrieval rank (0.242), 30-day item popularity (0.233), genre affinity (0.168), 7-day item popularity (0.134), item mean rating (0.114), year gap (0.100), days since the user's last activity (0.072). Recent popularity matters as much as the retrieval signal, consistent with the temporal drift seen in the baselines (recent popularity beats all-time popularity by 0.09 Recall@100).

Tuned setting: learning rate 0.054, 33 leaves, min 142 rows per leaf, feature fraction 0.57, bagging 0.65, L2 0.029, 248 trees (early stopped). Tuning and training took 198 s.

## Consequences

- `make train-ranker` exports candidates, builds features, trains, and converts to ONNX, each step in its own process (on macOS LightGBM's OpenMP runtime cannot share a process with PyTorch or FAISS).
- The ranker was trained on validation-period labels with candidates from the train-period retrieval model. The final test evaluation retrains retrieval on train plus validation and reuses this ranker (M11).

## Sources

- Burges, "From RankNet to LambdaRank to LambdaMART: An Overview", 2010: https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
- LightGBM lambdarank parameters: https://lightgbm.readthedocs.io/en/stable/Parameters.html
- LightGBM `pred_contrib` (TreeSHAP): https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html
- Lundberg et al., "From local explanations to global understanding with explainable AI for trees", 2020: https://www.nature.com/articles/s42256-019-0138-9
- Optuna TPE sampler: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
- onnxmltools LightGBM converter: https://github.com/onnx/onnxmltools
