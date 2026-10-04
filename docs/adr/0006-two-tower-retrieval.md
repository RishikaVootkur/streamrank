# 6. Two-tower retrieval with a sequential user tower

Date: 2026-10-04

## Status

Accepted

## Context

The retrieval stage must return 200 candidates from 65,723 items in a few milliseconds, use both long-term history and recent behavior, and embed items it has rarely seen. The bar is EASE (ADR 0003).

## Options considered

1. Matrix factorization (ALS) or EASE served directly. Strong offline, but no sequence awareness, no content features for new items, and EASE's item-item matrix does not map to a nearest-neighbor index.
2. A pure sequential model (SASRec) with a full softmax over items. Strong in the literature, but the output layer is ID-only.
3. A two-tower model: a SASRec-style causal transformer user tower and an item tower with ID plus content, trained with sampled softmax, mixed negatives, and log-Q correction. Its vectors go into an approximate nearest-neighbor index.

## Decision

Option 3.

- User tower: 2 pre-norm transformer blocks (dimension 128, 2 heads) over the last 200 events. Each token sums a content-aware item embedding (shared with the item tower), a positive or negative rating flag, a log2 time-gap bucket, and a position embedding. Attention uses an explicit float mask in fp32 (MPS issues with `is_causal` and half precision).
- Item tower: ID embedding plus genre multi-hot projection, 5-year release bucket, and the mean of the item's top tags (tags only from before the cutoff), then a residual MLP. Both towers are L2-normalized and scored by dot product divided by a temperature.
- Loss: sampled softmax over the unique target items in the batch plus 2,048 uniform random items. Log-Q subtracts the log probability that an item is among the de-duplicated candidates, `1 - (1-q)^P (1-1/|C|)^R`. The expected count `P*q` over-corrects the most popular items by up to about 4 nats. Random draws equal to the row's positive are masked. Loss only on positive next events, while all ratings stay inputs.
- Training windows: with probability 0.5 a window ends at the user's latest event, otherwise anywhere in the history. Uniform windows mostly train on transitions from earlier years, while evaluation asks what comes after the cutoff. On the 10% sample this alone raised held-out Recall@100 from 0.190 to 0.214.
- Same-second ties are ordered by a deterministic hash, not item index, because MovieLens has many bulk ratings in one second.
- Evaluation: the shared full-catalog path. Early stopping uses the 10% sample's validation users, and results are reported on the other 90%, the same users the baseline tuning never saw. The EASE comparison is a paired bootstrap on those users.
- Tuning on the 10% sample (held-out half of its validation users, paired differences against EASE on the same users). Temperature 0.1 beat 0.05 and 0.2; dimension 64 was worse than 128; dropout 0.3 matched 0.2 at temperature 0.1 and was kept for the larger full run. Full table in PR #35.
- Full data: about 9 minutes per epoch on MPS (3.6 GB device memory with a periodic cache release; without it the allocator grew past 19 GB). Early stopping chose epoch 5 of 8.

## Results (full split, validation)

Users outside the 10% sample, never used for early stopping or for baseline tuning (5,816 users; paired bootstrap against EASE on the same users):

| Model | Recall@10 | Recall@50 | Recall@100 | Recall@200 | NDCG@10 | Coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-tower | 0.1180 | 0.1815 | 0.2562 [0.2494, 0.2628] | 0.3604 | 0.1161 | 0.032 |
| EASE | 0.1460 | 0.2085 | 0.2865 | 0.3907 | 0.1478 | 0.027 |

Two-tower minus EASE, Recall@100: **-0.0303 [-0.0368, -0.0235]**. The two-tower model does not beat the best baseline.

### Ablations (10% sample, held-out half, 324 users)

| Variant | Recall@100 [95% CI] | minus EASE [95% CI] |
| --- | ---: | ---: |
| Full model | 0.2306 [0.2049, 0.2572] | -0.021 [-0.044, 0.003] |
| No sequence encoder (causal mean of inputs) | 0.1913 [0.1688, 0.2153] | -0.060 [-0.083, -0.037] |
| No content features (ID only) | 0.1834 [0.1619, 0.2088] | -0.068 [-0.093, -0.042] |
| No log-Q correction | 0.1792 [0.1560, 0.2029] | -0.072 [-0.098, -0.047] |
| Uniform training windows (no recent bias) | 0.1896 [0.1662, 0.2128] | -0.062 [-0.086, -0.039] |

Every component contributes: removing any one costs 0.04 to 0.05 Recall@100.

## Gap analysis

Per-user comparison on the 5,816 held-out users (`docs/two-tower-segments.md`):

- History length: the two-tower model ties EASE for users with 1 to 20 ratings (+0.016 [-0.012, 0.042]) and for users with more than 500 (+0.004 [-0.007, 0.016]). It loses for 21 to 500 ratings (about -0.05), where item co-occurrence is most informative and EASE is strongest.
- Recency: the gap is largest for users active within a day of the cutoff (-0.065 [-0.088, -0.044]) and vanishes for users inactive over a year (+0.001 [-0.016, 0.018]). Removing log-Q makes the model worse overall, so the popularity correction is not the cause. A more likely cause is that EASE scores items by co-occurrence with the user's entire history, while the user tower compresses the last 200 events into one 128-dimensional vector, and one dot product cannot express EASE's item-item interactions.
- Context: EASE's strength on MovieLens matches the literature (Steck 2019; Ferrari Dacrema et al. 2019). EASE cannot serve this system as is. Its 20,000 x 20,000 weight matrix (1.6 GB as float32) covers only the most popular items, cannot embed new items, and has no nearest-neighbor index for sub-millisecond retrieval. The two-tower model provides those properties, and the ranker recovers the ranking quality (M6).

## Consequences

- The two-tower model stays as the retrieval stage, with this gap documented (roadmap "Blocked or changed").
- Recall@200, the candidate set the ranker sees, is 0.360 against EASE's 0.391, so the ranker works from a slightly weaker pool.
- Next steps if time allows: an item-item co-occurrence candidate source unioned with the two-tower candidates, or more user-tower capacity (multiple interest vectors).

## Sources

- SASRec, Kang and McAuley 2018: https://arxiv.org/abs/1808.09781
- Klenitskiy and Vasilev, "Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?", RecSys 2023: https://arxiv.org/abs/2309.07602
- Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations", RecSys 2019: https://dl.acm.org/doi/10.1145/3298689.3346996
- Yang et al., "Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations", WWW 2020: https://doi.org/10.1145/3366424.3386195
- "Correcting the LogQ Correction", RecSys 2025: https://arxiv.org/abs/2507.09331
- Steck, "Embarrassingly Shallow Autoencoders for Sparse Data", WWW 2019: https://arxiv.org/abs/1905.03375
- Ferrari Dacrema, Cremonesi, Jannach, "Are We Really Making Much Progress?", RecSys 2019: https://arxiv.org/abs/1907.06902
- PyTorch MPS environment variables and known issues: https://docs.pytorch.org/docs/2.14/mps_environment_variables.html
