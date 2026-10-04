# 2. Global temporal split and evaluation population

Date: 2026-10-04

## Status

Accepted

## Context

Offline results must predict how the recommender would do when deployed: trained on the past and asked about the future. MovieLens 32M has 32,000,204 ratings from 1995-01-09 to 2023-10-13 (UTC). Rating volume per year varies widely (about 0.5M in 2014, 1.9M in 2016, 0.8M in 2023).

## Options considered

1. Random split of interactions. Trains on the future; inflates every metric.
2. Leave-one-out per user (last item per user is test). Common in sequential recommendation papers, but each user's cut is at a different time, so training contains other users' interactions after a given test moment. This changes model rankings (Ji et al.; Sun).
3. Per-user temporal split (last N% of each user). Same leak as option 2.
4. One global time cut for test, with validation taken from the window just before it.

## Decision

Option 4. T2 is the timestamp quantile that leaves the last 5% of interactions as test; T1 leaves the 5% before it as validation. Quantiles (not fixed dates) keep the same code valid on synthetic data. On MovieLens 32M this gives T1 = 2020-11-05 and T2 = 2022-02-01 (see `docs/data-split.md`).

- Train: ts < T1. Validation: T1 <= ts < T2. Test: ts >= T2. `check_no_leakage` enforces strict time order and no duplicated interaction, runs on every split, and is covered by tests that inject future rows.
- Label: rating >= 4.0 is positive. The rating stays available as a feature.
- Model selection and tuning use validation only. Final models are refit on train + validation and scored once on test.
- Evaluation population: users with at least one training interaction ("warm" users) are scored with full-catalog ranking, excluding items they interacted with before the cut. Users with no history are served by the popularity fallback and reported separately, so the share of cold traffic is visible (53% of validation users, 62% of test users).
- Items first seen after the cut cannot be retrieved by ID-only models; their count is reported (8,426 in validation). Content features in the item tower address this.
- Iteration uses a deterministic 10% user sample (stable multiplicative hash of the user ID), with identical cut points. Full data is used for final runs.
- Any statistic used as a feature (popularity, item averages) is computed only from data before the relevant cut.

## Consequences

- Fewer evaluable users than leave-one-out (6,742 warm validation users on full data, 673 in the 10% sample), so confidence intervals are wider and are always reported.
- The split matches deployment: no model sees the future.

## Sources

- Meng, McCreadie, Macdonald, Ounis. "Exploring Data Splitting Strategies for the Evaluation of Recommendation Models." RecSys 2020. https://arxiv.org/abs/2007.13237
- Ji, Sun, Zhang, Li. "A Critical Study on Data Leakage in Recommender System Offline Evaluation." TOIS 41(3), 2023. https://arxiv.org/abs/2010.11060
- Sun. "Take a Fresh Look at Recommender Systems from an Evaluation Standpoint." SIGIR 2023. https://arxiv.org/abs/2210.04149
- MovieLens 32M README. https://files.grouplens.org/datasets/movielens/ml-32m-README.html
- Pandera polars integration. https://pandera.readthedocs.io/en/stable/polars.html
