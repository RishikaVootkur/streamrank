# 11. Online test simulation with team-draft interleaving

Date: 2026-10-04

## Status

Accepted

## Context

Offline metrics score a fixed snapshot: every validation positive, ranked once at the cutoff. A deployed recommender is asked again after every event, with state that keeps moving. Before trusting the ranker's offline lift (ADR 0008), it should win when replayed the way users would meet it.

## Options considered

1. A/B split of replayed users. Needs many more users for the same power.
2. Team-draft interleaving (Radlinski, Kurup, Joachims 2008): both systems contribute to one merged list per impression, a coin decides who picks first in each round, and a click is credited to the system that placed the item. Much more sensitive per impression than an A/B split.

## Decision

Option 2, as a replay simulator (`make simulate`).

- Users: 2,000 sampled from the ranker's held-out evaluation users, so neither model trained or tuned on their validation labels. State in a separate Redis database, bulk-loaded from training history.
- Replay: each user's validation events in time order (canonical replay order). Before each positive event, both systems produce top-10 lists from the state at that moment (at most 20 impressions per user), and the lists are interleaved. A click is that event's item appearing in the merged list. Every event, positive or not, then advances the user's sequence and seen set, as the streaming job does.
- Outcome per user: (credited B - credited A) / clicks, averaged over users with at least one click, with a 95% bootstrap interval over users. B wins if the interval is above zero.
- Three runs: retrieval order (A) against retrieval + ranker (B), with batch features frozen at the cutoff (as served); the same pair with batch features refreshed every simulated day from the point-in-time snapshots (a daily pipeline); and recent popularity (A) against retrieval order (B) as a sanity check of the method's power.

## Results

23,197 impressions over 2,000 users in each run:

| A | B | Batch features | Users with clicks | Clicks A / B | Preference for B [95% CI] | Winner |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Retrieval | Retrieval + ranker | Frozen at cutoff | 1,140 | 1,343 / 1,191 | +0.018 [-0.030, +0.065] | tie |
| Retrieval | Retrieval + ranker | Refreshed daily | 1,109 | 1,368 / 1,121 | -0.027 [-0.072, +0.020] | tie |
| Recent popularity | Retrieval | Frozen at cutoff | 1,025 | 650 / 1,518 | **+0.371 [+0.324, +0.420]** | **retrieval** |

- The method has power: the two-tower model beats recent popularity on 70% of credited clicks, an interval far from zero.
- The ranker's offline lift does not carry over to next-event replay: retrieval order and the ranker tie (hit rate@10 0.104 against 0.101 frozen, 0.099 daily). Refreshing batch features daily does not change that, so stale features are not the explanation.
- Likely explanation: an objective mismatch. The ranker was trained to rank everything a user rates over the following 15 months (the offline NDCG target), which may favor broadly popular, well-rated films (its top features are 30- and 7-day popularity and item mean rating). The replay credits only the very next positive event, which is exactly what the sequential two-tower model was trained to predict. The offline evaluation measures long-horizon relevance; this test measures next-step relevance.

## Consequences

- Shipping decision under this evidence: the ranker improves long-horizon relevance (offline NDCG@10 +0.035) without hurting next-event clicks (tie), so it stays, but its claimed online benefit is unproven.
- A next iteration would train the ranker on next-event labels (each candidate list built from the state just before an event), so its training target matches what an online test measures.

## Sources

- Radlinski, Kurup, Joachims, "How Does Clickthrough Data Reflect Retrieval Quality?", CIKM 2008: https://doi.org/10.1145/1458082.1458092
- Chapelle, Joachims, Radlinski, Yue, "Large-scale validation and analysis of interleaved search evaluation", TOIS 2012: https://doi.org/10.1145/2094072.2094078
- Hofmann, Li, Radlinski, "Online Evaluation for Information Retrieval", 2016: https://doi.org/10.1561/1500000051
