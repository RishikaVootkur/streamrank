# Model card: StreamRank two-stage recommender

## Model details

- **System.** Two stages behind one API. A two-tower retrieval model returns 200 candidates from the full catalog, and a LambdaMART ranker orders them into the top 10. Unknown users get a popularity fallback.
- **Retrieval.** User tower: a 2-layer causal transformer (SASRec-style, dimension 128, 2 heads) over the last 200 events, each embedded from its item, its positive flag, and the time gap since the previous event. User statistics enter at the ranker instead of the user tower. Item tower: item ID, genres, release-year bucket, and tags. Trained with sampled softmax over in-batch and 2,048 random negatives, with log-Q correction by inclusion probability, temperature 0.1, and dropout 0.3, on Apple MPS. About 10 million parameters. Served as ONNX, with a FAISS HNSW index (M = 16, efSearch = 400).
- **Ranker.** LightGBM LambdaMART over 19 features: retrieval score and rank, user activity and rating statistics, item popularity over all time, 30 days, and 7 days, item age, rating statistics, genre affinity, and release-year gap. Hyperparameters were tuned with Optuna. Served as ONNX.
- **Versions.** The final retrieval model was trained on all interactions before 2022-02-01. The ranker was trained on validation-period labels. Code, configuration, data hashes, and seeds are recorded in each stage's `manifest.json` and in MLflow.
- **License.** MIT for the code. MovieLens terms apply to the data.

## Intended use

- A portfolio and teaching system that shows a production-style recommender end to end: temporal evaluation, feature store, streaming updates, low-latency serving, monitoring, and deployment.
- Movie recommendations for MovieLens-style explicit-rating data.

Not intended for real users without further work: no fairness review across user groups, no content safety filtering, and no consent or privacy handling beyond what the public dataset already has.

## Data

- **Source.** MovieLens 32M (GroupLens): 32 million ratings from 200,948 users on 87,585 movies, 1995 to 2023, plus genres, titles with years, and user tags.
- **Labels.** A rating of 4.0 or higher is a positive. Every rating counts as an interaction, so it is excluded from recommendation and enters the user's sequence.
- **Split.** Global temporal split. Train is before 2020-11-05; validation runs from 2020-11-05 to 2022-02-01; test is after that, with 5% of interactions in each of validation and test. Features are computed point in time. Users with no history before a cutoff are cold and get the popularity fallback; they are counted but not scored.
- **Development.** Models were tuned on a 10% user sample. Early stopping and ranker training used validation labels from disjoint user sets, so results are reported on users whose labels no stage used.

## Evaluation

- **Protocol.** Full-catalog ranking: every catalog item is a candidate, and items the user already rated are excluded. Metrics are Recall@K, NDCG@10, MRR, catalog coverage, and the average popularity of recommended items. 95% bootstrap intervals over users (1,000 resamples). A model counts as better only when a paired interval excludes zero.
- **Final run.** Retrieval was refitted on train + validation history for the epoch count that early stopping chose on validation. It was then scored on test labels, together with the unchanged ranker and EASE refitted on the same history.
- **Online proxy.** A team-draft interleaving replay of validation-period events, with the user state updated after every event.

### Results

See the [README results](../README.md#results) for the final test-period numbers and [results.md](results.md) for everything else.

## Limitations and risks

- **Popularity bias.** The ranker's strongest features after retrieval rank are recent popularity, and coverage of the catalog is low for every model. Niche movies are rarely recommended.
- **Two-tower against EASE.** The two-tower model trails EASE, a linear item-to-item model, on recall ([ADR 0006](adr/0006-two-tower-retrieval.md)). It is kept for serving reasons: index retrieval, updates from the latest events, and content-based embeddings for new items.
- **Offline against online.** The ranker's offline NDCG gain did not appear in the next-event replay ([ADR 0011](adr/0011-online-test-simulation.md)). Offline metrics here reward long-horizon relevance.
- **Explicit ratings as implicit feedback.** A rating means the user watched and chose to rate a movie. That is not the same as a click or a view, and rating behavior changed over the 28 years of data.
- **Session features.** Session features are computed online and parity-tested, but the ranker was trained on batch features only. The last few minutes of activity reach the recommendations through the user tower's updated sequence.
- **Drift.** The drift job flags item popularity shift between the 90 days before each cutoff, largely driven by lower overall rating volume ([drift.md](drift.md)).
- **Cold start.** New users get popular movies until they have history. New items get embeddings from content, but the ranker's popularity features are zero for them.
