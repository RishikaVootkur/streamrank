# Data drift report

Reference: rating events from 2020-08-07 to 2020-11-05 (the 90 days before the training cutoff). Current: 2021-11-03 to 2022-02-01. Sampled events: 50,000 reference, 50,000 current. Evidently `DataDriftPreset` with Wasserstein distance for numeric columns and Jensen-Shannon distance for categorical ones. Full report: `make drift` writes `artifacts/drift/drift_report.html`.

Drifted columns: 1 of 8 (share 0.125).

| Column | Method | Score | Threshold | Drift |
| --- | --- | ---: | ---: | :---: |
| rating | wasserstein | 0.0772 | 0.1 | no |
| hour_utc | wasserstein | 0.0546 | 0.1 | no |
| item_log_n_ratings_30d | wasserstein | 0.3675 | 0.1 | yes |
| item_age_days | wasserstein | 0.0654 | 0.1 | no |
| item_year | wasserstein | 0.0645 | 0.1 | no |
| positive | jensenshannon | 0.0281 | 0.1 | no |
| weekday | jensenshannon | 0.0660 | 0.1 | no |
| item_genre | jensenshannon | 0.0279 | 0.1 | no |

Event volume: 390,925 reference and 261,454 current events. Item popularity is an absolute 30-day count, so a change in overall volume shifts its whole distribution; check volume before reading popularity drift as a change in what people watch.
