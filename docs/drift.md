# Data drift report

Reference: rating events from 2020-08-07 to 2020-11-05 (the 90 days before the training cutoff). Current: 2021-11-03 to 2022-02-01. 50,000 sampled events per window. Evidently `DataDriftPreset` with its default tests for samples above 1,000 rows. Full report: `make drift` writes `artifacts/drift/drift_report.html`.

Drifted columns: 1 of 8 (share 0.125).

| Column | Method | Score | Threshold | Drift |
| --- | --- | ---: | ---: | :---: |
| rating | Wasserstein distance (normed) | 0.0693 | 0.1 | no |
| hour_utc | Wasserstein distance (normed) | 0.0562 | 0.1 | no |
| item_log_n_ratings_30d | Wasserstein distance (normed) | 0.3745 | 0.1 | yes |
| item_age_days | Wasserstein distance (normed) | 0.0672 | 0.1 | no |
| item_year | Wasserstein distance (normed) | 0.0641 | 0.1 | no |
| positive | Jensen-Shannon distance | 0.0297 | 0.1 | no |
| weekday | Jensen-Shannon distance | 0.0621 | 0.1 | no |
| item_genre | Jensen-Shannon distance | 0.0297 | 0.1 | no |
