# Baseline results

Evaluation: full split, `val` partition, 6,463 users with training history and at least one positive (7,368 more users with positives have no training history; they are served by the popularity fallback and are not in this table). Ranked against all 65,723 catalog items, excluding each user's training items. Cells show the mean with a 95% bootstrap interval over users (1000 resamples).

Hyperparameters were tuned on the sample10 split by recall@100. ARP@10 is the mean training popularity share of recommended items (lower means less popularity bias).

| Model | recall@10 | recall@50 | recall@100 | recall@200 | ndcg@10 | mrr | coverage@10 | ARP@10 | fit s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| popularity | 0.0665 [0.0632, 0.0701] | 0.0983 [0.0944, 0.1021] | 0.1411 [0.1368, 0.1455] | 0.1998 [0.1940, 0.2053] | 0.0695 [0.0660, 0.0731] | 0.1607 [0.1534, 0.1678] | 0.005 | 0.00218 | 0 |
| recent_popularity | 0.1150 [0.1106, 0.1197] | 0.1689 [0.1639, 0.1744] | 0.2341 [0.2281, 0.2403] | 0.3268 [0.3201, 0.3338] | 0.1155 [0.1114, 0.1199] | 0.2425 [0.2345, 0.2507] | 0.004 | 0.00136 | 0 |
| item_knn | 0.1066 [0.1024, 0.1109] | 0.1572 [0.1522, 0.1621] | 0.2311 [0.2252, 0.2370] | 0.3384 [0.3319, 0.3452] | 0.1087 [0.1044, 0.1130] | 0.2272 [0.2196, 0.2350] | 0.013 | 0.00151 | 36 |
| ease | 0.1464 [0.1415, 0.1515] | 0.2083 [0.2030, 0.2131] | 0.2854 [0.2795, 0.2912] | 0.3902 [0.3833, 0.3965] | 0.1484 [0.1432, 0.1534] | 0.2925 [0.2837, 0.3011] | 0.028 | 0.00105 | 54 |
| als | 0.1300 [0.1253, 0.1347] | 0.1897 [0.1846, 0.1948] | 0.2643 [0.2583, 0.2701] | 0.3670 [0.3607, 0.3737] | 0.1281 [0.1234, 0.1325] | 0.2541 [0.2458, 0.2621] | 0.031 | 0.00088 | 40 |

Tuned settings:

- popularity: `{"signal": "positive"}`
- recent_popularity: `{"signal": "all", "window_days": 30}`
- item_knn: `{"neighbors": 50, "shrink": 10.0, "signal": "positive"}`
- ease: `{"l2": 2000.0, "signal": "positive"}`
- als: `{"alpha": 10.0, "factors": 64, "regularization": 1.0, "signal": "positive"}`

Best baseline by recall@100: **ease**.
