# Two-tower versus EASE by user segment

Validation users outside the retrieval early-stopping set (5,816 users). Recall@100 on the full catalog; the difference is two-tower minus EASE per user, with a 95% bootstrap interval.

Overall: two-tower 0.2562, EASE 0.2865, difference -0.0303 [-0.0368, -0.0235].

## Training history length (ratings)

| Segment | Users | Two-tower | EASE | Difference [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| 1-20 | 179 | 0.3658 | 0.3503 | 0.0155 [-0.0116, 0.0422] |
| 21-100 | 1,040 | 0.3421 | 0.3907 | -0.0486 [-0.0635, -0.0330] |
| 101-500 | 2,726 | 0.2535 | 0.3036 | -0.0501 [-0.0606, -0.0401] |
| >500 | 1,871 | 0.2019 | 0.1977 | 0.0042 [-0.0074, 0.0155] |

## Days since last activity before the cutoff

| Segment | Users | Two-tower | EASE | Difference [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| <=1 day | 223 | 0.1943 | 0.2594 | -0.0651 [-0.0875, -0.0438] |
| 2-30 days | 1,976 | 0.2187 | 0.2591 | -0.0403 [-0.0487, -0.0299] |
| 31-365 days | 2,562 | 0.2674 | 0.2998 | -0.0324 [-0.0427, -0.0209] |
| >1 year | 1,055 | 0.3120 | 0.3114 | 0.0006 [-0.0157, 0.0183] |
