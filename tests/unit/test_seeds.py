import random

import numpy as np

from streamrank.common.seeds import set_global_seed


def test_same_seed_gives_same_draws() -> None:
    rng_a = set_global_seed(7)
    draws_a = (random.random(), np.random.rand(), rng_a.random())
    rng_b = set_global_seed(7)
    draws_b = (random.random(), np.random.rand(), rng_b.random())
    assert draws_a == draws_b


def test_different_seeds_differ() -> None:
    assert set_global_seed(1).random() != set_global_seed(2).random()
