"""Seed control so runs are reproducible."""

import os
import random

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy global state and return a fresh NumPy generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global state used by some libraries
    return np.random.default_rng(seed)
