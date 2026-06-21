import logging
from pathlib import Path

import numpy as np
import pytest

# Force a headless matplotlib backend before anything imports pyplot, so the
# visualisation tests run without a display.
import matplotlib

matplotlib.use("Agg")

# Ensure logs/ exists before pytest opens its log file.
Path("logs").mkdir(exist_ok=True)


@pytest.fixture
def log(request):
    """Per-test logger named after the test function."""
    return logging.getLogger(request.node.name)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def correlated_data(rng):
    """4 channels with a known correlation structure, length 600.

    - A and B share a latent signal (strong positive correlation).
    - C is independent noise.
    - D is the negation of the latent signal (strong negative correlation).
    """
    n = 600
    latent = rng.standard_normal(n)
    a = latent + 0.05 * rng.standard_normal(n)
    b = latent + 0.05 * rng.standard_normal(n)
    c = rng.standard_normal(n)
    d = -latent + 0.05 * rng.standard_normal(n)
    data = np.vstack([a, b, c, d])
    labels = ["A", "B", "C", "D"]
    return data, labels


@pytest.fixture
def chromophore_cube(correlated_data, rng):
    """3-D (chromophore, channel, time) cube with HbO and HbR."""
    data, labels = correlated_data
    hbo = data
    hbr = -0.5 * data + 0.1 * rng.standard_normal(data.shape)
    cube = np.stack([hbo, hbr])
    return cube, labels
