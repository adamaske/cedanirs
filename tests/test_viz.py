import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import cedanirs as cn
from cedanirs.viz import plot_matrix


@pytest.fixture
def matrix(correlated_data):
    data, labels = correlated_data
    return cn.connectivity(data, channels=labels).matrix


def test_plot_returns_axes(matrix):
    ax = matrix.plot()
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_on_existing_axes(matrix):
    fig, ax = plt.subplots()
    out = plot_matrix(matrix, ax=ax)
    assert out is ax
    plt.close(fig)


def test_plot_with_mask(matrix):
    mask = matrix.significant(0.05)
    ax = matrix.plot(mask=~mask)  # hide non-significant
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_plain_array():
    ax = plot_matrix(np.eye(3))
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_non_square_raises():
    with pytest.raises(ValueError):
        plot_matrix(np.zeros((3, 4)))
