import numpy as np
import pytest
import xarray as xr

from cedanirs import Chromophore, DataError, NirsTimeSeries


def test_from_2d_numpy_defaults():
    arr = np.random.default_rng(0).standard_normal((3, 100))
    ts = NirsTimeSeries(arr, sfreq=10.0)
    assert ts.n_channels == 3
    assert ts.n_times == 100
    assert ts.sfreq == 10.0
    assert ts.channels == ["ch0", "ch1", "ch2"]
    assert not ts.has_chromophore
    np.testing.assert_array_equal(ts.to_numpy(), arr)


def test_channel_labels_and_chromophore():
    arr = np.zeros((2, 50))
    ts = NirsTimeSeries(arr, channels=["S1_D1", "S1_D2"], chromophore="HbO")
    assert ts.channels == ["S1_D1", "S1_D2"]
    assert ts.chromophores == ["HbO"]


def test_time_coords_from_sfreq():
    ts = NirsTimeSeries(np.zeros((1, 10)), sfreq=2.0)
    times = ts.data.coords["time"].values
    np.testing.assert_allclose(times, np.arange(10) / 2.0)


def test_3d_cube_has_chromophores():
    cube = np.zeros((2, 4, 100))
    ts = NirsTimeSeries(cube, chromophores=["HbO", "HbR"], channels=list("ABCD"))
    assert ts.has_chromophore
    assert ts.chromophores == ["HbO", "HbR"]
    assert ts.n_channels == 4


def test_select_chromophore_returns_2d_slice():
    cube = np.arange(2 * 3 * 20, dtype=float).reshape(2, 3, 20)
    ts = NirsTimeSeries(cube, chromophores=["HbO", "HbR"])
    hbo = ts.select("HbO")
    assert not hbo.has_chromophore
    assert hbo.chromophores == ["HbO"]
    np.testing.assert_array_equal(hbo.to_numpy(), cube[0])


def test_select_missing_chromophore_raises():
    cube = np.zeros((2, 3, 20))
    ts = NirsTimeSeries(cube, chromophores=["HbO", "HbR"])
    with pytest.raises(DataError):
        ts.select("HbT")


def test_bad_ndim_raises():
    with pytest.raises(DataError):
        NirsTimeSeries(np.zeros((2, 2, 2, 2)))


def test_too_few_samples_raises():
    with pytest.raises(DataError):
        NirsTimeSeries(np.zeros((3, 1)))


def test_wrong_label_count_raises():
    with pytest.raises(DataError):
        NirsTimeSeries(np.zeros((3, 10)), channels=["only", "two"])


def test_coerce_passthrough():
    ts = NirsTimeSeries(np.zeros((2, 10)))
    assert NirsTimeSeries.coerce(ts) is ts


def test_accepts_dataarray():
    da = xr.DataArray(
        np.zeros((2, 10)),
        dims=("channel", "time"),
        coords={"channel": ["x", "y"]},
    )
    ts = NirsTimeSeries(da)
    assert ts.channels == ["x", "y"]


def test_chromophore_coerce():
    assert Chromophore.coerce("hbo") is Chromophore.HBO
    assert Chromophore.coerce("HbR") is Chromophore.HBR
    assert Chromophore.coerce(None) is Chromophore.UNKNOWN
    assert Chromophore.coerce("nonsense") is Chromophore.UNKNOWN
