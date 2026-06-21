"""Tests for the native h5py SNIRF reader.

Writes a minimal SNIRF file from scratch with h5py and reads it back, so the
test is fully self-contained (no external data).
"""

import numpy as np
import pytest

import cedanirs as cn

h5py = pytest.importorskip("h5py")


def _write_snirf(path, *, n_time=200, sfreq=10.0):
    """Write a tiny processed (HbO/HbR) SNIRF with 3 source-detector pairs."""
    pairs = [(1, 1), (1, 2), (2, 3)]
    rng = np.random.default_rng(0)
    t = np.arange(n_time) / sfreq
    # columns: for each pair, an HbO then an HbR channel (6 columns)
    cols, meta = [], []
    for (s, d) in pairs:
        cols.append(np.sin(2 * np.pi * 0.05 * t) + 0.1 * rng.standard_normal(n_time))
        meta.append((s, d, "HbO"))
        cols.append(-0.5 * np.sin(2 * np.pi * 0.05 * t) + 0.1 * rng.standard_normal(n_time))
        meta.append((s, d, "HbR"))
    data = np.column_stack(cols)  # (n_time, 6)

    sd = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("formatVersion", data="1.1", dtype=sd)
        nirs = f.create_group("nirs")
        md = nirs.create_group("metaDataTags")
        md.create_dataset("LengthUnit", data="mm", dtype=sd)
        probe = nirs.create_group("probe")
        probe.create_dataset("wavelengths", data=np.array([760.0, 850.0]))
        probe.create_dataset("sourcePos2D", data=np.array([[0.0, 0.0], [10.0, 0.0]]))
        probe.create_dataset("detectorPos2D",
                             data=np.array([[0, 5.0], [5, 5.0], [10, 5.0]]))
        d1 = nirs.create_group("data1")
        d1.create_dataset("time", data=t)
        d1.create_dataset("dataTimeSeries", data=data)
        for k, (s, d, label) in enumerate(meta, start=1):
            ml = d1.create_group(f"measurementList{k}")
            ml.create_dataset("sourceIndex", data=np.array([s], dtype="int32"))
            ml.create_dataset("detectorIndex", data=np.array([d], dtype="int32"))
            ml.create_dataset("dataType", data=np.array([99999], dtype="int32"))
            ml.create_dataset("dataTypeLabel", data=label, dtype=sd)
    return data, pairs


def test_read_snirf_structure(tmp_path):
    p = tmp_path / "mini.snirf"
    _write_snirf(p)
    snf = cn.read_snirf(str(p))
    assert snf.format_version == "1.1"
    assert len(snf.nirs) == 1
    block = snf.first.data_blocks[0]
    assert block.data.shape == (200, 6)
    assert len(block.measurements) == 6
    assert block.unique_pairs() == [(1, 1), (1, 2), (2, 3)]
    assert snf.first.metadata.get("LengthUnit") == "mm"
    assert list(snf.first.probe.wavelengths) == [760.0, 850.0]


def test_read_timeseries_groups_by_chromophore(tmp_path):
    p = tmp_path / "mini.snirf"
    data, pairs = _write_snirf(p, n_time=300, sfreq=10.0)
    ts = cn.read_timeseries(str(p))
    assert ts.has_chromophore
    assert ts.chromophores == ["HbO", "HbR"]
    assert ts.channels == ["S1_D1", "S1_D2", "S2_D3"]
    assert ts.n_channels == 3
    assert ts.n_times == 300
    assert ts.sfreq == pytest.approx(10.0)
    # HbO channel 0 equals the first written column
    np.testing.assert_allclose(ts.select("HbO").to_numpy()[0], data[:, 0])


def test_read_timeseries_feeds_connectivity(tmp_path):
    p = tmp_path / "mini.snirf"
    _write_snirf(p, n_time=400)
    res = cn.connectivity(cn.read_timeseries(str(p)), method="pearson")
    assert res.chromophores == ["HbO", "HbR"]
    m = res["HbO"]
    assert m.n == 3
    np.testing.assert_allclose(np.diag(m.values), 1.0)


def test_bandpass_option_runs(tmp_path):
    p = tmp_path / "mini.snirf"
    _write_snirf(p, n_time=600, sfreq=10.0)
    ts = cn.read_timeseries(str(p), bandpass=(0.01, 0.1))
    assert ts.n_times == 600
    assert np.all(np.isfinite(ts.to_numpy()))
