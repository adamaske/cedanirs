"""Native SNIRF (.snirf) reader on top of h5py.

SNIRF is an HDF5 file with a defined layout. This module parses it directly --
no MNE, no cedalion -- mirroring the structure of the NIRWizard Rust parser
(``snirf_parser.rs`` / ``snirf.rs``):

    /formatVersion
    /nirs (or /nirs1, /nirs2, ...)
        metaDataTags/*               name -> value strings
        probe/                       wavelengths, source/detector positions
        data{j}/                     time, dataTimeSeries[time, channel]
            measurementList{k}/      sourceIndex, detectorIndex,
                                     wavelengthIndex?, dataType, dataTypeLabel?
            (or the compact measurementLists/ with parallel arrays)
        stim{i}/                     name, data[onset, duration, value]
        aux{i}/                      name, dataTimeSeries, time

The classes are intentionally close to the SNIRF spec. :meth:`Snirf.to_timeseries`
turns a recording into a :class:`~nirconn.core.timeseries.NirsTimeSeries`,
grouping channels by chromophore (``dataTypeLabel`` for HbO/HbR/HbT, else by
wavelength), which is what connectivity analysis consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..core.exceptions import DependencyError
from ..core.types import Chromophore

if TYPE_CHECKING:  # pragma: no cover
    import h5py

    from ..core.timeseries import NirsTimeSeries

# SNIRF measurementList.dataType codes (spec appendix).
DATA_TYPE_CW_AMPLITUDE = 1
DATA_TYPE_PROCESSED = 99999


def _h5py():
    try:
        import h5py

        return h5py
    except ImportError as exc:  # pragma: no cover - exercised only without h5py
        raise DependencyError("Reading SNIRF files", "h5py", extra="io") from exc


# --------------------------------------------------------------------------- #
# low-level readers (encoding/shape tolerant, like the Rust read_string/read_i32)
# --------------------------------------------------------------------------- #

def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _read_str(group, name, default=None):
    if name not in group:
        return default
    v = group[name][()]
    if isinstance(v, np.ndarray):
        v = v.ravel()
        if v.size == 0:
            return default
        v = v[0]
    return _decode(v).strip()


def _read_int(group, name, default=None):
    if name not in group:
        return default
    v = np.ravel(group[name][()])
    return int(v[0]) if v.size else default


def _read_float(group, name, default=None):
    if name not in group:
        return default
    v = np.ravel(group[name][()])
    return float(v[0]) if v.size else default


def _read_1d(group, name):
    if name not in group:
        return None
    return np.asarray(group[name][()], dtype=float).ravel()


def _read_2d(group, name):
    if name not in group:
        return None
    arr = np.asarray(group[name][()], dtype=float)
    return np.atleast_2d(arr)


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #

@dataclass
class Measurement:
    """One column of ``dataTimeSeries`` -- a single source-detector-chromophore."""

    source_index: int
    detector_index: int
    data_type: int
    data_type_label: str = ""
    wavelength_index: int | None = None
    data_type_index: int = 0
    data_unit: str | None = None

    @property
    def pair(self) -> tuple[int, int]:
        return (self.source_index, self.detector_index)

    @property
    def channel(self) -> str:
        return f"S{self.source_index}_D{self.detector_index}"


@dataclass
class DataBlock:
    time: np.ndarray
    data: np.ndarray  # (n_time, n_channels)
    measurements: list[Measurement]

    @property
    def sfreq(self) -> float | None:
        if self.time is None or self.time.size < 2:
            return None
        dt = float(np.median(np.diff(self.time)))
        return 1.0 / dt if dt > 0 else None

    def unique_pairs(self) -> list[tuple[int, int]]:
        seen, out = set(), []
        for m in self.measurements:
            if m.pair not in seen:
                seen.add(m.pair)
                out.append(m.pair)
        return out


@dataclass
class Probe:
    wavelengths: np.ndarray
    source_pos: np.ndarray | None = None  # (n_src, 2 or 3)
    detector_pos: np.ndarray | None = None
    source_labels: list[str] = field(default_factory=list)
    detector_labels: list[str] = field(default_factory=list)


@dataclass
class StimEvent:
    name: str
    data: np.ndarray  # (n_markers, >=3): onset, duration, value


@dataclass
class NirsElement:
    metadata: dict
    probe: Probe
    data_blocks: list[DataBlock]
    stim: list[StimEvent] = field(default_factory=list)


@dataclass
class Snirf:
    """A parsed SNIRF file."""

    format_version: str
    filepath: str
    nirs: list[NirsElement]

    @property
    def first(self) -> NirsElement:
        return self.nirs[0]

    def to_timeseries(self, *, block: int = 0, bandpass=None) -> "NirsTimeSeries":
        return _block_to_timeseries(
            self.first.data_blocks[block],
            wavelengths=self.first.probe.wavelengths,
            bandpass=bandpass,
        )

    def __repr__(self) -> str:
        e = self.first
        b = e.data_blocks[0] if e.data_blocks else None
        nt = b.data.shape[0] if b is not None else 0
        nc = len(b.unique_pairs()) if b is not None else 0
        return (
            f"Snirf(v{self.format_version}, {len(self.nirs)} nirs, "
            f"{nc} channels, {nt} samples, {len(e.probe.wavelengths)} wavelengths)"
        )


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def read_snirf(path: str) -> Snirf:
    """Parse a SNIRF file into a :class:`Snirf` object."""
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        version = _read_str(f, "formatVersion", "1.0")
        elements: list[NirsElement] = []
        if "nirs" in f:
            elements.append(_parse_nirs(f["nirs"]))
        else:
            i = 1
            while f"nirs{i}" in f:
                elements.append(_parse_nirs(f[f"nirs{i}"]))
                i += 1
        if not elements:
            from ..core.exceptions import DataError

            raise DataError(f"No /nirs group found in {path!r}.")
    return Snirf(format_version=version, filepath=str(path), nirs=elements)


def _parse_nirs(g: "h5py.Group") -> NirsElement:
    metadata = {}
    if "metaDataTags" in g:
        md = g["metaDataTags"]
        for name in md.keys():
            try:
                metadata[name] = _read_str(md, name)
            except Exception:
                metadata[name] = None

    probe = _parse_probe(g["probe"]) if "probe" in g else Probe(np.array([]))

    blocks = []
    j = 1
    while f"data{j}" in g:
        blocks.append(_parse_data_block(g[f"data{j}"]))
        j += 1
    if not blocks and "data" in g:  # some writers omit the index
        blocks.append(_parse_data_block(g["data"]))

    stim = []
    i = 1
    while f"stim{i}" in g:
        s = g[f"stim{i}"]
        name = _read_str(s, "name", f"stim{i}")
        data = _read_2d(s, "data")
        stim.append(StimEvent(name=name, data=data if data is not None else np.empty((0, 3))))
        i += 1

    return NirsElement(metadata=metadata, probe=probe, data_blocks=blocks, stim=stim)


def _parse_probe(p: "h5py.Group") -> Probe:
    wavelengths = _read_1d(p, "wavelengths")
    s2 = _read_2d(p, "sourcePos2D")
    s3 = _read_2d(p, "sourcePos3D")
    d2 = _read_2d(p, "detectorPos2D")
    d3 = _read_2d(p, "detectorPos3D")
    src = s3 if s3 is not None else s2
    det = d3 if d3 is not None else d2
    n_src = src.shape[0] if src is not None else 0
    n_det = det.shape[0] if det is not None else 0
    return Probe(
        wavelengths=wavelengths if wavelengths is not None else np.array([]),
        source_pos=src,
        detector_pos=det,
        source_labels=[f"S{i + 1}" for i in range(n_src)],
        detector_labels=[f"D{i + 1}" for i in range(n_det)],
    )


def _parse_data_block(d: "h5py.Group") -> DataBlock:
    from ..core.exceptions import DataError

    if "dataTimeSeries" not in d or "time" not in d:
        raise DataError("data block missing dataTimeSeries/time.")
    ts = np.asarray(d["dataTimeSeries"][()], dtype=float)
    ts = np.atleast_2d(ts)
    time = np.asarray(d["time"][()], dtype=float).ravel()

    n_cols = ts.shape[1]
    # SNIRF stores [time x channel]; if it looks transposed, fix it.
    if ts.shape[0] == 1 and ts.shape[1] == time.size:
        pass
    elif ts.shape[0] != time.size and ts.shape[1] == time.size:
        ts = ts.T
        n_cols = ts.shape[1]

    measurements = _parse_measurements(d, n_cols)
    return DataBlock(time=time, data=ts, measurements=measurements)


def _parse_measurements(d: "h5py.Group", n_cols: int) -> list[Measurement]:
    # Compact form: a single measurementLists group with parallel arrays.
    if "measurementLists" in d:
        ml = d["measurementLists"]
        si = np.ravel(ml["sourceIndex"][()]).astype(int)
        di = np.ravel(ml["detectorIndex"][()]).astype(int)
        dt = (np.ravel(ml["dataType"][()]).astype(int)
              if "dataType" in ml else np.zeros(len(si), int))
        wl = (np.ravel(ml["wavelengthIndex"][()]).astype(int)
              if "wavelengthIndex" in ml else None)
        labels = None
        if "dataTypeLabel" in ml:
            labels = [_decode(x) for x in np.ravel(ml["dataTypeLabel"][()])]
        out = []
        for k in range(len(si)):
            out.append(Measurement(
                source_index=int(si[k]), detector_index=int(di[k]),
                data_type=int(dt[k]) if k < len(dt) else 0,
                data_type_label=(labels[k] if labels else ""),
                wavelength_index=(int(wl[k]) if wl is not None else None),
            ))
        return out

    # Per-channel form: measurementList1, measurementList2, ...
    out = []
    for col in range(n_cols):
        name = f"measurementList{col + 1}"
        if name not in d:
            break
        g = d[name]
        out.append(Measurement(
            source_index=_read_int(g, "sourceIndex", 0),
            detector_index=_read_int(g, "detectorIndex", 0),
            data_type=_read_int(g, "dataType", 0),
            data_type_label=_read_str(g, "dataTypeLabel", "") or "",
            wavelength_index=_read_int(g, "wavelengthIndex", None),
            data_type_index=_read_int(g, "dataTypeIndex", 0) or 0,
            data_unit=_read_str(g, "dataUnit", None),
        ))
    return out


# --------------------------------------------------------------------------- #
# conversion to NirsTimeSeries
# --------------------------------------------------------------------------- #

def _chromophore_key(m: Measurement, wavelengths: np.ndarray) -> str:
    """Group key for a measurement: HbO/HbR/HbT label, else a wavelength tag."""
    label = (m.data_type_label or "").strip()
    if label:
        c = Chromophore.coerce(label)
        if c is not Chromophore.UNKNOWN:
            return str(c)
        return label  # keep custom processed label as-is
    if m.wavelength_index is not None and wavelengths.size:
        idx = m.wavelength_index - 1  # SNIRF is 1-based
        if 0 <= idx < wavelengths.size:
            return f"{wavelengths[idx]:.0f}nm"
    return str(Chromophore.RAW)


def _block_to_timeseries(block: DataBlock, *, wavelengths=None, bandpass=None) -> "NirsTimeSeries":
    from ..core.exceptions import DataError
    from ..core.timeseries import NirsTimeSeries

    wavelengths = np.asarray(wavelengths if wavelengths is not None else [], dtype=float)
    # group columns by chromophore, preserving channel order
    groups: dict[str, list[tuple[str, int]]] = {}
    for col, m in enumerate(block.measurements):
        key = _chromophore_key(m, wavelengths)
        groups.setdefault(key, []).append((m.channel, col))

    if not groups:
        raise DataError("Data block has no measurements to convert.")

    # common channel order = channels present in the first chromophore group
    chroms = list(groups.keys())
    # Build channel label union (ordered by first appearance across groups).
    channel_order: list[str] = []
    seen = set()
    for key in chroms:
        for ch, _ in groups[key]:
            if ch not in seen:
                seen.add(ch)
                channel_order.append(ch)

    n_time = block.data.shape[0]
    cube = np.full((len(chroms), len(channel_order), n_time), np.nan)
    ch_index = {ch: i for i, ch in enumerate(channel_order)}
    for ci, key in enumerate(chroms):
        for ch, col in groups[key]:
            cube[ci, ch_index[ch], :] = block.data[:, col]

    # squeeze to 2D if a single chromophore
    sfreq = block.sfreq
    if bandpass is not None and sfreq:
        cube = _bandpass(cube, sfreq, bandpass)

    if len(chroms) == 1:
        return NirsTimeSeries(cube[0], sfreq=sfreq, channels=channel_order,
                              chromophore=chroms[0])
    return NirsTimeSeries(cube, sfreq=sfreq, channels=channel_order,
                          chromophores=chroms)


def _bandpass(cube: np.ndarray, sfreq: float, band) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    lo, hi = band
    nyq = sfreq / 2.0
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999999)
    b, a = butter(4, [lo_n, hi_n], btype="band")
    out = cube.copy()
    flat = out.reshape(-1, out.shape[-1])
    for i in range(flat.shape[0]):
        row = flat[i]
        if np.all(np.isfinite(row)):
            flat[i] = filtfilt(b, a, row)
    return flat.reshape(out.shape)


def read_timeseries(path: str, *, block: int = 0, bandpass=None) -> "NirsTimeSeries":
    """Read a SNIRF file straight into a :class:`NirsTimeSeries`.

    Channels are grouped by chromophore (``dataTypeLabel`` for HbO/HbR/HbT, else
    by wavelength). ``bandpass=(low, high)`` applies a 4th-order Butterworth
    band-pass (e.g. ``(0.01, 0.1)`` for resting-state connectivity).
    """
    return read_snirf(path).to_timeseries(block=block, bandpass=bandpass)
