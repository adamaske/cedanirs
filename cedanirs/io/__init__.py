"""Native I/O for cedanirs.

A dependency-light SNIRF reader built directly on :mod:`h5py` (SNIRF is an HDF5
container), ported from the NIRWizard Rust parser. It reads the full structure
-- metadata, probe, data blocks with their per-channel measurement lists, stim
and aux -- and converts a recording into a
:class:`~cedanirs.core.timeseries.NirsTimeSeries` for connectivity analysis.

For already-processed HbO/HbR files this is the preferred loader: it reads the
chromophores directly by ``dataTypeLabel``, with no MNE / Beer-Lambert step.

``h5py`` is an optional dependency (``pip install 'cedanirs[io]'``); it is
imported lazily so importing cedanirs never requires it.
"""

from __future__ import annotations

from .snirf import (
    DataBlock,
    Measurement,
    NirsElement,
    Probe,
    Snirf,
    read_snirf,
    read_timeseries,
)

__all__ = [
    "read_snirf",
    "read_timeseries",
    "Snirf",
    "NirsElement",
    "Probe",
    "DataBlock",
    "Measurement",
]
