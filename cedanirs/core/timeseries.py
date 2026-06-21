"""The :class:`NirsTimeSeries` container.

This is the single input type every estimator understands. It is a thin,
labelled wrapper around an :class:`xarray.DataArray` so that channel names,
sampling rate and chromophore identity travel *with* the numbers instead of
being passed around as loose side-channel arguments.

Two layouts are supported:

* **2-D** ``(channel, time)`` -- a single chromophore.
* **3-D** ``(chromophore, channel, time)`` -- several chromophores stacked,
  the natural fNIRS case where each channel carries both HbO and HbR.

Estimators never branch on the layout themselves; the
:class:`~cedanirs.estimators.base.ConnectivityEstimator` base class iterates
over chromophores and hands each estimator a plain 2-D ``(channel, time)``
array.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

from .exceptions import DataError
from .types import Chromophore

CHANNEL_DIM = "channel"
TIME_DIM = "time"
CHROMO_DIM = "chromophore"


class NirsTimeSeries:
    """A labelled fNIRS time-series cube.

    Parameters
    ----------
    data:
        The signal. Accepts a NumPy array (2-D ``(channel, time)`` or 3-D
        ``(chromophore, channel, time)``), an existing :class:`xarray.DataArray`
        with the appropriate dims, or another :class:`NirsTimeSeries` (returned
        as-is by :meth:`coerce`).
    sfreq:
        Sampling frequency in Hz. Optional for time-domain methods such as
        Pearson correlation, but required by frequency-domain estimators.
    channels:
        Channel labels. Defaults to ``["ch0", "ch1", ...]``.
    chromophores:
        Labels for the leading dimension when ``data`` is 3-D. Defaults to
        ``HbO``/``HbR``/... as appropriate.
    chromophore:
        For 2-D data, the single chromophore the signal represents.
    times:
        Explicit time coordinates. Defaults to ``arange(n_times) / sfreq`` when
        ``sfreq`` is known, else a plain integer index.
    name:
        Optional human-readable label (e.g. subject/run id).
    """

    __slots__ = ("_da", "_sfreq", "_name")

    def __init__(
        self,
        data,
        *,
        sfreq: float | None = None,
        channels: Sequence[str] | None = None,
        chromophores: Sequence[str | Chromophore] | None = None,
        chromophore: str | Chromophore | None = None,
        times: Sequence[float] | None = None,
        name: str | None = None,
    ):
        if isinstance(data, NirsTimeSeries):
            # Copy the wrapped array; allow overriding metadata.
            self._da = data._da.copy()
            self._sfreq = sfreq if sfreq is not None else data._sfreq
            self._name = name if name is not None else data._name
            return

        if isinstance(data, xr.DataArray):
            da = self._validate_dataarray(data)
        else:
            da = self._build_dataarray(
                np.asarray(data),
                channels=channels,
                chromophores=chromophores,
                chromophore=chromophore,
                times=times,
                sfreq=sfreq,
            )

        self._da = da
        self._sfreq = float(sfreq) if sfreq is not None else da.attrs.get("sfreq")
        self._name = name

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _build_dataarray(
        arr: np.ndarray,
        *,
        channels,
        chromophores,
        chromophore,
        times,
        sfreq,
    ) -> xr.DataArray:
        if arr.ndim == 2:
            n_channels, n_times = arr.shape
            dims = (CHANNEL_DIM, TIME_DIM)
        elif arr.ndim == 3:
            n_chromo, n_channels, n_times = arr.shape
            dims = (CHROMO_DIM, CHANNEL_DIM, TIME_DIM)
        else:
            raise DataError(
                f"NirsTimeSeries expects a 2-D (channel, time) or 3-D "
                f"(chromophore, channel, time) array, got shape {arr.shape}."
            )

        if n_times < 2:
            raise DataError(
                f"Need at least 2 time samples to estimate connectivity, "
                f"got {n_times}."
            )

        if channels is None:
            channels = [f"ch{i}" for i in range(n_channels)]
        elif len(channels) != n_channels:
            raise DataError(
                f"Got {len(channels)} channel labels for {n_channels} channels."
            )

        coords: dict = {CHANNEL_DIM: list(channels)}

        if times is None:
            if sfreq:
                coords[TIME_DIM] = np.arange(n_times) / float(sfreq)
            else:
                coords[TIME_DIM] = np.arange(n_times)
        else:
            if len(times) != n_times:
                raise DataError(
                    f"Got {len(times)} time coordinates for {n_times} samples."
                )
            coords[TIME_DIM] = np.asarray(times)

        if arr.ndim == 3:
            if chromophores is None:
                default = [Chromophore.HBO, Chromophore.HBR, Chromophore.HBT]
                chromophores = [str(default[i]) if i < len(default) else f"chromo{i}"
                                for i in range(n_chromo)]
            elif len(chromophores) != n_chromo:
                raise DataError(
                    f"Got {len(chromophores)} chromophore labels for "
                    f"{n_chromo} chromophores."
                )
            coords[CHROMO_DIM] = [str(Chromophore.coerce(c)) for c in chromophores]

        attrs: dict = {}
        if sfreq:
            attrs["sfreq"] = float(sfreq)
        if arr.ndim == 2:
            attrs["chromophore"] = str(Chromophore.coerce(chromophore))

        return xr.DataArray(arr, dims=dims, coords=coords, attrs=attrs)

    @staticmethod
    def _validate_dataarray(da: xr.DataArray) -> xr.DataArray:
        if CHANNEL_DIM not in da.dims or TIME_DIM not in da.dims:
            raise DataError(
                f"DataArray must have '{CHANNEL_DIM}' and '{TIME_DIM}' dims, "
                f"got {da.dims}."
            )
        if da.ndim == 3 and CHROMO_DIM not in da.dims:
            raise DataError(
                f"3-D DataArray must have a '{CHROMO_DIM}' dim, got {da.dims}."
            )
        if CHANNEL_DIM not in da.coords:
            da = da.assign_coords(
                {CHANNEL_DIM: [f"ch{i}" for i in range(da.sizes[CHANNEL_DIM])]}
            )
        return da

    @classmethod
    def coerce(cls, data, **kwargs) -> "NirsTimeSeries":
        """Return ``data`` as a :class:`NirsTimeSeries`, wrapping if needed."""
        if isinstance(data, cls) and not kwargs:
            return data
        return cls(data, **kwargs)

    # -- properties ----------------------------------------------------------

    @property
    def data(self) -> xr.DataArray:
        """The underlying :class:`xarray.DataArray`."""
        return self._da

    @property
    def sfreq(self) -> float | None:
        """Sampling frequency in Hz, or ``None`` if unknown."""
        return self._sfreq

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def channels(self) -> list[str]:
        return [str(c) for c in self._da.coords[CHANNEL_DIM].values]

    @property
    def n_channels(self) -> int:
        return int(self._da.sizes[CHANNEL_DIM])

    @property
    def n_times(self) -> int:
        return int(self._da.sizes[TIME_DIM])

    @property
    def has_chromophore(self) -> bool:
        """True when the cube stacks several chromophores (3-D)."""
        return CHROMO_DIM in self._da.dims

    @property
    def chromophores(self) -> list[str]:
        """The chromophore labels (3-D), or the single label (2-D) as a list."""
        if self.has_chromophore:
            return [str(c) for c in self._da.coords[CHROMO_DIM].values]
        return [self._da.attrs.get("chromophore", str(Chromophore.UNKNOWN))]

    # -- access --------------------------------------------------------------

    def select(self, chromophore: str | Chromophore) -> "NirsTimeSeries":
        """Return the 2-D ``(channel, time)`` slice for one chromophore."""
        label = str(Chromophore.coerce(chromophore))
        if not self.has_chromophore:
            if label != str(Chromophore.UNKNOWN) and label not in self.chromophores:
                raise DataError(
                    f"Time series has no chromophore {label!r} "
                    f"(it is {self.chromophores[0]!r})."
                )
            return self
        if label not in self.chromophores:
            raise DataError(
                f"Chromophore {label!r} not present; have {self.chromophores}."
            )
        sub = self._da.sel({CHROMO_DIM: label})
        sub.attrs = {**self._da.attrs, "chromophore": label}
        return NirsTimeSeries(sub, sfreq=self._sfreq, name=self._name)

    def to_numpy(self) -> np.ndarray:
        """The raw signal as a NumPy array (same layout as construction)."""
        return np.asarray(self._da.values)

    def __array__(self, dtype=None):  # pragma: no cover - numpy protocol
        return np.asarray(self._da.values, dtype=dtype)

    def __repr__(self) -> str:
        bits = [f"{self.n_channels} ch", f"{self.n_times} samples"]
        if self._sfreq:
            bits.append(f"{self._sfreq:g} Hz")
        if self.has_chromophore:
            bits.append("/".join(self.chromophores))
        else:
            bits.append(self.chromophores[0])
        if self._name:
            bits.insert(0, repr(self._name))
        return f"NirsTimeSeries({', '.join(bits)})"
