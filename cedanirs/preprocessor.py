from __future__ import annotations

import xarray as xr
import cedalion.io as io
import cedalion.nirs.cw as cw
import cedalion.sigproc.frequency as frequency


class Preprocessor:
    """Configurable preprocessing pipeline for cedalion fNIRS data.

    Steps (in execution order):
    1. Raw intensity -> Optical density
    2. Motion correction (TDDR)
    3. Optical density -> Haemoglobin (Beer-Lambert)
    4. Bandpass temporal filtering

    Usage::

        pp = Preprocessor()
        pp.set_bandpass(0.01, 0.2)
        result = pp.apply(recording)

    Or fluent::

        result = Preprocessor().set_bandpass(0.01, 0.2).apply(recording)
    """

    def __init__(
        self,
        optical_density: bool = True,
        motion_correction: bool = True,
        hemoglobin: bool = True,
        bandpass: bool = True,
        bandpass_low: float = 0.01,
        bandpass_high: float = 0.1,
        dpf: float = 6.0,
    ):
        self.optical_density = optical_density
        self.motion_correction = motion_correction
        self.hemoglobin = hemoglobin
        self.bandpass = bandpass
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.dpf = dpf

    def set_optical_density(self, enabled: bool = True) -> "Preprocessor":
        self.optical_density = enabled
        return self

    def set_motion_correction(self, enabled: bool = True) -> "Preprocessor":
        self.motion_correction = enabled
        return self

    def set_hemoglobin(self, enabled: bool = True, dpf: float = 6.0) -> "Preprocessor":
        self.hemoglobin = enabled
        self.dpf = dpf
        return self

    def set_bandpass(self, low: float = 0.01, high: float = 0.1, enabled: bool = True) -> "Preprocessor":
        self.bandpass = enabled
        self.bandpass_low = low
        self.bandpass_high = high
        return self

    def apply(self, recording):
        """Apply the configured pipeline to a cedalion recording.

        Parameters
        ----------
        recording:
            A cedalion ``Recording`` or ``ContinuousData`` object.

        Returns the processed recording.
        """
        raise NotImplementedError(
            "Preprocessor.apply() is not yet implemented. "
            "Wire cedalion calls here based on the loaded recording type."
        )

    def print(self) -> None:
        print("Preprocessor Settings:")
        print(f"  optical_density:          {self.optical_density}")
        print(f"  motion_correction (TDDR): {self.motion_correction}")
        print(f"  hemoglobin (Beer-Lambert):{self.hemoglobin} (dpf={self.dpf})")
        print(f"  bandpass:                 {self.bandpass} ({self.bandpass_low}-{self.bandpass_high} Hz)")

    def __repr__(self) -> str:
        steps = []
        if self.optical_density:
            steps.append("OD")
        if self.motion_correction:
            steps.append("TDDR")
        if self.hemoglobin:
            steps.append("HbX")
        if self.bandpass:
            steps.append(f"BP({self.bandpass_low}-{self.bandpass_high})")
        return f"Preprocessor([{' -> '.join(steps)}])"
