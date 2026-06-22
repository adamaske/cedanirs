"""Configurable fNIRS preprocessing pipeline.

The :class:`Preprocessor` is a fluent builder describing *what* preprocessing to
apply; :meth:`Preprocessor.apply` executes it against a recording using a
pluggable backend. The cedalion backend is imported lazily inside
:meth:`apply`, so importing nirconn (or running connectivity on already-clean
data) never requires cedalion to be installed.
"""

from __future__ import annotations

from ..core.exceptions import DependencyError


class Preprocessor:
    """Configurable preprocessing pipeline for fNIRS recordings.

    Steps (in execution order):

    1. Raw intensity → optical density
    2. Motion correction (TDDR)
    3. Optical density → haemoglobin (modified Beer-Lambert law)
    4. Band-pass temporal filtering

    Usage::

        pp = Preprocessor()
        pp.set_bandpass(0.01, 0.2)
        result = pp.apply(recording)

    or fluently::

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

    def set_bandpass(
        self, low: float = 0.01, high: float = 0.1, enabled: bool = True
    ) -> "Preprocessor":
        self.bandpass = enabled
        self.bandpass_low = low
        self.bandpass_high = high
        return self

    def apply(self, recording):
        """Apply the configured pipeline to a recording.

        Parameters
        ----------
        recording:
            A cedalion ``Recording`` / ``ContinuousData`` object.

        Returns the processed recording.
        """
        try:
            import cedalion  # noqa: F401
        except ImportError as exc:
            raise DependencyError(
                "The cedalion preprocessing backend", "cedalion", extra="cedalion"
            ) from exc

        raise NotImplementedError(
            "Preprocessor.apply() is not yet wired to the cedalion backend. "
            "Connectivity estimation operates on already-preprocessed time "
            "series in the meantime."
        )

    def print(self) -> None:
        print("Preprocessor settings:")
        print(f"  optical_density:           {self.optical_density}")
        print(f"  motion_correction (TDDR):  {self.motion_correction}")
        print(f"  hemoglobin (Beer-Lambert): {self.hemoglobin} (dpf={self.dpf})")
        print(
            f"  bandpass:                  {self.bandpass} "
            f"({self.bandpass_low}-{self.bandpass_high} Hz)"
        )

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
