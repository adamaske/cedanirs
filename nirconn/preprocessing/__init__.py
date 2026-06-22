"""Preprocessing pipeline for raw fNIRS recordings.

Connectivity estimates are only as trustworthy as the signals they are computed
from, so nirconn ships a configurable preprocessing pipeline covering the steps
the literature treats as essential: optical-density conversion, motion-artifact
correction, the modified Beer-Lambert law, and band-pass filtering to isolate
the low-frequency oscillations (typically 0.01-0.1 Hz) that carry resting-state
connectivity.

The pipeline is backend-pluggable. The reference backend wraps
`cedalion <https://github.com/ibs-lab/cedalion>`_, imported lazily so that the
core connectivity functionality never depends on it being installed.
"""

from __future__ import annotations

from .preprocessor import Preprocessor

__all__ = ["Preprocessor"]
