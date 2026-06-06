import pytest
from cedanirs import Preprocessor


def test_preprocessor_defaults(log):
    pp = Preprocessor()
    log.debug("Created Preprocessor with defaults: %r", pp)

    assert pp.optical_density is True
    assert pp.motion_correction is True
    assert pp.hemoglobin is True
    assert pp.bandpass is True
    assert pp.bandpass_low == 0.01
    assert pp.bandpass_high == 0.1

    log.info("Default settings verified")


def test_preprocessor_fluent_builder(log):
    pp = Preprocessor().set_bandpass(0.02, 0.3).set_motion_correction(False)
    log.debug("Built pipeline: %r", pp)

    assert pp.bandpass_low == 0.02
    assert pp.bandpass_high == 0.3
    assert pp.motion_correction is False

    log.info("Fluent builder verified")


def test_preprocessor_repr(log):
    pp = Preprocessor()
    r = repr(pp)
    log.debug("repr: %s", r)

    assert "OD" in r
    assert "TDDR" in r
    assert "HbX" in r
    assert "BP" in r

    log.info("repr verified")
