import numpy as np
import pytest

import nirconn as cn
from nirconn import (
    EstimatorNotFoundError,
    PearsonCorrelation,
    create_estimator,
    get_estimator,
    list_estimators,
    register_estimator,
)
from nirconn.core.registry import is_registered
from nirconn.estimators.base import ConnectivityEstimator, EstimateOutput


def test_pearson_is_registered():
    assert is_registered("pearson")
    assert get_estimator("pearson") is PearsonCorrelation
    assert isinstance(create_estimator("pearson"), PearsonCorrelation)


def test_list_estimators_contains_pearson():
    names = [e["name"] for e in list_estimators()]
    assert "pearson" in names
    entry = next(e for e in list_estimators() if e["name"] == "pearson")
    assert entry["kind"] == "functional"
    assert entry["directed"] is False
    assert entry["description"]


def test_unknown_estimator_raises():
    with pytest.raises(EstimatorNotFoundError) as exc:
        get_estimator("does_not_exist")
    assert "does_not_exist" in str(exc.value)
    assert "pearson" in str(exc.value)  # lists available


def test_register_custom_estimator_roundtrip():
    @register_estimator(name="_test_constant")
    class ConstantConnectivity(ConnectivityEstimator):
        """A trivial estimator for testing the registry."""

        name = "_test_constant"

        def _estimate(self, x):
            n = x.shape[0]
            return EstimateOutput(matrix=np.full((n, n), 0.5))

    try:
        assert is_registered("_test_constant")
        result = cn.connectivity(np.zeros((3, 50)), method="_test_constant")
        vals = result.matrix.values
        assert vals.shape == (3, 3)
        assert np.all(vals == 0.5)
    finally:
        from nirconn.core.registry import _REGISTRY

        _REGISTRY.pop("_test_constant", None)


def test_duplicate_registration_raises():
    with pytest.raises(ValueError):

        @register_estimator(name="pearson")
        class Dup(ConnectivityEstimator):
            name = "pearson"

            def _estimate(self, x):
                n = x.shape[0]
                return EstimateOutput(matrix=np.eye(n))
