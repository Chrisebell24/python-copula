"""Regression tests for :mod:`rcopula.bootstrap` (shape handling and ``n_jobs``)."""

from __future__ import annotations

import numpy as np
import pytest

import rcopula as rc
from rcopula.bootstrap import bootstrap


def _kendall_matrix(x: np.ndarray) -> np.ndarray:
    return np.asarray(rc.cor_kendall(x), dtype=float)


@pytest.mark.parametrize("method", ["bca", "percentile", "basic"])
def test_matrix_valued_statistic_keeps_its_shape(method: str) -> None:
    u = rc.GaussianCopula([0.5, 0.2, 0.3], dim=3, dispstr="un").rvs(20, random_state=0)
    result = bootstrap(u, _kendall_matrix, n_resamples=30, method=method, random_state=0)
    assert result.estimate.shape == (3, 3)
    lower, upper = result.confidence_interval
    assert lower.shape == upper.shape == (3, 3)
    assert result.standard_error.shape == (3, 3)
    assert result.replicates.shape == (30, 3, 3)
    assert result.bias.shape == (3, 3)
    assert np.all(lower <= upper + 1e-12)
    # The diagonal is identically 1 in every replicate.
    np.testing.assert_allclose(np.diag(lower), 1.0)
    assert len(result.summary().splitlines()) == 3 + 9


def test_matrix_statistic_matches_flattened_statistic() -> None:
    u = rc.ClaytonCopula(2.0, dim=3).rvs(25, random_state=1)
    matrix = bootstrap(u, _kendall_matrix, n_resamples=25, random_state=3)
    flat = bootstrap(u, lambda d: _kendall_matrix(d).ravel(), n_resamples=25, random_state=3)
    np.testing.assert_array_equal(
        matrix.confidence_interval[0].ravel(), flat.confidence_interval[0]
    )
    np.testing.assert_array_equal(
        matrix.confidence_interval[1].ravel(), flat.confidence_interval[1]
    )


def test_vector_and_scalar_shapes_unchanged() -> None:
    u = rc.ClaytonCopula(2.0).rvs(40, random_state=0)
    scalar = bootstrap(u, lambda d: float(np.mean(d[:, 0])), n_resamples=20, random_state=0)
    assert isinstance(scalar.estimate, float)
    assert scalar.replicates.shape == (20,)
    vector = bootstrap(u, lambda d: np.mean(d, axis=0), n_resamples=20, random_state=0)
    assert vector.estimate.shape == (2,)
    assert vector.replicates.shape == (20, 2)


@pytest.mark.parametrize("n_jobs", [0, 1.5, "2"])
def test_invalid_n_jobs_rejected(n_jobs: object) -> None:
    u = rc.ClaytonCopula(2.0).rvs(30, random_state=0)
    with pytest.raises(ValueError, match="n_jobs"):
        bootstrap(u, np.mean, n_resamples=5, n_jobs=n_jobs)  # type: ignore[arg-type]
