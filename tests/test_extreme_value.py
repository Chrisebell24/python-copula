"""Behavioural tests for the extreme-value families.

The defining structural facts are checked directly: the Pickands function must
be convex and sit between ``max(t, 1-t)`` and ``1``, and every extreme-value
copula must have upper tail dependence but none in the lower tail.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import rcopula as rc

FAMILIES = [
    (rc.GalambosCopula, (2.0,), {}),
    (rc.HuslerReissCopula, (1.5,), {}),
    (rc.TawnCopula, (0.6,), {}),
    (rc.TEVCopula, (0.5,), {"df": 4}),
]

W = np.linspace(0.001, 0.999, 401)


def _make(cls, args, kw):
    return cls(*args, **kw)


@pytest.mark.parametrize(("cls", "args", "kw"), FAMILIES)
def test_pickands_lies_between_its_bounds(cls, args, kw) -> None:
    """max(t, 1-t) <= A(t) <= 1 -- the comonotone and independence envelopes."""
    a = _make(cls, args, kw).A(W)
    assert np.all(a <= 1.0 + 1e-12)
    assert np.all(a >= np.maximum(W, 1.0 - W) - 1e-12)


@pytest.mark.parametrize(("cls", "args", "kw"), FAMILIES)
def test_pickands_is_convex(cls, args, kw) -> None:
    cop = _make(cls, args, kw)
    assert np.all(cop.d2A(W) >= -1e-9)
    # and the discrete second difference agrees
    a = cop.A(W)
    assert np.all(np.diff(a, 2) >= -1e-9)


@pytest.mark.parametrize(("cls", "args", "kw"), FAMILIES)
def test_analytic_derivatives_match_finite_differences(cls, args, kw) -> None:
    """Guards the hand-derived A' and A'' against algebra slips."""
    cop = _make(cls, args, kw)
    t = np.linspace(0.15, 0.85, 25)
    h = 1e-5
    fd1 = (cop.A(t + h) - cop.A(t - h)) / (2 * h)
    fd2 = (cop.A(t + h) - 2 * cop.A(t) + cop.A(t - h)) / h**2
    assert np.allclose(cop.dA(t), fd1, atol=1e-8)
    assert np.allclose(cop.d2A(t), fd2, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(("cls", "args", "kw"), FAMILIES)
def test_upper_tail_dependence_only(cls, args, kw) -> None:
    lam = _make(cls, args, kw).lambda_()
    assert lam.lower == 0.0
    assert lam.upper > 0.0
    assert lam.upper == pytest.approx(2 * (1 - float(_make(cls, args, kw).A(0.5))))


@pytest.mark.parametrize(("cls", "args", "kw"), FAMILIES)
def test_uniform_margins_of_the_cdf(cls, args, kw) -> None:
    cop = _make(cls, args, kw)
    u = np.linspace(0.02, 0.98, 30)
    grid = np.column_stack([u, np.ones_like(u)])
    assert np.allclose(cop.cdf(grid), u, atol=1e-10)


@pytest.mark.parametrize(("cls", "args", "kw"), FAMILIES)
def test_frechet_hoeffding_bounds(cls, args, kw) -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(0.02, 0.98, size=(300, 2))
    c = _make(cls, args, kw).cdf(u)
    assert np.all(c >= np.maximum(u.sum(axis=1) - 1.0, 0.0) - 1e-12)
    assert np.all(c <= u.min(axis=1) + 1e-12)


@pytest.mark.parametrize(("cls", "args", "kw"), FAMILIES)
def test_pdf_matches_numerical_derivative_of_cdf(cls, args, kw) -> None:
    cop = _make(cls, args, kw)
    h = 1e-5
    for u, v in [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]:
        numerical = (
            cop.cdf([[u + h, v + h]])[0]
            - cop.cdf([[u + h, v - h]])[0]
            - cop.cdf([[u - h, v + h]])[0]
            + cop.cdf([[u - h, v - h]])[0]
        ) / (4 * h * h)
        assert cop.pdf([[u, v]])[0] == pytest.approx(numerical, rel=1e-4)


@pytest.mark.parametrize(("cls", "args", "kw"), FAMILIES)
def test_density_integrates_to_one(cls, args, kw) -> None:
    rng = np.random.default_rng(2)
    u = rng.uniform(size=(100_000, 2))
    assert _make(cls, args, kw).pdf(u).mean() == pytest.approx(1.0, abs=0.03)


@pytest.mark.slow
@pytest.mark.parametrize(("cls", "args", "kw"), FAMILIES)
def test_sampling_reproduces_the_population_tau(cls, args, kw) -> None:
    cop = _make(cls, args, kw)
    u = cop.rvs(20_000, random_state=4)
    for j in range(2):
        assert stats.kstest(u[:, j], "uniform").pvalue > 0.001
    empirical = stats.kendalltau(u[:, 0], u[:, 1]).statistic
    assert empirical == pytest.approx(cop.tau(), abs=0.02)


def test_independence_limits() -> None:
    """A == 1 is independence; each family reaches it at its own boundary."""
    assert np.allclose(rc.TawnCopula(0.0).A(W), 1.0)
    assert np.allclose(rc.GalambosCopula(1e-8).A(W), 1.0, atol=1e-7)


def test_gumbel_is_the_only_archimedean_extreme_value_family() -> None:
    """Its Pickands function is available for comparison."""
    from rcopula.core.extreme_value import gumbel_pickands

    theta = 2.0
    g = rc.GumbelCopula(theta)
    u = np.array([[0.3, 0.6], [0.8, 0.2]])
    log_uv = np.log(u).sum(axis=1)
    t = np.log(u[:, 1]) / log_uv
    assert np.allclose(g.cdf(u), np.exp(log_uv * gumbel_pickands(t, theta)))


def test_tawn_dependence_range_is_limited() -> None:
    """R notes Tawn is valid only for tau < 0.4184; theta = 1 is the maximum."""
    assert rc.TawnCopula(1.0).tau() == pytest.approx(0.418399152, abs=1e-8)


@pytest.mark.parametrize(("cls", "args", "kw"), FAMILIES)
def test_rejects_non_bivariate(cls, args, kw) -> None:
    with pytest.raises(ValueError, match="bivariate only"):
        cls(*args, dim=3, **kw)


@pytest.mark.parametrize("tau", [0.1, 0.3, 0.5, 0.7])
def test_from_tau_round_trips(tau: float) -> None:
    for cls in (rc.GalambosCopula, rc.HuslerReissCopula):
        assert cls.from_tau(tau).tau() == pytest.approx(tau, abs=1e-8)
