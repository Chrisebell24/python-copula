"""Tests for the two-parameter BB1 and BB7 Archimedean copulas.

The formulas (Joe 2014, sections 4.17 and 4.23) are checked three ways: against
their own definitions by numerical differentiation (the h-function is dC/dv,
the density is dh/du), against the closed-form special cases (``delta = 1`` /
``theta = 1`` are Clayton), and against simulation. R parity for the same
quantities is in ``test_golden_vine.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.serialize import from_dict, to_dict

CASES = [
    rc.BB1Copula(0.8, 1.6),
    rc.BB1Copula(0.2, 3.5),
    rc.BB1Copula(3.0, 1.2),
    rc.BB7Copula(1.5, 2.0),
    rc.BB7Copula(3.0, 0.4),
    rc.BB7Copula(1.2, 5.0),
]
POINTS = np.random.default_rng(0).uniform(0.02, 0.98, size=(200, 2))


@pytest.mark.parametrize("cop", CASES, ids=repr)
class TestDefinitions:
    def test_h_is_the_derivative_of_the_cdf(self, cop) -> None:
        e = 1e-5
        for given in (0, 1):
            hi, lo = POINTS.copy(), POINTS.copy()
            hi[:, given] += e
            lo[:, given] -= e
            numeric = (cop.cdf(hi) - cop.cdf(lo)) / (2 * e)
            np.testing.assert_allclose(cop.hfunc(POINTS, given=given), numeric, atol=1e-7)

    def test_density_is_the_derivative_of_h(self, cop) -> None:
        e = 1e-5
        hi, lo = POINTS.copy(), POINTS.copy()
        hi[:, 0] += e
        lo[:, 0] -= e
        numeric = (cop.hfunc(hi, 1) - cop.hfunc(lo, 1)) / (2 * e)
        # Central differences of h carry ~1e-11 absolute error, which is all
        # that is left where the density itself is tiny (~1e-5 in a far corner).
        np.testing.assert_allclose(cop.pdf(POINTS), numeric, rtol=1e-6, atol=1e-9)

    def test_hinv_inverts_h(self, cop) -> None:
        h = cop.hfunc(POINTS, given=1)
        np.testing.assert_allclose(cop.hinv(h, POINTS[:, 1]), POINTS[:, 0], atol=1e-10)

    def test_cdf_has_uniform_margins_and_the_frechet_bounds(self, cop) -> None:
        x = np.linspace(0.01, 0.99, 50)
        np.testing.assert_allclose(cop.cdf(np.column_stack([x, np.ones_like(x)])), x, atol=1e-9)
        np.testing.assert_allclose(cop.cdf(np.column_stack([np.ones_like(x), x])), x, atol=1e-9)
        c = cop.cdf(POINTS)
        assert np.all(c <= POINTS.min(axis=1) + 1e-12)
        assert np.all(c >= POINTS.prod(axis=1) - 1e-12)  # positive quadrant dependence

    def test_sampling_matches_tau_and_margins(self, cop) -> None:
        u = cop.rvs(20_000, random_state=1)
        assert stats.kendalltau(u[:, 0], u[:, 1]).statistic == pytest.approx(cop.tau(), abs=0.015)
        for j in range(2):
            assert stats.kstest(u[:, j], "uniform").pvalue > 1e-3

    def test_round_trips_through_serialisation(self, cop) -> None:
        assert from_dict(to_dict(cop)) == cop


def test_bb1_reduces_to_clayton() -> None:
    np.testing.assert_allclose(
        rc.BB1Copula(2.0, 1.0).logpdf(POINTS), rc.ClaytonCopula(2.0).logpdf(POINTS), atol=1e-10
    )


def test_bb7_reduces_to_clayton() -> None:
    np.testing.assert_allclose(
        rc.BB7Copula(1.0, 2.0).logpdf(POINTS), rc.ClaytonCopula(2.0).logpdf(POINTS), atol=1e-10
    )


def test_closed_form_tau_and_tails() -> None:
    cop = rc.BB1Copula(0.5, 1.5)
    assert cop.tau() == pytest.approx(1 - 2 / (1.5 * 2.5))
    lam = cop.lambda_()
    assert lam.lower == pytest.approx(2 ** (-1 / 0.75))
    assert lam.upper == pytest.approx(2 - 2 ** (1 / 1.5))
    lam = rc.BB7Copula(1.5, 2.0).lambda_()
    assert lam.lower == pytest.approx(2**-0.5)
    assert lam.upper == pytest.approx(2 - 2 ** (1 / 1.5))


def test_tail_dependence_shows_in_simulation() -> None:
    """BB1 with strong lower and weak upper tails: the joint-exceedance ratio says so."""
    cop = rc.BB1Copula(2.0, 1.1)
    u = cop.rvs(400_000, random_state=3)
    q = 0.005
    lower = np.mean((u[:, 0] < q) & (u[:, 1] < q)) / q
    upper = np.mean((u[:, 0] > 1 - q) & (u[:, 1] > 1 - q)) / q
    lam = cop.lambda_()
    assert lower == pytest.approx(lam.lower, abs=0.06)
    assert upper < lower


def test_parameter_validation() -> None:
    with pytest.raises(ValueError, match="theta"):
        rc.BB1Copula(0.0, 1.5)
    with pytest.raises(ValueError, match="delta"):
        rc.BB1Copula(1.0, 0.5)
    with pytest.raises(ValueError, match="delta"):
        rc.BB7Copula(1.5, 0.0)
    with pytest.raises(ValueError, match="theta"):
        rc.BB7Copula(0.5, 1.0)
    with pytest.raises(ValueError, match="bivariate"):
        rc.BB1Copula(1.0, 1.5, dim=3)


def test_fit_recovers_both_parameters() -> None:
    truth = rc.BB1Copula(1.0, 1.8)
    u = truth.rvs(3000, random_state=5)
    fitted = rc.fit(rc.BB1Copula(), u).copula
    assert fitted.params == pytest.approx(truth.params, rel=0.15)


def test_rotated_bb_selected_for_negative_dependence() -> None:
    u = rc.RotatedCopula(rc.BB7Copula(1.6, 1.5), 90).rvs(2000, random_state=2)
    ranking = rc.select_copula(u, families=["bb7", "bb7_90", "bb7_180", "bb7_270"])
    assert ranking.best_name == "bb7_90"
