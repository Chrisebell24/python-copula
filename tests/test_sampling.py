"""Tests for :mod:`rcopula.sampling` and the inverse Rosenblatt transform.

Three things are worth asserting here and the rest is scaffolding.

The **inverse must actually invert**. It is built by bisection against
``rosenblatt`` itself rather than from a second derivation, so round-tripping is
not a coincidence of two formulas agreeing -- but that only holds if the
bisection converges, which is what these check, in both directions.

A variance-reduction method must **not change the answer**. A sampler that
lowers the standard error and moves the mean has not reduced variance, it has
introduced bias, and that is the failure worth catching.

And the **warnings must be true**. The module says antithetic pairing can
*raise* variance for a symmetric payoff. That claim is measured, not asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.sampling import (
    antithetic_rvs,
    latin_hypercube_rvs,
    quasi_rvs,
    variance_ratio,
)
from rcopula.transforms import inverse_rosenblatt, rosenblatt

ROUND_TRIP = [
    rc.ClaytonCopula(3.0, dim=3),
    rc.ClaytonCopula(0.4, dim=2),
    rc.GumbelCopula(2.0, dim=4),
    rc.FrankCopula(6.0, dim=3),
    rc.JoeCopula(2.5, dim=3),
    rc.GaussianCopula(0.5, dim=4, dispstr="ex"),
    rc.GaussianCopula(-0.6, dim=2),
    rc.StudentCopula(0.5, df=4.0, dim=3, dispstr="ex"),
    rc.PlackettCopula(4.0),
    rc.GalambosCopula(1.2),
    rc.MarshallOlkinCopula([0.4, 0.7]),
    rc.KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(4.0), shapes=(0.4, 0.95)),
]


class TestInverseRosenblatt:
    @pytest.mark.parametrize("copula", ROUND_TRIP, ids=lambda c: c.describe()[:32])
    def test_u_to_z_to_u(self, copula: rc.Copula) -> None:
        u = copula.rvs(1000, random_state=0)
        back = inverse_rosenblatt(copula, rosenblatt(copula, u))
        # 1e-8 covers the families whose conditional CDF is differentiated
        # numerically; the analytic ones land at 1e-13.
        assert np.max(np.abs(back - u)) < 1e-8

    @pytest.mark.parametrize("copula", ROUND_TRIP, ids=lambda c: c.describe()[:32])
    def test_z_to_u_to_z(self, copula: rc.Copula) -> None:
        z = np.random.default_rng(0).uniform(size=(1000, copula.dim))
        back = rosenblatt(copula, inverse_rosenblatt(copula, z))
        assert np.max(np.abs(back - z)) < 1e-8

    @pytest.mark.parametrize("copula", ROUND_TRIP, ids=lambda c: c.describe()[:32])
    def test_independent_uniforms_in_gives_the_copula_out(self, copula: rc.Copula) -> None:
        z = np.random.default_rng(1).uniform(size=(20_000, copula.dim))
        u = inverse_rosenblatt(copula, z)
        assert np.all((u > 0) & (u < 1))
        for j in range(copula.dim):
            assert stats.kstest(u[:, j], "uniform").pvalue > 0.001
        observed = float(rc.cor_kendall(u)[0, 1])
        assert abs(observed - copula.tau()) < 0.02

    def test_the_first_coordinate_passes_straight_through(self) -> None:
        # Z_1 = U_1 in the forward transform, so the inverse must leave it alone.
        copula = rc.ClaytonCopula(2.0, dim=3)
        z = np.random.default_rng(0).uniform(size=(200, 3))
        assert np.allclose(inverse_rosenblatt(copula, z)[:, 0], z[:, 0])

    def test_it_is_monotone_in_each_coordinate(self) -> None:
        # The conditional CDF is monotone in its own argument, which is what
        # makes the bisection safe. If that failed, the inverse would be
        # returning an arbitrary root.
        copula = rc.GumbelCopula(2.5, dim=3)
        grid = np.linspace(0.01, 0.99, 60)
        z = np.column_stack([np.full(60, 0.4), grid, np.full(60, 0.6)])
        assert np.all(np.diff(inverse_rosenblatt(copula, z)[:, 1]) > 0)

    def test_rejects_a_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="columns"):
            inverse_rosenblatt(rc.ClaytonCopula(2.0, dim=3), np.full((5, 2), 0.5))


class TestQuasiRandom:
    @pytest.mark.parametrize("sequence", ["sobol", "halton"])
    def test_the_distribution_is_right(self, sequence: str) -> None:
        copula = rc.GumbelCopula(2.0, dim=3)
        u = quasi_rvs(copula, 4096, sequence=sequence, random_state=0)
        assert u.shape == (4096, 3)
        assert np.all(np.abs(u.mean(axis=0) - 0.5) < 0.01)
        assert abs(float(rc.cor_kendall(u)[0, 1]) - copula.tau()) < 0.02

    def test_coverage_is_more_even_than_independent_draws(self) -> None:
        copula = rc.ClaytonCopula(2.0, dim=2)
        quasi = quasi_rvs(copula, 4096, random_state=0)
        plain = copula.rvs(4096, random_state=0)
        # The largest gap between consecutive order statistics is the direct
        # measure of "evenly filled".
        assert np.max(np.diff(np.sort(quasi[:, 0]))) < np.max(np.diff(np.sort(plain[:, 0])))

    def test_scrambling_makes_it_reproducible_but_not_constant(self) -> None:
        copula = rc.ClaytonCopula(2.0)
        first = quasi_rvs(copula, 256, random_state=3)
        again = quasi_rvs(copula, 256, random_state=3)
        other = quasi_rvs(copula, 256, random_state=4)
        assert np.array_equal(first, again)
        assert not np.array_equal(first, other)

    def test_unscrambled_is_deterministic(self) -> None:
        # Which is exactly why it is not the default: one answer, no error bar.
        copula = rc.ClaytonCopula(2.0)
        a = quasi_rvs(copula, 256, scramble=False, random_state=1)
        b = quasi_rvs(copula, 256, scramble=False, random_state=2)
        assert np.array_equal(a, b)

    def test_rejects_an_unknown_sequence(self) -> None:
        with pytest.raises(ValueError, match="sobol"):
            quasi_rvs(rc.ClaytonCopula(2.0), 64, sequence="faure")

    def test_rejects_a_zero_size(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            quasi_rvs(rc.ClaytonCopula(2.0), 0)


class TestAntithetic:
    def test_the_pairing_is_exact_on_the_first_coordinate(self) -> None:
        u = antithetic_rvs(rc.ClaytonCopula(2.0), 1000, random_state=0)
        assert np.allclose(u[:500, 0] + u[500:, 0], 1.0)

    def test_the_distribution_is_right(self) -> None:
        copula = rc.FrankCopula(5.0)
        u = antithetic_rvs(copula, 20_000, random_state=0)
        assert abs(float(rc.cor_kendall(u)[0, 1]) - copula.tau()) < 0.02
        for j in range(2):
            assert stats.kstest(u[:, j], "uniform").pvalue > 0.001

    def test_an_odd_size_still_returns_pairs(self) -> None:
        u = antithetic_rvs(rc.ClaytonCopula(2.0), 101, random_state=0)
        assert u.shape[0] % 2 == 0
        assert u.shape[0] >= 101


class TestLatinHypercube:
    def test_marginal_coverage_is_guaranteed_not_random(self) -> None:
        copula = rc.FrankCopula(5.0, dim=3)
        lhs = latin_hypercube_rvs(copula, 2000, random_state=0)
        plain = copula.rvs(2000, random_state=0)
        # Stratification pins the mean of the first coordinate far tighter than
        # sampling noise would.
        assert abs(lhs[:, 0].mean() - 0.5) < abs(plain[:, 0].mean() - 0.5)
        assert abs(lhs[:, 0].mean() - 0.5) < 0.005

    def test_the_dependence_survives_the_stratification(self) -> None:
        copula = rc.FrankCopula(5.0, dim=3)
        u = latin_hypercube_rvs(copula, 4000, random_state=0)
        assert abs(float(rc.cor_kendall(u)[0, 1]) - copula.tau()) < 0.03


def _smooth(u: np.ndarray) -> np.ndarray:
    return np.asarray(u[:, 0] * u[:, 1])


def _kinked(u: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(u.sum(axis=1) - 1.0, 0.0))


def _indicator(u: np.ndarray) -> np.ndarray:
    return np.asarray(np.all(u < 0.1, axis=1), dtype=float)


def _symmetric(u: np.ndarray) -> np.ndarray:
    return np.asarray(np.abs(u[:, 0] - u[:, 1]))


class TestVarianceRatio:
    """Whether the module's claims about when each method helps are true."""

    COPULA = rc.ClaytonCopula(2.0)

    @pytest.mark.parametrize(
        "payoff", [_smooth, _kinked, _indicator, _symmetric], ids=lambda f: f.__name__
    )
    @pytest.mark.parametrize("method", ["sobol", "antithetic", "lhs"])
    def test_the_estimate_is_unbiased(self, payoff, method: str) -> None:
        # The important one. A method that lowers the standard error and moves
        # the mean has not reduced variance; it has introduced bias.
        out = variance_ratio(
            self.COPULA, payoff, 1024, method=method, replicates=20, random_state=0
        )
        assert abs(out["plain_mean"] - out["reduced_mean"]) < 0.02

    def test_quasi_random_helps_most_on_a_smooth_payoff(self) -> None:
        smooth = variance_ratio(
            self.COPULA, _smooth, 1024, method="sobol", replicates=20, random_state=0
        )
        rough = variance_ratio(
            self.COPULA, _indicator, 1024, method="sobol", replicates=20, random_state=0
        )
        assert smooth["ratio"] > 20 * rough["ratio"]
        assert rough["ratio"] > 1.0  # it still helps, just far less

    def test_antithetic_hurts_a_symmetric_payoff(self) -> None:
        # The module warns about this rather than presenting antithetic pairing
        # as free. The warning is measured here, not asserted.
        out = variance_ratio(
            self.COPULA, _symmetric, 1024, method="antithetic", replicates=20, random_state=0
        )
        assert out["ratio"] < 1.0

    def test_antithetic_helps_a_monotone_payoff(self) -> None:
        out = variance_ratio(
            self.COPULA, _smooth, 1024, method="antithetic", replicates=20, random_state=0
        )
        assert out["ratio"] > 1.3

    def test_the_equivalent_sample_factor_is_the_squared_ratio(self) -> None:
        out = variance_ratio(self.COPULA, _smooth, 512, method="lhs", replicates=12, random_state=0)
        assert out["equivalent_sample_factor"] == pytest.approx(out["ratio"] ** 2)

    def test_rejects_too_few_replicates(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            variance_ratio(self.COPULA, _smooth, 128, replicates=1)

    def test_rejects_an_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="sobol, halton"):
            variance_ratio(self.COPULA, _smooth, 128, method="importance", replicates=3)
