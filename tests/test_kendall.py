"""Tests for the Kendall distribution function.

``K(t) = P(C(U) <= t)`` is validated three independent ways, because each
catches a different kind of error:

* **against simulation** -- draw from the copula, evaluate ``C`` at its own
  draws, compare distributions. Catches an outright wrong formula.
* **against Kendall's tau**, through ``tau = 3 - 4 int K``. This connects ``K``
  to a quantity computed by an entirely separate route, so agreement to five
  decimals is hard to arrange by accident. It is what caught an argument-order
  slip that was invisible on every exchangeable family.
* **against the closed forms** that exist for independence and the comonotone
  bound in every dimension.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.special import gammaln

import rcopula as rc
from rcopula.kendall import (
    kendall_cdf,
    kendall_empirical,
    kendall_pdf,
    kendall_ppf,
    kendall_return_period,
    kendall_rvs,
    return_period_level,
)
from rcopula.structural import KhoudrajiCopula

GRID = np.array([0.05, 0.2, 0.5, 0.8, 0.95])

BIVARIATE = [
    rc.ClaytonCopula(3.0),
    rc.GumbelCopula(2.5),
    rc.FrankCopula(5.0),
    rc.FrankCopula(-4.0),
    rc.JoeCopula(3.0),
    rc.AMHCopula(0.7),
    rc.GaussianCopula(0.6),
    rc.GalambosCopula(1.5),
    rc.PlackettCopula(4.0),
    rc.FGMCopula(0.6),
    rc.MarshallOlkinCopula(0.4, 0.7),
    KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(3.0), [0.4, 0.95]),
]

MULTIVARIATE = [
    rc.ClaytonCopula(3.0, dim=3),
    rc.ClaytonCopula(2.0, dim=5),
    rc.GumbelCopula(2.5, dim=4),
    rc.FrankCopula(4.0, dim=3),
    rc.IndependenceCopula(4),
    rc.FrechetUpperCopula(3),
]


def _tau_via_kendall(copula: rc.Copula) -> float:
    """``tau = 3 - 4 int_0^1 K``, on a fixed rule.

    Adaptive quadrature warns about roundoff here: ``K`` has an infinite slope
    at the origin for tail-dependent families, so the integrator keeps
    subdividing an interval that contributes almost nothing. A fixed
    Gauss-Legendre rule is quieter and, at 2000 nodes, well inside the tolerance
    this is checked to.
    """
    nodes, weights = np.polynomial.legendre.leggauss(2000)
    grid = 0.5 * (nodes + 1.0)
    return float(3.0 - 4.0 * np.sum(0.5 * weights * kendall_cdf(copula, grid)))


class TestKendallCdfIsADistributionFunction:
    @pytest.mark.parametrize("cop", [*BIVARIATE, *MULTIVARIATE])
    def test_it_is_a_cdf_on_the_unit_interval(self, cop: rc.Copula) -> None:
        t = np.linspace(0.001, 0.999, 300)
        k = kendall_cdf(cop, t)
        assert np.all((k >= 0.0) & (k <= 1.0))
        assert np.all(np.diff(k) >= -1e-12)
        assert kendall_cdf(cop, 0.0)[0] == 0.0
        assert kendall_cdf(cop, 1.0)[0] == 1.0

    @pytest.mark.parametrize("cop", [*BIVARIATE, *MULTIVARIATE])
    def test_it_dominates_the_uniform(self, cop: rc.Copula) -> None:
        """``K(t) >= t`` for every copula -- which is why the Kendall return
        period is never shorter than the univariate one."""
        t = np.linspace(0.01, 0.99, 99)
        assert np.all(kendall_cdf(cop, t) >= t - 1e-12)

    @pytest.mark.parametrize("cop", [*BIVARIATE, *MULTIVARIATE])
    def test_it_matches_the_law_of_c_at_its_own_draws(self, cop: rc.Copula) -> None:
        """The definition, checked by simulation."""
        w = cop.cdf(cop.rvs(150_000, random_state=0))
        empirical = np.array([np.mean(w <= x) for x in GRID])
        assert np.allclose(kendall_cdf(cop, GRID), empirical, atol=6e-3)


class TestKendallCdfRecoversTau:
    """``tau = 3 - 4 int_0^1 K``. The two sides are computed by completely
    different code, so agreement is strong evidence for both."""

    @pytest.mark.parametrize("cop", BIVARIATE)
    def test_the_tau_identity_holds(self, cop: rc.Copula) -> None:
        assert _tau_via_kendall(cop) == pytest.approx(cop.tau(), abs=1e-4)

    def test_it_catches_a_transposed_conditional(self) -> None:
        """Marshall-Olkin is asymmetric, so it is the case that notices.

        Evaluating the conditional distribution at ``(v, u)`` instead of
        ``(u, v)`` gives identical answers for every exchangeable family and a
        3.5% error here -- which is exactly how the slip survived until the tau
        identity was checked.
        """
        cop = rc.MarshallOlkinCopula(0.4, 0.7)
        assert _tau_via_kendall(cop) == pytest.approx(cop.tau(), abs=1e-4)


class TestClosedForms:
    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_independence(self, dim: int) -> None:
        """``K(t) = t sum_{k<d} (-log t)^k / k!`` -- a Poisson tail."""
        t = np.array([0.05, 0.3, 0.7, 0.95])
        k = np.arange(dim)
        expected = t * np.sum(np.exp(k * np.log(-np.log(t))[:, None] - gammaln(k + 1.0)), axis=1)
        assert np.allclose(kendall_cdf(rc.IndependenceCopula(dim), t), expected)

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_comonotonicity_is_uniform(self, dim: int) -> None:
        """``C(U) = min_j U_j = U_1``, so there is nothing left to summarise."""
        t = np.array([0.05, 0.3, 0.7, 0.95])
        assert np.allclose(kendall_cdf(rc.FrechetUpperCopula(dim), t), t)

    def test_the_bivariate_archimedean_form(self) -> None:
        """``K(t) = t - s psi'(s)`` with ``s = psi^{-1}(t)``, the Genest-Rivest form."""
        cop = rc.ClaytonCopula(3.0)
        gen, theta = cop.generator, cop.theta
        t = np.array([0.1, 0.4, 0.9])
        s = gen.ipsi(t, theta)
        expected = t + s * np.exp(gen.log_abs_dpsi(s, theta))
        assert np.allclose(kendall_cdf(cop, t), expected)


class TestKendallDensity:
    @pytest.mark.parametrize("cop", [*BIVARIATE[:6], *MULTIVARIATE[:4]])
    def test_it_integrates_to_one(self, cop: rc.Copula) -> None:
        total = integrate.quad(lambda x: kendall_pdf(cop, x)[0], 0.0, 1.0, limit=200)[0]
        assert total == pytest.approx(1.0, abs=2e-6)

    @pytest.mark.parametrize("cop", [*BIVARIATE[:6], *MULTIVARIATE[:4]])
    def test_it_is_non_negative(self, cop: rc.Copula) -> None:
        assert np.all(kendall_pdf(cop, np.linspace(0.001, 0.999, 500)) >= 0.0)

    @pytest.mark.parametrize("cop", [*BIVARIATE[:6], *MULTIVARIATE[:4]])
    def test_it_is_the_derivative_of_the_cdf(self, cop: rc.Copula) -> None:
        h = 1e-6
        for t in (0.2, 0.5, 0.8):
            slope = (kendall_cdf(cop, t + h)[0] - kendall_cdf(cop, t - h)[0]) / (2 * h)
            assert kendall_pdf(cop, t)[0] == pytest.approx(slope, rel=1e-4)


class TestKendallQuantile:
    @pytest.mark.parametrize("cop", [*BIVARIATE[:6], *MULTIVARIATE[:4]])
    def test_it_inverts_the_cdf(self, cop: rc.Copula) -> None:
        p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        assert np.allclose(kendall_cdf(cop, kendall_ppf(cop, p)), p, atol=1e-9)

    def test_the_endpoints(self) -> None:
        cop = rc.ClaytonCopula(3.0)
        assert kendall_ppf(cop, 0.0)[0] == 0.0
        assert kendall_ppf(cop, 1.0)[0] == 1.0

    def test_it_rejects_probabilities_outside_the_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="p must lie"):
            kendall_ppf(rc.ClaytonCopula(3.0), [0.5, 1.5])


class TestKendallSampling:
    @pytest.mark.parametrize("cop", [*BIVARIATE[:6], *MULTIVARIATE[:4]])
    def test_the_draws_follow_k(self, cop: rc.Copula) -> None:
        w = kendall_rvs(cop, 20_000, random_state=0)
        assert stats.kstest(w, lambda x: kendall_cdf(cop, x)).pvalue > 0.01

    def test_it_is_reproducible(self) -> None:
        cop = rc.GumbelCopula(2.0)
        assert np.array_equal(
            kendall_rvs(cop, 50, random_state=3), kendall_rvs(cop, 50, random_state=3)
        )


class TestEmpiricalKendallFunction:
    @pytest.mark.parametrize(
        "cop", [rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5), rc.FrankCopula(5.0)]
    )
    def test_it_tracks_the_true_k_without_knowing_the_family(self, cop: rc.Copula) -> None:
        x = cop.rvs(4000, random_state=0)
        grid = np.array([0.1, 0.3, 0.5, 0.8])
        assert np.allclose(kendall_empirical(x, grid), kendall_cdf(cop, grid), atol=0.03)

    def test_it_is_invariant_to_the_margins(self) -> None:
        """Rank-based, so applying any increasing transform changes nothing."""
        u = rc.ClaytonCopula(3.0).rvs(1500, random_state=0)
        raw = np.column_stack([stats.norm.ppf(u[:, 0]), np.exp(u[:, 1])])
        grid = np.array([0.2, 0.5, 0.8])
        assert np.allclose(kendall_empirical(u, grid), kendall_empirical(raw, grid))

    def test_it_works_above_two_dimensions(self) -> None:
        cop = rc.ClaytonCopula(3.0, dim=3)
        x = cop.rvs(4000, random_state=0)
        grid = np.array([0.1, 0.3, 0.6])
        assert np.allclose(kendall_empirical(x, grid), kendall_cdf(cop, grid), atol=0.04)

    def test_without_a_grid_it_returns_the_sorted_statistics(self) -> None:
        x = rc.ClaytonCopula(2.0).rvs(200, random_state=0)
        w = kendall_empirical(x)
        assert w.shape == (200,)
        assert np.all(np.diff(w) >= 0)
        assert np.all((w >= 0.0) & (w <= 1.0))

    def test_it_needs_more_than_one_observation(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            kendall_empirical([[0.5, 0.5]])


class TestKendallReturnPeriod:
    def test_comonotonicity_gives_the_univariate_answer(self) -> None:
        """Perfectly dependent variables leave no second chance, so the joint
        return period collapses to the marginal one."""
        assert kendall_return_period(rc.FrechetUpperCopula(2), 0.99)[0] == pytest.approx(100.0)

    def test_every_other_copula_waits_longer(self) -> None:
        for cop in BIVARIATE:
            assert kendall_return_period(cop, 0.99)[0] > 100.0

    def test_equal_tau_can_mean_a_thirty_fold_difference(self) -> None:
        """The fact worth designing around.

        Gumbel and Clayton at Kendall's tau = 0.5 give 199 and 6689 years at the
        same critical level. The critical layer is an upper-corner event, Gumbel
        has upper tail dependence and Clayton has none -- so a rank correlation,
        however well estimated, does not determine the design life.
        """
        gumbel = kendall_return_period(rc.GumbelCopula(2.0), 0.99)[0]
        clayton = kendall_return_period(rc.ClaytonCopula(2.0), 0.99)[0]
        assert rc.GumbelCopula(2.0).tau() == pytest.approx(0.5)
        assert rc.ClaytonCopula(2.0).tau() == pytest.approx(0.5)
        assert gumbel == pytest.approx(199.0, abs=1.0)
        assert clayton == pytest.approx(6689.0, abs=10.0)
        assert clayton / gumbel > 30.0

    def test_the_interval_scales_it(self) -> None:
        cop = rc.GumbelCopula(2.0)
        assert kendall_return_period(cop, 0.99, interval=0.5)[0] == pytest.approx(
            0.5 * kendall_return_period(cop, 0.99)[0]
        )

    def test_it_rejects_a_non_positive_interval(self) -> None:
        with pytest.raises(ValueError, match="interval must be positive"):
            kendall_return_period(rc.GumbelCopula(2.0), 0.99, interval=0.0)

    @pytest.mark.parametrize(
        "cop", [rc.ClaytonCopula(3.0), rc.GumbelCopula(2.0), rc.FrankCopula(5.0)]
    )
    @pytest.mark.parametrize("period", [10.0, 100.0, 1000.0])
    def test_the_level_and_the_period_invert_each_other(
        self, cop: rc.Copula, period: float
    ) -> None:
        t = return_period_level(cop, period)
        assert kendall_return_period(cop, t)[0] == pytest.approx(period, rel=1e-6)

    def test_it_rejects_a_non_positive_period(self) -> None:
        with pytest.raises(ValueError, match="period must be positive"):
            return_period_level(rc.GumbelCopula(2.0), -5.0)
