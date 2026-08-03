"""Tests for conditional copulas, pairs trading and portfolio optimisation.

The Rosenblatt transform is compared against R's ``cCopula`` (fixtures from
``tools/rgolden/08_transforms.R``). Everything downstream is validated by the
defining property -- conditional CDFs are uniform under the true copula -- and
by closed forms where they exist (the unconstrained minimum-variance portfolio,
and the CVaR of a normal portfolio).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.portfolio import (
    backtest_pairs,
    efficient_frontier,
    mean_cvar_weights,
    min_variance_weights,
    mispricing_index,
    pairs_signal,
    simulate_returns,
)
from rcopula.risk import expected_shortfall
from rcopula.transforms import conditional_cdf, conditional_ppf, rosenblatt

GOLDEN = Path(__file__).parent / "golden" / "transforms.json"

FAMILIES = [
    rc.ClaytonCopula(3.0),
    rc.GumbelCopula(2.5),
    rc.FrankCopula(5.0),
    rc.JoeCopula(2.5),
    rc.GaussianCopula(0.6),
    rc.StudentCopula(0.6, df=5),
]


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run `make golden`")
    return json.loads(GOLDEN.read_text())


# ======================================================================
# Conditional distributions
# ======================================================================


class TestConditionalCdf:
    @pytest.mark.parametrize("cop", FAMILIES, ids=lambda c: c.name)
    def test_is_uniform_under_the_true_copula(self, cop) -> None:
        """The defining property, and what makes the value a usable signal."""
        u = cop.rvs(20_000, random_state=0)
        for given in (0, 1):
            h = conditional_cdf(cop, u, given=given)
            assert stats.kstest(h, "uniform").pvalue > 0.001

    @pytest.mark.parametrize("cop", FAMILIES, ids=lambda c: c.name)
    def test_matches_numerical_differentiation(self, cop) -> None:
        """Guards the analytic forms against algebra slips."""
        u = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.4], [0.35, 0.9]])
        step = 1e-6
        for given in (0, 1):
            hi, lo = u.copy(), u.copy()
            hi[:, given] += step
            lo[:, given] -= step
            numeric = (cop.cdf(hi) - cop.cdf(lo)) / (2 * step)
            assert np.allclose(conditional_cdf(cop, u, given), numeric, atol=1e-5)

    def test_independence_is_the_identity(self) -> None:
        u = np.array([[0.3, 0.8], [0.7, 0.2]])
        assert np.allclose(conditional_cdf(rc.IndependenceCopula(2), u), [0.3, 0.7])

    @pytest.mark.parametrize("cop", FAMILIES, ids=lambda c: c.name)
    def test_is_monotone_in_the_conditioned_coordinate(self, cop) -> None:
        x = np.linspace(0.01, 0.99, 50)
        h = conditional_cdf(cop, np.column_stack([x, np.full(50, 0.4)]))
        assert np.all(np.diff(h) >= -1e-9)

    @pytest.mark.parametrize("cop", FAMILIES, ids=lambda c: c.name)
    def test_inverts_correctly(self, cop) -> None:
        w = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        cond = np.array([0.2, 0.4, 0.5, 0.7, 0.9])
        x = conditional_ppf(cop, w, cond)
        back = conditional_cdf(cop, np.column_stack([x, cond]))
        assert np.allclose(back, w, atol=1e-7)

    def test_works_for_families_without_a_closed_form(self) -> None:
        """Plackett and FGM fall through to numerical differentiation."""
        for cop in (rc.PlackettCopula(4.0), rc.FGMCopula(0.7)):
            h = conditional_cdf(cop, cop.rvs(20_000, random_state=0))
            assert stats.kstest(h, "uniform").pvalue > 0.001

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="bivariate"):
            conditional_cdf(rc.ClaytonCopula(2.0, dim=3), np.full((5, 3), 0.5))
        with pytest.raises(ValueError, match="given must be"):
            conditional_cdf(rc.ClaytonCopula(2.0), np.full((5, 2), 0.5), given=2)


class TestRosenblatt:
    @pytest.mark.golden
    def test_matches_r(self, golden: dict) -> None:
        u = np.asarray(golden["u"], dtype=float)
        builders = {
            "clayton": rc.ClaytonCopula(3.0, dim=3),
            "gumbel": rc.GumbelCopula(2.5, dim=3),
            "frank": rc.FrankCopula(5.0, dim=3),
            "normal": rc.GaussianCopula(0.5, dim=3),
            "t": rc.StudentCopula(0.5, dim=3, df=5),
        }
        for name, cop in builders.items():
            expected = np.asarray(golden[name], dtype=float)
            assert np.allclose(rosenblatt(cop, u), expected, atol=1e-9), name

    @pytest.mark.parametrize(
        "cop",
        [
            rc.ClaytonCopula(3.0, dim=3),
            rc.GumbelCopula(2.5, dim=3),
            rc.GaussianCopula(0.5, dim=3),
            rc.StudentCopula(0.5, dim=3, df=5),
        ],
        ids=lambda c: c.name,
    )
    def test_produces_independent_uniforms(self, cop) -> None:
        """The whole point: if the copula is right, the output is iid uniform."""
        z = rosenblatt(cop, cop.rvs(8000, random_state=0))
        for j in range(3):
            assert stats.kstest(z[:, j], "uniform").pvalue > 0.001
        corr = np.corrcoef(z, rowvar=False)
        assert np.max(np.abs(corr - np.eye(3))) < 0.05

    def test_the_wrong_copula_is_visibly_wrong(self) -> None:
        truth = rc.ClaytonCopula(6.0, dim=3)
        u = truth.rvs(5000, random_state=0)
        bad = rosenblatt(rc.ClaytonCopula(0.2, dim=3), u)
        assert stats.kstest(bad[:, 1], "uniform").pvalue < 1e-6

    def test_first_coordinate_passes_through(self) -> None:
        cop = rc.ClaytonCopula(2.0, dim=3)
        u = cop.rvs(100, random_state=0)
        assert np.allclose(rosenblatt(cop, u)[:, 0], u[:, 0])

    def test_unsupported_high_dimensional_family_says_so(self) -> None:
        cop = rc.IndependenceCopula(4)
        # Independence is neither Archimedean nor elliptical in this hierarchy.
        with pytest.raises(NotImplementedError, match="dim=2 only"):
            rosenblatt(cop, cop.rvs(10, random_state=0))


# ======================================================================
# Pairs trading
# ======================================================================


class TestPairsTrading:
    def test_mispricing_indices_are_uniform(self) -> None:
        cop = rc.ClaytonCopula(3.0)
        h1, h2 = mispricing_index(cop, cop.rvs(20_000, random_state=0))
        assert stats.kstest(h1, "uniform").pvalue > 0.001
        assert stats.kstest(h2, "uniform").pvalue > 0.001

    def test_signal_direction(self) -> None:
        cop = rc.ClaytonCopula(3.0)
        u = np.array([[0.02, 0.95], [0.95, 0.02], [0.50, 0.50]])
        assert list(pairs_signal(cop, u)) == [1, -1, 0]

    def test_signals_are_rare_and_tighten_with_the_threshold(self) -> None:
        cop = rc.GaussianCopula(0.7)
        u = cop.rvs(20_000, random_state=0)
        loose = np.mean(pairs_signal(cop, u, entry=0.10) != 0)
        tight = np.mean(pairs_signal(cop, u, entry=0.02) != 0)
        assert 0.0 < tight < loose < 0.25

    def test_a_common_move_does_not_trigger_a_signal(self) -> None:
        """Both assets falling together is not relative mispricing.

        This is the property a z-scored spread gets right by accident and a
        naive per-asset rule gets wrong.
        """
        cop = rc.GaussianCopula(0.8)
        both_low = np.array([[0.03, 0.03], [0.02, 0.04], [0.97, 0.96]])
        assert np.all(pairs_signal(cop, both_low) == 0)

    def test_rejects_bad_thresholds_and_shapes(self) -> None:
        cop = rc.ClaytonCopula(2.0)
        with pytest.raises(ValueError, match="entry must lie"):
            pairs_signal(cop, np.full((3, 2), 0.5), entry=0.8)
        with pytest.raises(ValueError, match="bivariate"):
            mispricing_index(rc.ClaytonCopula(2.0, dim=3), np.full((3, 3), 0.5))

    def test_backtest_runs_and_is_causal(self) -> None:
        mv = rc.CopulaDistribution(rc.ClaytonCopula(4.0), [stats.norm(0, 0.01)] * 2)
        r = mv.rvs(600, random_state=0)
        result = backtest_pairs(r, rc.ClaytonCopula(), train=250, refit_every=25)

        assert np.isfinite(result.annualised_sharpe)
        assert result.returns.shape == (600,)
        # No position can be taken before the first training window completes.
        assert np.all(result.positions[: 250 + 1] == 0)
        # A flat position earns exactly zero, never a stale return.
        assert np.all(result.returns[result.positions == 0] == 0.0)

    @pytest.mark.slow
    def test_no_look_ahead_bias_on_unexploitable_data(self) -> None:
        """On IID data the strategy must earn approximately nothing.

        This is the sharpest available check on the backtest. The simulated
        pairs have dependence but no mean reversion, so there is nothing for a
        relative-value signal to capture and the expected return of every trade
        is exactly zero. A strategy that peeked at time-t data when forming the
        time-t signal would show a spuriously positive Sharpe here; a causal one
        shows noise around zero.
        """
        mv = rc.CopulaDistribution(rc.ClaytonCopula(5.0), [stats.norm(0, 0.012)] * 2)
        means = []
        for seed in range(8):
            r = mv.rvs(900, random_state=seed)
            result = backtest_pairs(r, rc.ClaytonCopula(), train=250, refit_every=25)
            active = result.returns[result.positions != 0]
            if active.size > 20:
                means.append(active.mean() / active.std(ddof=1) * np.sqrt(active.size))

        # Per-run t-statistics should straddle zero, not sit systematically above.
        assert abs(float(np.mean(means))) < 2.0

    def test_backtest_rejects_short_or_malformed_input(self) -> None:
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            backtest_pairs(np.zeros((100, 3)), rc.ClaytonCopula())
        with pytest.raises(ValueError, match="more than"):
            backtest_pairs(np.zeros((50, 2)), rc.ClaytonCopula(), train=250)


# ======================================================================
# Portfolio optimisation
# ======================================================================


class TestPortfolioOptimisation:
    def test_min_variance_matches_the_closed_form(self) -> None:
        """Unconstrained, the answer is inv(S) 1 / (1' inv(S) 1)."""
        rng = np.random.default_rng(0)
        r = rng.multivariate_normal(
            np.zeros(3),
            np.array([[4.0, 1.0, 0.5], [1.0, 9.0, 2.0], [0.5, 2.0, 16.0]]) * 1e-4,
            size=5000,
        )
        w = min_variance_weights(r, bounds=(-1.0, 2.0))
        cov = np.cov(r, rowvar=False)
        inv_ones = np.linalg.solve(cov, np.ones(3))
        closed = inv_ones / inv_ones.sum()
        assert np.allclose(w, closed, atol=1e-4)

    def test_min_variance_survives_tiny_covariances(self) -> None:
        """Daily returns give objectives ~1e-4, below SLSQP's default ftol.

        Unscaled, the optimiser returned equal weights without moving.
        """
        rng = np.random.default_rng(0)
        r = np.column_stack([rng.normal(0, 0.01, 2000), rng.normal(0, 0.03, 2000)])
        w = min_variance_weights(r)
        assert w[0] > 0.8
        cov = np.cov(r, rowvar=False)
        assert w @ cov @ w < np.full(2, 0.5) @ cov @ np.full(2, 0.5)

    def test_cvar_optimiser_avoids_the_fat_tail(self) -> None:
        """Same mean, same variance, different tail -- mean-variance cannot see
        the difference and mean-CVaR can."""
        rng = np.random.default_rng(0)
        thin = rng.normal(0.001, 0.02, 8000)
        fat = rng.standard_t(2.2, 8000)
        fat = 0.001 + fat / fat.std() * 0.02
        scenarios = np.column_stack([thin, fat])

        cvar_w = mean_cvar_weights(scenarios, alpha=0.95)
        assert cvar_w[0] > cvar_w[1]

    def test_weights_sum_to_one_and_respect_bounds(self) -> None:
        r = simulate_returns(
            rc.StudentCopula(0.4, dim=4, df=5),
            stats.norm(0.0005, 0.012),
            n=4000,
            random_state=0,
        )
        w = mean_cvar_weights(r, bounds=(0.05, 0.5))
        assert abs(w.sum() - 1.0) < 1e-8
        assert np.all(w >= 0.05 - 1e-8) and np.all(w <= 0.5 + 1e-8)

    def test_target_return_is_met(self) -> None:
        rng = np.random.default_rng(0)
        r = np.column_stack(
            [
                rng.normal(0.0003, 0.01, 4000),
                rng.normal(0.0012, 0.025, 4000),
            ]
        )
        target = 0.0008
        w = mean_cvar_weights(r, target_return=target)
        assert w @ r.mean(axis=0) >= target - 1e-8

    def test_infeasible_target_is_reported(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0.0005, 0.01, (2000, 2))
        with pytest.raises(ValueError, match="infeasible"):
            mean_cvar_weights(r, target_return=1.0)

    def test_cvar_optimum_beats_equal_weights(self) -> None:
        r = simulate_returns(
            rc.ClaytonCopula(4.0, dim=4),
            stats.norm(0.0005, 0.015),
            n=8000,
            random_state=0,
        )
        w = mean_cvar_weights(r, alpha=0.95)
        equal = np.full(4, 0.25)
        assert expected_shortfall(-(r @ w), 0.95) <= expected_shortfall(-(r @ equal), 0.95) + 1e-12

    def test_frontier_is_monotone(self) -> None:
        rng = np.random.default_rng(0)
        r = np.column_stack(
            [
                rng.normal(0.0004, 0.01, 3000),
                rng.normal(0.0010, 0.02, 3000),
            ]
        )
        mu, cvar, weights = efficient_frontier(r, n_points=10)
        assert np.all(np.diff(mu) >= -1e-9)
        assert np.all(np.diff(cvar) >= -1e-9)
        assert weights.shape[1] == 2


class TestRadialSimplex:
    """McNeil and Neslehova's decomposition.

    The angular part is uniform on the simplex for *every* Archimedean copula in
    *every* dimension at *every* parameter -- all the family-specific
    information is in the radial part. That universality is the property worth
    testing, because it is what makes a nonparametric test of Archimedeanity
    possible at all.
    """

    @pytest.mark.parametrize(
        "copula",
        [
            rc.ClaytonCopula(0.5, dim=3),
            rc.ClaytonCopula(6.0, dim=3),
            rc.GumbelCopula(1.5, dim=4),
            rc.GumbelCopula(4.0, dim=4),
            rc.FrankCopula(8.0, dim=3),
            rc.JoeCopula(2.5, dim=3),
        ],
        ids=lambda c: f"{type(c).__name__}{c.params[0]:g}d{c.dim}",
    )
    def test_angular_part_is_uniform_on_the_simplex(self, copula: rc.Copula) -> None:
        from rcopula.transforms import radial_simplex

        _, angular = radial_simplex(copula, copula.rvs(20_000, random_state=0))
        np.testing.assert_allclose(angular.sum(axis=1), 1.0, atol=1e-12)
        # A uniform point on the d-simplex has mean 1/d and variance
        # (d-1)/(d^2 (d+1)) in every coordinate.
        d = copula.dim
        np.testing.assert_allclose(angular.mean(axis=0), 1 / d, atol=0.01)
        expected_variance = (d - 1) / (d**2 * (d + 1))
        np.testing.assert_allclose(angular.var(axis=0), expected_variance, rtol=0.06)

    def test_radial_and_angular_parts_are_independent(self) -> None:
        from rcopula.transforms import radial_simplex

        copula = rc.ClaytonCopula(2.0, dim=3)
        radial, angular = radial_simplex(copula, copula.rvs(50_000, random_state=0))
        for j in range(3):
            assert abs(float(np.corrcoef(radial, angular[:, j])[0, 1])) < 0.02

    def test_radial_part_is_the_sum_of_the_inverse_generator(self) -> None:
        from rcopula.transforms import radial_simplex

        copula = rc.GumbelCopula(2.0, dim=3)
        u = copula.rvs(200, random_state=0)
        radial, _ = radial_simplex(copula, u)
        expected = sum(copula.generator.ipsi(u[:, j], 2.0) for j in range(3))
        np.testing.assert_allclose(radial, expected, rtol=1e-12)

    def test_refuses_a_non_archimedean_copula(self) -> None:
        from rcopula.transforms import radial_simplex

        with pytest.raises(TypeError, match="Archimedean"):
            radial_simplex(rc.GaussianCopula(0.5), np.full((4, 2), 0.5))

    def test_rejects_a_dimension_mismatch(self) -> None:
        from rcopula.transforms import radial_simplex

        with pytest.raises(ValueError, match="columns"):
            radial_simplex(rc.ClaytonCopula(2.0, dim=3), np.full((4, 2), 0.5))


class TestHtrafo:
    """The Hering-Hofert transform.

    Under the null it produces independent uniforms, so both properties are
    checked -- uniformity alone would pass for a transform that got the
    dependence wrong.
    """

    @pytest.mark.parametrize(
        "copula",
        [
            rc.ClaytonCopula(2.0, dim=3),
            rc.ClaytonCopula(0.5, dim=5),
            rc.GumbelCopula(2.5, dim=4),
            rc.FrankCopula(6.0, dim=4),
            rc.JoeCopula(3.0, dim=3),
        ],
        ids=lambda c: f"{type(c).__name__}d{c.dim}",
    )
    def test_each_component_is_uniform(self, copula: rc.Copula) -> None:
        from rcopula.transforms import htrafo

        y = htrafo(copula, copula.rvs(20_000, random_state=0))
        assert y.shape == (20_000, copula.dim)
        for j in range(copula.dim):
            assert stats.kstest(y[:, j], "uniform").pvalue > 0.001

    @pytest.mark.parametrize(
        "copula",
        [rc.ClaytonCopula(2.0, dim=4), rc.GumbelCopula(2.5, dim=4), rc.FrankCopula(6.0, dim=4)],
        ids=lambda c: type(c).__name__,
    )
    def test_the_components_are_independent(self, copula: rc.Copula) -> None:
        from rcopula.transforms import htrafo

        y = htrafo(copula, copula.rvs(20_000, random_state=0))
        off_diagonal = np.abs(rc.cor_kendall(y) - np.eye(copula.dim))
        # At n = 20000 the sampling standard deviation of a Kendall tau is about
        # 0.0047, so 0.02 is four of them.
        assert off_diagonal.max() < 0.02

    def test_the_wrong_copula_is_detected(self) -> None:
        from rcopula.transforms import htrafo

        clayton = rc.ClaytonCopula(2.0, dim=5)
        y = htrafo(clayton, rc.GumbelCopula(3.0, dim=5).rvs(4000, random_state=0))
        assert stats.kstest(y.ravel(), "uniform").pvalue < 1e-6

    def test_the_wrong_parameter_is_detected(self) -> None:
        from rcopula.transforms import htrafo

        wrong = rc.ClaytonCopula(6.0, dim=5).rvs(4000, random_state=0)
        y = htrafo(rc.ClaytonCopula(0.5, dim=5), wrong)
        assert stats.kstest(y.ravel(), "uniform").pvalue < 1e-6

    def test_survives_a_dimension_the_rosenblatt_transform_would_not(self) -> None:
        # The point of the transform: no high-order generator derivatives, so
        # d = 100 is no harder than d = 3.
        from rcopula.transforms import htrafo

        copula = rc.GumbelCopula(2.0, dim=100)
        y = htrafo(copula, copula.rvs(400, random_state=0))
        assert y.shape == (400, 100)
        assert np.all(np.isfinite(y))
        assert np.all((y >= 0) & (y <= 1))
        assert stats.kstest(y.ravel(), "uniform").pvalue > 0.001

    def test_output_stays_in_the_unit_cube(self) -> None:
        from rcopula.transforms import htrafo

        copula = rc.ClaytonCopula(8.0, dim=6)
        y = htrafo(copula, copula.rvs(2000, random_state=0))
        assert np.all((y >= 0.0) & (y <= 1.0))


class TestFitLambda:
    """Nonparametric tail dependence.

    The estimator is accurate where there is genuine tail dependence and
    *systematically positive* where there is none, because the Gaussian
    copula's coefficient converges to zero only logarithmically. Both halves
    are tested: the first is the point of the function, the second is the trap
    it exists to make visible.
    """

    @pytest.mark.parametrize(
        "copula",
        [
            rc.ClaytonCopula.from_tau(0.5),
            rc.GumbelCopula.from_tau(0.5),
            rc.StudentCopula.from_tau(0.5, df=3.0),
        ],
        ids=lambda c: type(c).__name__,
    )
    def test_it_recovers_a_nonzero_coefficient(self, copula: rc.Copula) -> None:
        from rcopula.dependence import fit_lambda

        estimate = fit_lambda(copula.rvs(20_000, random_state=0))
        truth = copula.lambda_()
        for observed, expected in (
            (estimate.lower, truth.lower),
            (estimate.upper, truth.upper),
        ):
            if expected > 0.0:
                assert abs(observed - expected) < 0.1

    @pytest.mark.parametrize(
        "copula",
        [
            rc.ClaytonCopula.from_tau(0.5),
            rc.GumbelCopula.from_tau(0.5),
            rc.FrankCopula.from_tau(0.5),
            rc.GaussianCopula.from_tau(0.5),
        ],
        ids=lambda c: type(c).__name__,
    )
    def test_a_zero_coefficient_comes_back_small_but_not_zero(self, copula: rc.Copula) -> None:
        # This is the finite-threshold bias, not an error: at any usable k the
        # corner still holds more points than an asymptotically independent
        # copula eventually would. Roughly 0.1 to 0.25 here. Reporting it as a
        # clean zero would require a threshold no finite sample supports, which
        # is exactly why `path` exists.
        from rcopula.dependence import fit_lambda

        estimate = fit_lambda(copula.rvs(20_000, random_state=0))
        truth = copula.lambda_()
        for observed, expected in (
            (estimate.lower, truth.lower),
            (estimate.upper, truth.upper),
        ):
            if expected == 0.0:
                assert 0.0 <= observed < 0.3

    def test_it_distinguishes_the_two_tails(self) -> None:
        from rcopula.dependence import fit_lambda

        clayton = fit_lambda(rc.ClaytonCopula.from_tau(0.5).rvs(20_000, random_state=0))
        gumbel = fit_lambda(rc.GumbelCopula.from_tau(0.5).rvs(20_000, random_state=0))
        assert clayton.lower > 3 * clayton.upper
        assert gumbel.upper > 3 * gumbel.lower

    def test_the_gaussian_estimate_is_positive_and_should_be_distrusted(self) -> None:
        # True lambda is exactly zero. Anyone reading a single threshold
        # estimate as an answer would conclude otherwise, which is why the path
        # is returned and the docstring says to look at it.
        from rcopula.dependence import fit_lambda

        estimate = fit_lambda(rc.GaussianCopula.from_tau(0.5).rvs(20_000, random_state=0))
        assert rc.GaussianCopula.from_tau(0.5).lambda_().upper == 0.0
        assert estimate.upper > 0.1

    def test_the_path_is_what_separates_them(self) -> None:
        # A real coefficient gives a plateau; the Gaussian's slides towards zero
        # as the threshold is pushed out, so the slope against log k is the
        # discriminator that the point estimate is not.
        from rcopula.dependence import fit_lambda

        def slope(u: np.ndarray) -> float:
            path = fit_lambda(u).path
            keep = path[:, 0] >= 50
            return float(np.polyfit(np.log(path[keep, 0]), path[keep, 2], 1)[0])

        heavy = slope(rc.StudentCopula.from_tau(0.5, df=3.0).rvs(40_000, random_state=0))
        light = slope(rc.GaussianCopula.from_tau(0.5).rvs(40_000, random_state=0))
        assert heavy < 0.5 * light

    def test_independence_gives_nothing_in_either_tail(self) -> None:
        from rcopula.dependence import fit_lambda

        estimate = fit_lambda(rc.IndependenceCopula(2).rvs(20_000, random_state=0))
        assert estimate.lower < 0.1
        assert estimate.upper < 0.1

    def test_the_standard_error_shrinks_with_the_threshold(self) -> None:
        from rcopula.dependence import fit_lambda

        data = rc.ClaytonCopula.from_tau(0.5).rvs(20_000, random_state=0)
        assert fit_lambda(data, k=2000).lower_se < fit_lambda(data, k=100).lower_se

    def test_the_log_method_also_works(self) -> None:
        from rcopula.dependence import fit_lambda

        data = rc.GumbelCopula.from_tau(0.5).rvs(20_000, random_state=0)
        estimate = fit_lambda(data, method="log")
        assert estimate.method == "log"
        assert abs(estimate.upper - rc.GumbelCopula.from_tau(0.5).lambda_().upper) < 0.1

    @pytest.mark.parametrize(
        "copula",
        [rc.ClaytonCopula.from_tau(0.5), rc.GumbelCopula.from_tau(0.5)],
        ids=lambda c: type(c).__name__,
    )
    def test_the_two_estimators_agree(self, copula: rc.Copula) -> None:
        # They are different formulas for the same quantity, so a disagreement
        # means one of them is wrong. An earlier version of the log estimator
        # returned the two tails swapped, and this is what catches that.
        from rcopula.dependence import fit_lambda

        data = copula.rvs(20_000, random_state=0)
        counting = fit_lambda(data, method="schmidt-stadtmuller")
        logarithmic = fit_lambda(data, method="log")
        assert abs(counting.lower - logarithmic.lower) < 0.02
        assert abs(counting.upper - logarithmic.upper) < 0.02

    @pytest.mark.parametrize("method", ["schmidt-stadtmuller", "log"])
    def test_the_boundary_cases_are_exact(self, method: str) -> None:
        # Comonotone is 1 in both tails and independence is 0 in both, with no
        # threshold effect either way -- so these are the two places an
        # estimator has no excuse.
        from rcopula.dependence import fit_lambda

        comonotone = fit_lambda(rc.FrechetUpperCopula(2).rvs(20_000, random_state=0), method=method)
        assert comonotone.lower == pytest.approx(1.0, abs=1e-12)
        assert comonotone.upper == pytest.approx(1.0, abs=1e-12)

        independent = fit_lambda(
            rc.IndependenceCopula(2).rvs(20_000, random_state=0), method=method
        )
        assert independent.lower == pytest.approx(0.0, abs=1e-12)
        assert independent.upper == pytest.approx(0.0, abs=1e-12)

    def test_estimates_stay_in_the_unit_interval(self) -> None:
        from rcopula.dependence import fit_lambda

        for copula in (rc.ClaytonCopula(20.0), rc.IndependenceCopula(2), rc.GumbelCopula(12.0)):
            estimate = fit_lambda(copula.rvs(5000, random_state=0), k=40)
            assert 0.0 <= estimate.lower <= 1.0
            assert 0.0 <= estimate.upper <= 1.0
            assert np.all((estimate.path[:, 1:] >= 0.0) & (estimate.path[:, 1:] <= 1.0))

    def test_summary_warns_about_the_threshold(self) -> None:
        from rcopula.dependence import fit_lambda

        text = fit_lambda(rc.ClaytonCopula(2.0).rvs(2000, random_state=0)).summary()
        assert "threshold estimates" in text
        assert "95% lower" in text

    def test_rejects_a_non_bivariate_input(self) -> None:
        from rcopula.dependence import fit_lambda

        with pytest.raises(ValueError, match="bivariate"):
            fit_lambda(rc.ClaytonCopula(2.0, dim=3).rvs(200, random_state=0))

    def test_rejects_an_impossible_threshold(self) -> None:
        from rcopula.dependence import fit_lambda

        with pytest.raises(ValueError, match="1 <= k < n"):
            fit_lambda(rc.ClaytonCopula(2.0).rvs(200, random_state=0), k=500)

    def test_rejects_an_unknown_method(self) -> None:
        from rcopula.dependence import fit_lambda

        with pytest.raises(ValueError, match="schmidt-stadtmuller"):
            fit_lambda(rc.ClaytonCopula(2.0).rvs(200, random_state=0), method="hill")


class TestToEmpiricalMargins:
    """The inverse of pseudo_obs: uniforms back onto a sample's own margins."""

    def test_it_keeps_the_copula_and_takes_the_margins(self) -> None:
        from rcopula.dependence import to_emp_margins

        rng = np.random.default_rng(0)
        history = rng.lognormal(size=(3000, 2)) * [1.0, 5.0]
        drawn = to_emp_margins(rc.ClaytonCopula(2.0).rvs(8000, random_state=0), history)
        # The dependence is the copula's, which the history did not have.
        assert float(rc.cor_kendall(drawn)[0, 1]) == pytest.approx(0.5, abs=0.03)
        # The margins are the history's.
        ratio = np.median(drawn, axis=0) / np.median(history, axis=0)
        assert np.all(np.abs(ratio - 1.0) < 0.1)

    def test_nothing_outside_the_reference_range_comes_out(self) -> None:
        # An empirical quantile function cannot extrapolate, which is a feature
        # for historical simulation and a trap for a tail study: a 99.9% number
        # computed this way can never exceed the worst loss already seen.
        from rcopula.dependence import to_emp_margins

        history = np.random.default_rng(0).standard_normal((500, 2))
        drawn = to_emp_margins(rc.GaussianCopula(0.5).rvs(20_000, random_state=0), history)
        assert drawn.max() <= history.max()
        assert drawn.min() >= history.min()

    def test_every_output_value_is_in_the_reference_sample(self) -> None:
        from rcopula.dependence import to_emp_margins

        history = np.random.default_rng(0).standard_normal((200, 2))
        drawn = to_emp_margins(rc.ClaytonCopula(2.0).rvs(1000, random_state=0), history)
        for j in range(2):
            assert set(np.unique(drawn[:, j])).issubset(set(history[:, j]))

    def test_it_round_trips_with_pseudo_obs(self) -> None:
        # to_emp_margins(pseudo_obs(x), x) should return x, up to the tie
        # convention at the largest observation.
        from rcopula.dependence import to_emp_margins

        data = np.random.default_rng(0).standard_normal((400, 3))
        back = to_emp_margins(np.asarray(rc.pseudo_obs(data)), data)
        assert np.mean(np.isclose(back, data)) > 0.99

    def test_a_column_mismatch_is_refused(self) -> None:
        from rcopula.dependence import to_emp_margins

        with pytest.raises(ValueError, match="must match"):
            to_emp_margins(np.full((10, 2), 0.5), np.zeros((50, 3)))

    def test_values_outside_the_unit_interval_are_refused(self) -> None:
        from rcopula.dependence import to_emp_margins

        with pytest.raises(ValueError, match="uniform"):
            to_emp_margins(np.array([[1.5, 0.5]]), np.zeros((50, 2)))


class TestRadialDistribution:
    """The radial part's law, which carries all the family-specific information.

    The angular half of the McNeil-Neslehova split is the same for every
    Archimedean copula in every dimension, so everything that distinguishes one
    family from another is in here. It is checked against the empirical
    distribution of the radial part of an actual sample, which is the only
    reference that does not go through the same identity.
    """

    CASES: ClassVar[list] = [
        rc.ClaytonCopula(2.0, dim=3),
        rc.ClaytonCopula(0.5, dim=2),
        rc.GumbelCopula(2.0, dim=4),
        rc.FrankCopula(5.0, dim=3),
        rc.JoeCopula(2.5, dim=3),
    ]

    @pytest.mark.parametrize("copula", CASES, ids=lambda c: f"{type(c).__name__}d{c.dim}")
    def test_it_matches_the_sampled_radial_part(self, copula: rc.Copula) -> None:
        from rcopula.transforms import radial_cdf, radial_simplex

        radii, _ = radial_simplex(copula, copula.rvs(100_000, random_state=0))
        for level in (0.1, 0.25, 0.5, 0.75, 0.9):
            x = float(np.quantile(radii, level))
            assert float(radial_cdf(copula, x)[0]) == pytest.approx(level, abs=0.01)

    @pytest.mark.parametrize("copula", CASES, ids=lambda c: f"{type(c).__name__}d{c.dim}")
    def test_the_quantile_inverts_the_cdf(self, copula: rc.Copula) -> None:
        from rcopula.transforms import radial_cdf, radial_ppf

        levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        np.testing.assert_allclose(
            radial_cdf(copula, radial_ppf(copula, levels)), levels, atol=1e-6
        )

    @pytest.mark.parametrize("copula", CASES, ids=lambda c: f"{type(c).__name__}d{c.dim}")
    def test_the_cdf_is_a_distribution_function(self, copula: rc.Copula) -> None:
        from rcopula.transforms import radial_cdf, radial_ppf

        # The grid has to reach as far as the family does. Clayton's radial part
        # is heavy enough that a fixed upper limit of 1e4 only gets to 0.981,
        # which is the distribution behaving correctly rather than a defect --
        # so the top of the grid comes from the quantile function.
        top = float(radial_ppf(copula, 0.9999)[0])
        grid = np.concatenate([[0.0], np.geomspace(1e-6, top, 80)])
        values = radial_cdf(copula, grid)
        assert np.all(np.diff(values) >= -1e-12)
        assert 0.0 <= values[0] <= 1e-9
        assert values[-1] > 0.999

    def test_the_quantile_is_increasing(self) -> None:
        from rcopula.transforms import radial_ppf

        levels = np.linspace(0.01, 0.99, 40)
        assert np.all(np.diff(radial_ppf(rc.ClaytonCopula(2.0, dim=3), levels)) > 0)

    def test_a_non_archimedean_copula_is_refused(self) -> None:
        from rcopula.transforms import radial_cdf, radial_ppf

        with pytest.raises(TypeError, match="Archimedean"):
            radial_cdf(rc.GaussianCopula(0.5), 1.0)
        with pytest.raises(TypeError, match="Archimedean"):
            radial_ppf(rc.GaussianCopula(0.5), 0.5)

    def test_a_negative_radius_is_refused(self) -> None:
        from rcopula.transforms import radial_cdf

        with pytest.raises(ValueError, match="negative"):
            radial_cdf(rc.ClaytonCopula(2.0), -1.0)

    def test_a_probability_outside_the_unit_interval_is_refused(self) -> None:
        from rcopula.transforms import radial_ppf

        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            radial_ppf(rc.ClaytonCopula(2.0), 1.4)
