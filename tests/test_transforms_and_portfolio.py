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
