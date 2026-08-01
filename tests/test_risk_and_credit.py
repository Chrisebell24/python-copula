"""Tests for the applied risk and credit utilities.

There is no R oracle for these -- R's ``copula`` package does not price CDOs --
so they are validated three other ways:

1. **Closed-form limits.** The Vasicek large-homogeneous-pool distribution is
   the exact limit of the one-factor Gaussian model, so a simulated portfolio
   with many names must converge to it.
2. **Structural identities.** Euler risk contributions must sum to portfolio
   ES; expected shortfall must dominate VaR; tranche losses must partition the
   portfolio loss; marginal default rates must equal the input PD whatever the
   copula.
3. **Qualitative orderings that follow from theory.** Tail dependence must
   raise senior tranche losses and lower equity ones; independence must
   diversify more than dependence; first-to-default must be worth more under
   independence and nth-to-default more under dependence.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.credit import (
    default_indicators,
    default_times,
    implied_correlation,
    nth_to_default_probability,
    portfolio_loss,
    tranche_expected_loss,
    tranche_loss,
    tranche_spread,
    vasicek_loss_cdf,
)
from rcopula.risk import (
    covar,
    delta_covar,
    diversification_benefit,
    expected_shortfall,
    marginal_expected_shortfall,
    risk_contributions,
    simulate_losses,
    stress_scenario,
    value_at_risk,
)

# ======================================================================
# Risk measures
# ======================================================================


class TestRiskMeasures:
    def test_var_is_the_quantile(self) -> None:
        x = np.arange(1, 1001, dtype=float)
        assert value_at_risk(x, 0.99) == 990.0
        assert value_at_risk(x, 0.50) == 500.0

    @pytest.mark.parametrize("alpha", [0.9, 0.95, 0.99, 0.999])
    def test_expected_shortfall_dominates_var(self, alpha: float) -> None:
        x = np.random.default_rng(0).standard_t(3, size=200_000)
        assert expected_shortfall(x, alpha) > value_at_risk(x, alpha)

    def test_var_is_monotone_in_the_level(self) -> None:
        x = np.random.default_rng(0).lognormal(size=50_000)
        levels = [0.5, 0.9, 0.95, 0.99, 0.999]
        values = [value_at_risk(x, a) for a in levels]
        assert all(np.diff(values) > 0)

    def test_expected_shortfall_is_subadditive(self) -> None:
        """The property VaR lacks and ES has -- diversification never hurts."""
        u = rc.GaussianCopula(0.3, dim=2).rvs(200_000, random_state=0)
        a, b = stats.lognorm(0.9).ppf(u[:, 0]), stats.lognorm(0.9).ppf(u[:, 1])
        combined = expected_shortfall(a + b, 0.99)
        separate = expected_shortfall(a, 0.99) + expected_shortfall(b, 0.99)
        assert combined <= separate * (1 + 1e-9)

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="alpha must lie"):
            value_at_risk([1.0, 2.0], 1.5)
        with pytest.raises(ValueError, match="empty"):
            value_at_risk([])


class TestPortfolioRisk:
    def test_portfolio_risk_orders_by_upper_tail_dependence(self) -> None:
        """The headline claim: same tau, same margins, different tail.

        Note the direction. In *loss* space it is the **upper** tail that
        matters, so Clayton -- which is lower-tail dependent -- lands below
        Gaussian rather than above it. Getting this backwards is the easiest way
        to understate capital, which is why the module docstring leads with it.
        """
        tau, margins = 0.5, stats.lognorm(0.8)
        es = {}
        for name, cop in (
            ("clayton", rc.ClaytonCopula.from_tau(tau, dim=5)),
            ("gaussian", rc.GaussianCopula.from_tau(tau, dim=5)),
            ("student", rc.StudentCopula.from_tau(tau, dim=5, df=4)),
            ("gumbel", rc.GumbelCopula.from_tau(tau, dim=5)),
        ):
            losses = simulate_losses(cop, margins, n=120_000, random_state=0)
            es[name] = expected_shortfall(losses, 0.99)

        # Ordering follows lambda_U: 0.00, 0.00, 0.40, 0.59.
        assert es["clayton"] < es["gaussian"] < es["student"] < es["gumbel"]
        assert es["gumbel"] > 1.5 * es["clayton"]

    def test_risk_contributions_sum_to_portfolio_es(self) -> None:
        """The Euler property: allocated capital adds up to capital held."""
        cop, margins = rc.ClaytonCopula(3.0, dim=4), stats.lognorm(0.7)
        parts = risk_contributions(cop, margins, n=80_000, random_state=0)
        total = expected_shortfall(simulate_losses(cop, margins, n=80_000, random_state=0), 0.99)
        assert parts.sum() == pytest.approx(total, rel=1e-9)

    def test_contributions_reflect_exposure(self) -> None:
        weights = np.array([0.7, 0.1, 0.1, 0.1])
        parts = risk_contributions(
            rc.GaussianCopula(0.4, dim=4),
            stats.lognorm(0.6),
            weights,
            n=60_000,
            random_state=0,
        )
        assert parts[0] == max(parts)

    def test_independence_diversifies_more_than_dependence(self) -> None:
        margins = stats.lognorm(0.6)
        free = diversification_benefit(rc.IndependenceCopula(4), margins, n=60_000, random_state=0)
        tied = diversification_benefit(
            rc.ClaytonCopula(8.0, dim=4), margins, n=60_000, random_state=0
        )
        assert free["benefit_pct"] > tied["benefit_pct"] > 0

    def test_comonotone_gives_no_benefit(self) -> None:
        result = diversification_benefit(
            rc.FrechetUpperCopula(3), stats.lognorm(0.5), n=40_000, random_state=0
        )
        assert result["benefit_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_weight_mismatch_is_reported(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            simulate_losses(rc.ClaytonCopula(2.0, dim=3), stats.norm(), [0.5, 0.5], n=10)


class TestSystemicRisk:
    @pytest.fixture(scope="class")
    @staticmethod
    def samples() -> tuple[np.ndarray, np.ndarray]:
        return (
            rc.ClaytonCopula(6.0).rvs(80_000, random_state=0),
            rc.IndependenceCopula(2).rvs(80_000, random_state=0),
        )

    def test_covar_exceeds_unconditional_var_under_dependence(self, samples) -> None:
        tied, _ = samples
        assert covar(tied[:, 0], tied[:, 1], 0.95) > value_at_risk(tied[:, 0], 0.95) * 0.99

    def test_covar_is_higher_when_the_firm_is_connected(self, samples) -> None:
        tied, free = samples
        assert covar(tied[:, 0], tied[:, 1]) > covar(free[:, 0], free[:, 1])

    def test_delta_covar_is_near_zero_under_independence(self, samples) -> None:
        _, free = samples
        assert abs(delta_covar(free[:, 0], free[:, 1])) < 0.05

    def test_mes_detects_connection(self, samples) -> None:
        tied, free = samples
        assert marginal_expected_shortfall(tied[:, 0], tied[:, 1]) > marginal_expected_shortfall(
            free[:, 0], free[:, 1]
        )

    def test_mismatched_lengths_are_reported(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            covar(np.zeros(10), np.zeros(5))
        with pytest.raises(ValueError, match="same length"):
            marginal_expected_shortfall(np.zeros(10), np.zeros(5))


class TestStressScenario:
    def test_stressing_one_factor_drags_the_others(self) -> None:
        draws = stress_scenario(
            rc.ClaytonCopula(6.0, dim=3),
            stats.norm(),
            {0: 0.99},
            n=150_000,
            random_state=0,
        )
        assert draws[:, 1].mean() > 0.5
        assert draws[:, 2].mean() > 0.5

    def test_independence_leaves_the_others_alone(self) -> None:
        draws = stress_scenario(
            rc.IndependenceCopula(3),
            stats.norm(),
            {0: 0.99},
            n=150_000,
            random_state=0,
        )
        assert abs(draws[:, 1].mean()) < 0.15

    def test_rejects_bad_conditioning(self) -> None:
        with pytest.raises(ValueError, match="at least one factor"):
            stress_scenario(rc.ClaytonCopula(2.0), stats.norm(), {}, n=100)
        with pytest.raises(ValueError, match="outside"):
            stress_scenario(rc.ClaytonCopula(2.0), stats.norm(), {5: 0.9}, n=100)
        with pytest.raises(ValueError, match="satisfy the conditioning"):
            stress_scenario(rc.ClaytonCopula(2.0), stats.norm(), {0: 0.99}, n=50, band=1e-6)


# ======================================================================
# Credit
# ======================================================================


class TestCreditPortfolio:
    def test_marginal_default_rate_is_the_input_pd(self) -> None:
        """True for every copula -- dependence changes the joint, not the margins."""
        for cop in (
            rc.GaussianCopula(0.3, dim=50),
            rc.ClaytonCopula(2.0, dim=50),
            rc.IndependenceCopula(50),
        ):
            d = default_indicators(cop, 0.02, 60_000, random_state=0)
            assert d.mean() == pytest.approx(0.02, abs=0.002), cop.name

    def test_expected_loss_is_pd_times_lgd(self) -> None:
        loss = portfolio_loss(rc.GaussianCopula(0.2, dim=100), 0.03, 0.6, n=60_000, random_state=0)
        assert loss.mean() == pytest.approx(0.03 * 0.6, abs=0.002)
        assert 0.0 <= loss.min() and loss.max() <= 1.0

    def test_default_times_have_exponential_margins(self) -> None:
        t = default_times(rc.ClaytonCopula(3.0, dim=4), 0.02, 30_000, random_state=0)
        for j in range(4):
            assert stats.kstest(t[:, j], stats.expon(scale=50).cdf).pvalue > 0.01

    def test_heterogeneous_inputs_are_honoured(self) -> None:
        pd_ = np.array([0.01, 0.05, 0.10])
        d = default_indicators(rc.GaussianCopula(0.2, dim=3), pd_, 80_000, random_state=0)
        assert np.allclose(d.mean(axis=0), pd_, atol=0.005)

    def test_bad_input_is_reported(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            default_indicators(rc.GaussianCopula(0.2, dim=3), 1.5, 10)
        with pytest.raises(ValueError, match="length"):
            portfolio_loss(rc.GaussianCopula(0.2, dim=3), [0.1, 0.2], n=10)
        with pytest.raises(ValueError, match="positive"):
            default_times(rc.GaussianCopula(0.2, dim=3), -0.1, 10)


class TestVasicekLimit:
    """The closed-form anchor: simulation must converge to it."""

    @pytest.mark.parametrize(("pd_", "rho"), [(0.02, 0.1), (0.05, 0.2), (0.10, 0.3)])
    def test_simulation_converges_to_the_closed_form(self, pd_: float, rho: float) -> None:
        loss = portfolio_loss(
            rc.GaussianCopula(rho, dim=800), pd_, lgd=1.0, n=40_000, random_state=0
        )
        for x in (0.05, 0.10, 0.20):
            assert np.mean(loss <= x) == pytest.approx(
                float(vasicek_loss_cdf(x, pd_, rho)), abs=0.03
            )

    def test_it_is_a_proper_distribution_function(self) -> None:
        x = np.linspace(0.001, 0.999, 200)
        cdf = vasicek_loss_cdf(x, 0.05, 0.2)
        assert np.all(np.diff(cdf) >= -1e-12)
        assert cdf[0] < 0.01 and cdf[-1] > 0.99

    def test_higher_correlation_fattens_the_tail(self) -> None:
        low = float(vasicek_loss_cdf(0.20, 0.05, 0.05))
        high = float(vasicek_loss_cdf(0.20, 0.05, 0.40))
        assert high < low  # less mass below 0.20 means more above it

    def test_rejects_out_of_range_parameters(self) -> None:
        with pytest.raises(ValueError, match="default_prob"):
            vasicek_loss_cdf(0.1, 1.5, 0.2)
        with pytest.raises(ValueError, match="correlation"):
            vasicek_loss_cdf(0.1, 0.05, 1.5)


class TestTranches:
    def test_tranche_losses_partition_the_portfolio_loss(self) -> None:
        """Tranche losses weighted by width must reconstruct the total."""
        loss = portfolio_loss(
            rc.GaussianCopula(0.2, dim=100), 0.05, lgd=1.0, n=20_000, random_state=0
        )
        edges = [0.0, 0.03, 0.07, 0.15, 0.30, 1.0]
        total = np.zeros_like(loss)
        for a, b in pairwise(edges):
            total += (b - a) * tranche_loss(loss, a, b)
        assert np.allclose(total, loss, atol=1e-12)

    def test_equity_absorbs_before_senior(self) -> None:
        loss = portfolio_loss(rc.GaussianCopula(0.2, dim=100), 0.05, n=40_000, random_state=0)
        equity = tranche_expected_loss(loss, 0.0, 0.03)
        mezz = tranche_expected_loss(loss, 0.03, 0.07)
        senior = tranche_expected_loss(loss, 0.15, 0.30)
        assert equity > mezz > senior

    def test_tail_dependence_reprices_the_senior_tranche(self) -> None:
        """The 2008 lesson, in one assertion.

        At identical Kendall's tau and identical marginal default probability,
        a Gaussian copula prices the senior tranche at a fraction of what a
        tail-dependent copula does -- because it says many simultaneous
        defaults essentially cannot happen.
        """
        tau, d, pd_ = 0.3, 100, 0.05
        gauss = portfolio_loss(
            rc.GaussianCopula.from_tau(tau, dim=d), pd_, 0.6, n=60_000, random_state=0
        )
        clayton = portfolio_loss(
            rc.ClaytonCopula.from_tau(tau, dim=d), pd_, 0.6, n=60_000, random_state=0
        )
        senior_gauss = tranche_expected_loss(gauss, 0.15, 0.30)
        senior_clayton = tranche_expected_loss(clayton, 0.15, 0.30)
        assert senior_clayton > 2 * senior_gauss

        # ...while the equity tranche moves the other way.
        assert tranche_expected_loss(gauss, 0.0, 0.03) > tranche_expected_loss(clayton, 0.0, 0.03)

    def test_spread_ordering_follows_expected_loss(self) -> None:
        loss = portfolio_loss(rc.GaussianCopula(0.2, dim=100), 0.05, n=40_000, random_state=0)
        assert tranche_spread(loss, 0.0, 0.03) > 10 * tranche_spread(loss, 0.15, 0.30)

    def test_rejects_invalid_attachment_points(self) -> None:
        with pytest.raises(ValueError, match="attachment < detachment"):
            tranche_loss([0.1], 0.5, 0.3)
        with pytest.raises(ValueError, match="attachment < detachment"):
            tranche_loss([0.1], -0.1, 0.3)


class TestBasketDefault:
    def test_first_to_default_is_worth_more_under_independence(self) -> None:
        """Independence gives many chances for a first default."""
        free = nth_to_default_probability(rc.IndependenceCopula(10), 0.05, 1, 60_000, 0)
        tied = nth_to_default_probability(rc.ClaytonCopula(5.0, dim=10), 0.05, 1, 60_000, 0)
        assert free > tied

    def test_nth_to_default_inverts_that_ordering(self) -> None:
        """Dependence is what makes many defaults arrive together."""
        free = nth_to_default_probability(rc.IndependenceCopula(10), 0.05, 5, 60_000, 0)
        tied = nth_to_default_probability(rc.ClaytonCopula(5.0, dim=10), 0.05, 5, 60_000, 0)
        assert tied > free

    def test_probability_decreases_in_n(self) -> None:
        cop = rc.GaussianCopula(0.3, dim=10)
        probs = [nth_to_default_probability(cop, 0.05, k, 40_000, 0) for k in (1, 2, 5, 8)]
        assert all(np.diff(probs) <= 0)

    def test_rejects_out_of_range_n(self) -> None:
        with pytest.raises(ValueError, match="n_th must lie"):
            nth_to_default_probability(rc.GaussianCopula(0.2, dim=5), 0.05, 9, 100)


class TestImpliedCorrelation:
    def test_round_trips(self) -> None:
        loss = portfolio_loss(
            rc.GaussianCopula(0.25, dim=200), 0.05, lgd=1.0, n=40_000, random_state=1
        )
        el = tranche_expected_loss(loss, 0.03, 0.07)
        rho = implied_correlation(el, 0.05, 0.03, 0.07, n_names=200, random_state=1)
        assert rho == pytest.approx(0.25, abs=0.06)

    def test_unattainable_target_is_reported(self) -> None:
        with pytest.raises(ValueError, match="not attainable"):
            implied_correlation(0.999, 0.01, 0.15, 0.30, n_names=100, n=5_000)
