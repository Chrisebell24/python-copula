"""Tests for the insurance, actuarial and operational-risk utilities.

Validation is by exact moment formulas and structural identities, since there is
no R oracle:

* The compound distribution has **exact** first two moments,
  ``E[S] = E[N]E[X]`` and ``Var[S] = E[N]Var[X] + Var[N]E[X]^2``.
* Rank reordering must preserve every marginal value exactly while reproducing
  the copula's Kendall tau.
* Disjoint reinsurance layers must partition the loss.
* Diversification must be larger under independence than under dependence.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.insurance import (
    aggregate_loss,
    catastrophe_bond,
    excess_of_loss,
    layer_statistics,
    operational_risk_capital,
    reinsurance_premium,
)
from rcopula.risk import rank_reorder


class TestRankReorder:
    def test_margins_are_preserved_exactly(self) -> None:
        """Not approximately: the same values come back, only rearranged."""
        rng = np.random.default_rng(0)
        x = np.column_stack(
            [rng.lognormal(size=5000), rng.exponential(size=5000), rng.gamma(2, size=5000)]
        )
        y = rank_reorder(x, rc.ClaytonCopula(4.0, dim=3), random_state=0)
        assert np.array_equal(np.sort(x, axis=0), np.sort(y, axis=0))

    @pytest.mark.parametrize("tau", [0.2, 0.5, 0.8])
    def test_dependence_becomes_the_copulas(self, tau: float) -> None:
        rng = np.random.default_rng(0)
        x = np.column_stack([rng.lognormal(size=8000), rng.exponential(size=8000)])
        y = rank_reorder(x, rc.ClaytonCopula.from_tau(tau), random_state=0)
        assert stats.kendalltau(y[:, 0], y[:, 1]).statistic == pytest.approx(tau, abs=0.03)

    def test_dimension_mismatch_is_reported(self) -> None:
        with pytest.raises(ValueError, match="columns"):
            rank_reorder(np.zeros((10, 3)), rc.ClaytonCopula(2.0))


class TestAggregateLoss:
    def test_compound_mean_matches_the_exact_formula(self) -> None:
        freq, sev = stats.poisson(40), stats.lognorm(0.8, scale=1000)
        s = aggregate_loss(freq, sev, n=120_000, random_state=0)
        assert s.mean() == pytest.approx(freq.mean() * sev.mean(), rel=0.02)

    def test_compound_variance_matches_the_exact_formula(self) -> None:
        """Var[S] = E[N] Var[X] + Var[N] E[X]^2."""
        freq, sev = stats.poisson(40), stats.lognorm(0.8, scale=1000)
        s = aggregate_loss(freq, sev, n=200_000, random_state=0)
        expected = freq.mean() * sev.var() + freq.var() * sev.mean() ** 2
        assert s.var() == pytest.approx(expected, rel=0.05)

    def test_works_for_non_poisson_frequency(self) -> None:
        """Negative binomial, the standard over-dispersed alternative."""
        freq, sev = stats.nbinom(20, 0.4), stats.expon(scale=500)
        s = aggregate_loss(freq, sev, n=100_000, random_state=0)
        assert s.mean() == pytest.approx(freq.mean() * sev.mean(), rel=0.03)
        expected = freq.mean() * sev.var() + freq.var() * sev.mean() ** 2
        assert s.var() == pytest.approx(expected, rel=0.06)

    def test_zero_frequency_gives_zero_loss(self) -> None:
        s = aggregate_loss(stats.poisson(0), stats.lognorm(1.0), n=100, random_state=0)
        assert np.all(s == 0.0)

    def test_is_non_negative_and_right_skewed(self) -> None:
        s = aggregate_loss(
            stats.poisson(15), stats.lognorm(1.5, scale=1000), n=50_000, random_state=0
        )
        assert np.all(s >= 0)
        assert stats.skew(s) > 0


class TestOperationalRisk:
    @staticmethod
    def _cells(k: int = 3) -> list:
        return [(stats.poisson(30), stats.lognorm(1.2, scale=500))] * k

    def test_dependence_consumes_the_diversification_credit(self) -> None:
        cells = self._cells()
        free = operational_risk_capital(cells, None, n=60_000, random_state=0)
        tied = operational_risk_capital(
            cells, rc.GaussianCopula(0.8, dim=3), n=60_000, random_state=0
        )
        assert tied["capital"] > free["capital"]
        assert free["diversification_benefit"] > tied["diversification_benefit"] > 0

    def test_comonotone_cells_leave_no_benefit(self) -> None:
        """Perfect dependence means capital is simply additive."""
        result = operational_risk_capital(
            self._cells(), rc.FrechetUpperCopula(3), n=60_000, random_state=0
        )
        assert result["diversification_benefit"] == pytest.approx(0.0, abs=1e-6)

    def test_capital_is_var_less_expected_loss(self) -> None:
        """The Basel definition: expected loss is provisioned, not capitalised."""
        r = operational_risk_capital(self._cells(), None, n=40_000, random_state=0)
        assert r["capital"] == pytest.approx(r["var"] - r["expected_loss"])

    def test_expected_shortfall_exceeds_var(self) -> None:
        r = operational_risk_capital(self._cells(), None, n=40_000, random_state=0)
        assert r["expected_shortfall"] > r["var"]

    def test_tail_dependence_costs_more_than_correlation_alone(self) -> None:
        """Same Kendall tau, different tail: the point of the whole exercise."""
        cells = self._cells()
        tau = 0.4
        gauss = operational_risk_capital(
            cells, rc.GaussianCopula.from_tau(tau, dim=3), n=80_000, random_state=0
        )
        gumbel = operational_risk_capital(
            cells, rc.GumbelCopula.from_tau(tau, dim=3), n=80_000, random_state=0
        )
        assert gumbel["capital"] > gauss["capital"]

    def test_rejects_mismatched_input(self) -> None:
        with pytest.raises(ValueError, match="at least one cell"):
            operational_risk_capital([], None, n=100)
        with pytest.raises(ValueError, match="dim="):
            operational_risk_capital(self._cells(2), rc.GaussianCopula(0.5, dim=3), n=100)


class TestReinsurance:
    def test_layer_recovery(self) -> None:
        losses = np.array([0.0, 50.0, 150.0, 400.0, 1000.0])
        assert np.allclose(excess_of_loss(losses, 100.0, 200.0), [0.0, 0.0, 50.0, 200.0, 200.0])

    def test_disjoint_layers_partition_the_loss(self) -> None:
        losses = aggregate_loss(
            stats.poisson(40), stats.lognorm(1.0, scale=1000), n=20_000, random_state=0
        )
        edges = [0.0, 20_000.0, 50_000.0, 100_000.0, 1e12]
        total = sum(excess_of_loss(losses, a, b - a) for a, b in pairwise(edges))
        assert np.allclose(total, losses, rtol=1e-12)

    def test_higher_layers_are_cheaper_and_rarer(self) -> None:
        losses = aggregate_loss(
            stats.poisson(40), stats.lognorm(1.0, scale=1000), n=60_000, random_state=0
        )
        low = layer_statistics(losses, 60_000, 20_000)
        high = layer_statistics(losses, 150_000, 20_000)
        assert low.attachment_probability > high.attachment_probability
        assert low.expected_loss > high.expected_loss
        assert low.expected_loss_ratio > high.expected_loss_ratio

    def test_attachment_dominates_exhaustion(self) -> None:
        losses = aggregate_loss(
            stats.poisson(30), stats.lognorm(1.2, scale=800), n=40_000, random_state=0
        )
        s = layer_statistics(losses, 40_000, 30_000)
        assert s.attachment_probability >= s.exhaustion_probability
        assert 0.0 <= s.expected_loss_ratio <= 1.0

    @pytest.mark.parametrize(
        "method", ["expected_value", "standard_deviation", "expected_shortfall"]
    )
    def test_premium_exceeds_expected_loss(self, method: str) -> None:
        losses = aggregate_loss(
            stats.poisson(40), stats.lognorm(1.2, scale=1000), n=40_000, random_state=0
        )
        stat = layer_statistics(losses, 100_000, 50_000)
        premium = reinsurance_premium(losses, 100_000, 50_000, method=method)
        assert premium > stat.expected_loss

    def test_tail_sensitive_principles_charge_more(self) -> None:
        losses = aggregate_loss(
            stats.poisson(40), stats.lognorm(1.4, scale=1000), n=40_000, random_state=0
        )
        ev = reinsurance_premium(losses, 100_000, 50_000, method="expected_value")
        es = reinsurance_premium(losses, 100_000, 50_000, method="expected_shortfall")
        assert es > ev

    def test_rejects_bad_layers_and_methods(self) -> None:
        with pytest.raises(ValueError, match="attachment >= 0"):
            excess_of_loss([1.0], -1.0, 10.0)
        with pytest.raises(ValueError, match="limit > 0"):
            excess_of_loss([1.0], 1.0, 0.0)
        with pytest.raises(ValueError, match="method must be"):
            reinsurance_premium([1.0, 2.0], 0.5, 1.0, method="wang")


class TestCatastropheBond:
    @staticmethod
    def _losses() -> np.ndarray:
        return aggregate_loss(
            stats.poisson(20), stats.lognorm(1.5, scale=2000), n=80_000, random_state=0
        )

    def test_remote_layers_have_lower_expected_loss_and_higher_multiples(self) -> None:
        losses = self._losses()
        near = catastrophe_bond(losses, 60_000, 120_000)
        far = catastrophe_bond(losses, 200_000, 300_000)
        assert far["expected_loss"] < near["expected_loss"]
        assert far["multiple"] > near["multiple"]

    def test_expected_loss_is_bounded_by_the_attachment_probability(self) -> None:
        """Principal can only be lost once the trigger is breached."""
        bond = catastrophe_bond(self._losses(), 100_000, 200_000)
        assert bond["expected_loss"] <= bond["attachment_probability"]
        assert bond["exhaustion_probability"] <= bond["attachment_probability"]

    def test_expected_return_is_spread_less_expected_loss(self) -> None:
        bond = catastrophe_bond(self._losses(), 150_000, 250_000, coupon=0.08, risk_free=0.03)
        assert bond["spread"] == pytest.approx(0.05)
        assert bond["expected_return"] == pytest.approx(0.05 - bond["expected_loss"])

    def test_a_never_triggered_bond_has_an_infinite_multiple(self) -> None:
        bond = catastrophe_bond(self._losses(), 1e12, 2e12)
        assert bond["expected_loss"] == 0.0
        assert np.isinf(bond["multiple"])

    def test_rejects_inverted_triggers(self) -> None:
        with pytest.raises(ValueError, match="attachment < exhaustion"):
            catastrophe_bond([1.0], 200.0, 100.0)
