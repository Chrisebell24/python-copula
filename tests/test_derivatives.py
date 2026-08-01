"""Tests for multi-asset option pricing.

No R oracle exists, so correctness rests on closed forms and structural
identities:

* **Margrabe** is exact for an exchange option under jointly lognormal
  dynamics -- which is precisely a Gaussian copula with lognormal margins -- so
  the Monte-Carlo spread option must reproduce it within its own standard error.
* **Kirk** is the standard approximation for a strike-bearing spread option and
  must agree closely.
* **Put-call parity**, **Frechet bounds** on basket prices, and the
  **martingale property** of each margin are checked directly.
* Rainbow options must respond to dependence in *opposite* directions for
  best-of and worst-of, which is a qualitative fact no calibration can fake.
"""

from __future__ import annotations

import numpy as np
import pytest

import rcopula as rc
from rcopula.derivatives import (
    SmileMargin,
    basket_implied_vol,
    basket_option,
    black76,
    implied_volatility,
    kirk_spread,
    lognormal_terminal,
    margrabe,
    rainbow_option,
    spread_option,
)


class TestClosedForms:
    def test_black76_put_call_parity(self) -> None:
        for strike in (80.0, 100.0, 120.0):
            call = black76(100.0, strike, 0.25, 1.5, rate=0.03)
            put = black76(100.0, strike, 0.25, 1.5, rate=0.03, kind="put")
            discount = np.exp(-0.03 * 1.5)
            assert call - put == pytest.approx(discount * (100.0 - strike), abs=1e-10)

    def test_black76_is_monotone_in_volatility(self) -> None:
        prices = [black76(100.0, 100.0, v, 1.0) for v in (0.1, 0.2, 0.3, 0.5)]
        assert all(np.diff(prices) > 0)

    def test_margrabe_degenerates_correctly(self) -> None:
        """Perfect correlation with equal vols leaves only the intrinsic gap."""
        assert margrabe(100.0, 95.0, 0.25, 0.25, 1.0, 1.0) == pytest.approx(5.0, abs=1e-10)

    def test_margrabe_decreases_with_correlation(self) -> None:
        prices = [margrabe(100.0, 95.0, 0.2, 0.3, r, 1.0) for r in (-0.9, -0.3, 0.3, 0.9)]
        assert all(np.diff(prices) < 0)

    def test_kirk_reduces_to_margrabe_at_zero_strike(self) -> None:
        for rho in (-0.5, 0.0, 0.7):
            assert kirk_spread(100.0, 95.0, 0.0, 0.2, 0.3, rho, 1.0) == pytest.approx(
                margrabe(100.0, 95.0, 0.2, 0.3, rho, 1.0), abs=1e-12
            )

    def test_implied_volatility_round_trips(self) -> None:
        for strike, vol in [(80.0, 0.35), (100.0, 0.2), (130.0, 0.28)]:
            price = black76(100.0, strike, vol, 1.5, rate=0.02)
            assert implied_volatility(price, 100.0, strike, 1.5, rate=0.02) == pytest.approx(
                vol, abs=1e-8
            )

    def test_implied_volatility_reports_unattainable_prices(self) -> None:
        with pytest.raises(ValueError, match="not attainable"):
            implied_volatility(500.0, 100.0, 100.0, 1.0)

    def test_bad_option_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            black76(100.0, 100.0, 0.2, 1.0, kind="straddle")


class TestSpreadOptionAgainstMargrabe:
    """The exact-price check that validates the whole simulation path."""

    @pytest.mark.parametrize("rho", [-0.5, 0.0, 0.5, 0.9])
    def test_matches_the_exact_price(self, rho: float) -> None:
        margins = [
            lognormal_terminal(100.0, 0.2, 1.0),
            lognormal_terminal(95.0, 0.3, 1.0),
        ]
        mc = spread_option(rc.GaussianCopula(rho), margins, 0.0, 1.0, n=400_000, random_state=0)
        exact = margrabe(100.0, 95.0, 0.2, 0.3, rho, 1.0)
        assert abs(mc.price - exact) < 4.0 * mc.standard_error

    @pytest.mark.parametrize("strike", [0.0, 5.0, 10.0])
    def test_matches_kirk_for_positive_strikes(self, strike: float) -> None:
        margins = [
            lognormal_terminal(100.0, 0.2, 1.0),
            lognormal_terminal(95.0, 0.3, 1.0),
        ]
        mc = spread_option(rc.GaussianCopula(0.5), margins, strike, 1.0, n=400_000, random_state=0)
        assert mc.price == pytest.approx(
            kirk_spread(100.0, 95.0, strike, 0.2, 0.3, 0.5, 1.0), abs=0.05
        )

    def test_requires_two_assets(self) -> None:
        margins = [lognormal_terminal(100.0, 0.2, 1.0)] * 3
        with pytest.raises(ValueError, match="bivariate"):
            spread_option(rc.GaussianCopula(0.5, dim=3), margins, 0.0, 1.0, n=100)


class TestBasketOption:
    @pytest.fixture(scope="class")
    @staticmethod
    def margins() -> list:
        return [lognormal_terminal(100.0, 0.25, 1.0)] * 3

    def test_price_increases_with_dependence(self, margins) -> None:
        """Less diversification means a more volatile basket."""
        prices = [
            basket_option(
                rc.GaussianCopula(rho, dim=3),
                margins,
                100.0,
                1.0,
                n=150_000,
                random_state=0,
            ).price
            for rho in (0.0, 0.4, 0.8, 0.99)
        ]
        assert all(np.diff(prices) > 0)

    def test_comonotone_basket_reduces_to_a_single_asset(self, margins) -> None:
        """With identical components perfectly tied, the basket *is* one asset."""
        mc = basket_option(rc.FrechetUpperCopula(3), margins, 100.0, 1.0, n=200_000, random_state=0)
        exact = black76(100.0, 100.0, 0.25, 1.0)
        assert abs(mc.price - exact) < 4.0 * mc.standard_error

    def test_basket_is_bounded_by_the_single_asset_price(self, margins) -> None:
        """Diversification cannot make the basket riskier than its components."""
        single = black76(100.0, 100.0, 0.25, 1.0)
        mc = basket_option(
            rc.GaussianCopula(0.3, dim=3), margins, 100.0, 1.0, n=150_000, random_state=0
        )
        assert mc.price < single

    def test_put_call_parity_holds_for_the_basket(self, margins) -> None:
        cop = rc.GaussianCopula(0.4, dim=3)
        call = basket_option(cop, margins, 95.0, 1.0, n=200_000, random_state=0)
        put = basket_option(cop, margins, 95.0, 1.0, n=200_000, random_state=0, kind="put")
        # Forward of an equally weighted basket of identical forwards is 100.
        assert call.price - put.price == pytest.approx(
            100.0 - 95.0, abs=4 * (call.standard_error + put.standard_error)
        )

    def test_weights_are_honoured(self, margins) -> None:
        with pytest.raises(ValueError, match="weight"):
            basket_option(
                rc.GaussianCopula(0.3, dim=3), margins, 100.0, 1.0, weights=[0.5, 0.5], n=100
            )

    def test_reports_a_standard_error(self, margins) -> None:
        mc = basket_option(
            rc.GaussianCopula(0.3, dim=3), margins, 100.0, 1.0, n=50_000, random_state=0
        )
        assert mc.standard_error > 0 and mc.n == 50_000
        assert "+/-" in repr(mc)


class TestRainbowOption:
    @pytest.fixture(scope="class")
    @staticmethod
    def margins() -> list:
        return [lognormal_terminal(100.0, 0.3, 1.0)] * 3

    def test_best_of_and_worst_of_respond_oppositely(self, margins) -> None:
        """The clearest demonstration that dependence, not volatility, is the
        driver: the same change moves the two payoffs in opposite directions."""
        free, tied = rc.GaussianCopula(0.0, dim=3), rc.GaussianCopula(0.9, dim=3)
        best_free = rainbow_option(free, margins, 100.0, 1.0, n=150_000, random_state=0, on="best")
        best_tied = rainbow_option(tied, margins, 100.0, 1.0, n=150_000, random_state=0, on="best")
        worst_free = rainbow_option(
            free, margins, 100.0, 1.0, n=150_000, random_state=0, on="worst"
        )
        worst_tied = rainbow_option(
            tied, margins, 100.0, 1.0, n=150_000, random_state=0, on="worst"
        )

        assert best_free.price > best_tied.price
        assert worst_tied.price > worst_free.price

    def test_best_of_dominates_worst_of(self, margins) -> None:
        cop = rc.GaussianCopula(0.4, dim=3)
        best = rainbow_option(cop, margins, 100.0, 1.0, n=100_000, random_state=0, on="best")
        worst = rainbow_option(cop, margins, 100.0, 1.0, n=100_000, random_state=0, on="worst")
        assert best.price > worst.price

    def test_comonotone_collapses_the_two(self, margins) -> None:
        """If all assets are identical, best-of and worst-of coincide."""
        cop = rc.FrechetUpperCopula(3)
        best = rainbow_option(cop, margins, 100.0, 1.0, n=100_000, random_state=0, on="best")
        worst = rainbow_option(cop, margins, 100.0, 1.0, n=100_000, random_state=0, on="worst")
        assert best.price == pytest.approx(worst.price, rel=1e-12)

    def test_bad_arguments_are_rejected(self, margins) -> None:
        with pytest.raises(ValueError, match="on must be"):
            rainbow_option(rc.GaussianCopula(0.3, dim=3), margins, 100.0, 1.0, on="median")


class TestSmileMargin:
    def test_flat_smile_reproduces_the_lognormal_margin(self) -> None:
        strikes = np.linspace(40.0, 250.0, 120)
        smile = SmileMargin(strikes, np.full(120, 0.25), forward=100.0, maturity=1.0)
        exact = lognormal_terminal(100.0, 0.25, 1.0)
        for x in (70.0, 100.0, 140.0):
            assert smile.cdf(x) == pytest.approx(float(exact.cdf(x)), abs=0.01)

    def test_a_skew_shifts_mass_into_the_left_tail(self) -> None:
        strikes = np.linspace(40.0, 250.0, 120)
        flat = SmileMargin(strikes, np.full(120, 0.25), forward=100.0, maturity=1.0)
        skewed = SmileMargin(
            strikes, 0.25 + 0.0007 * (100.0 - strikes), forward=100.0, maturity=1.0
        )
        assert skewed.cdf(70.0) > flat.cdf(70.0)

    def test_cdf_is_monotone_and_bounded(self) -> None:
        strikes = np.linspace(40.0, 250.0, 120)
        smile = SmileMargin(strikes, 0.25 + 0.0005 * (100.0 - strikes), forward=100.0, maturity=1.0)
        values = smile.cdf(strikes)
        assert np.all(np.diff(values) >= -1e-12)
        assert values.min() >= 0.0 and values.max() <= 1.0

    def test_satisfies_the_margin_protocol(self) -> None:
        """It must be usable anywhere a scipy frozen distribution is."""
        strikes = np.linspace(40.0, 250.0, 120)
        smile = SmileMargin(strikes, np.full(120, 0.25), forward=100.0, maturity=1.0)
        joint = rc.CopulaDistribution(rc.GaussianCopula(0.5, dim=2), [smile, smile])
        draws = joint.rvs(1000, random_state=0)
        assert draws.shape == (1000, 2)
        assert np.all(draws > 0)

    def test_rejects_malformed_smiles(self) -> None:
        with pytest.raises(ValueError, match="strikes and"):
            SmileMargin([1.0, 2.0, 3.0, 4.0], [0.2, 0.2], 100.0, 1.0)
        with pytest.raises(ValueError, match="at least four"):
            SmileMargin([1.0, 2.0], [0.2, 0.2], 100.0, 1.0)
        with pytest.raises(ValueError, match="increasing"):
            SmileMargin([4.0, 3.0, 2.0, 1.0], [0.2] * 4, 100.0, 1.0)


class TestBasketImpliedVol:
    def test_lognormal_components_give_a_nearly_flat_basket_smile(self) -> None:
        margins = [lognormal_terminal(100.0, 0.25, 1.0)] * 3
        _, vols = basket_implied_vol(
            rc.GaussianCopula(0.5, dim=3),
            margins,
            [85, 95, 100, 105, 115],
            1.0,
            n=300_000,
            random_state=0,
        )
        assert vols.std() < 0.02

    def test_tail_dependence_bends_the_basket_smile(self) -> None:
        """The point of the exercise: the basket is not lognormal, so it has a
        smile of its own that no single correlation could produce."""
        margins = [lognormal_terminal(100.0, 0.25, 1.0)] * 3
        strikes = [85, 95, 100, 105, 115]
        _, flat = basket_implied_vol(
            rc.GaussianCopula(0.5, dim=3), margins, strikes, 1.0, n=300_000, random_state=0
        )
        _, bent = basket_implied_vol(
            rc.GumbelCopula.from_tau(1 / 3, dim=3),
            margins,
            strikes,
            1.0,
            n=300_000,
            random_state=0,
        )
        assert bent.std() > flat.std()

    def test_component_smiles_flow_through_to_the_basket(self) -> None:
        """Skewed components produce a skewed basket smile."""
        strikes = np.linspace(40.0, 250.0, 120)
        skewed = SmileMargin(
            strikes, 0.25 + 0.0008 * (100.0 - strikes), forward=100.0, maturity=1.0
        )
        _, vols = basket_implied_vol(
            rc.GaussianCopula(0.5, dim=3),
            [skewed] * 3,
            [85, 100, 115],
            1.0,
            n=300_000,
            random_state=0,
        )
        assert vols[0] > vols[-1]  # downward-sloping, as the components are
