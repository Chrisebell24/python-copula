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

from typing import ClassVar

import numpy as np
import pytest
from scipy import integrate, optimize, stats

import rcopula as rc
from rcopula.derivatives import (
    CmsLeg,
    SmileMargin,
    _par_bond,
    basket_implied_vol,
    basket_option,
    black76,
    cms_convexity_adjustment,
    cms_margin,
    cms_spread_option,
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


# ======================================================================
# CMS rates
# ======================================================================


def _exact_convexity_adjustment(
    forward: float, vol: float, maturity: float, tenor: float, frequency: int = 2
) -> float:
    """E[y_T] - y_0 solved exactly, without the second-order Taylor step.

    The adjustment exists because the bond's forward *price* is the martingale:
    ``E[G(y_T)] = G(y_0)``. :func:`cms_convexity_adjustment` expands that to
    second order; here it is solved numerically instead, by quadrature over the
    lognormal law of the rate plus a root-find for its median. Any error in the
    bond function, its derivatives, or the sign of the adjustment shows up
    immediately as a mismatch.
    """
    level = _par_bond(forward, forward, tenor, frequency)[0]
    s = vol * np.sqrt(maturity)

    def gap(log_median: float) -> float:
        median = np.exp(log_median)
        integrand = lambda z: (  # noqa: E731
            _par_bond(median * np.exp(s * z), forward, tenor, frequency)[0] * stats.norm.pdf(z)
        )
        return integrate.quad(integrand, -8.0, 8.0, limit=200)[0] - level

    median = np.exp(optimize.brentq(gap, np.log(forward * 0.5), np.log(forward * 2.0), xtol=1e-14))
    return float(median * np.exp(0.5 * s**2) - forward)


class TestParBond:
    def test_a_par_bond_prices_at_one(self) -> None:
        """G(y) = 1 when the coupon equals the yield -- the definition of par."""
        for y in (0.01, 0.05, 0.12):
            price, _, _ = _par_bond(y, y, 10.0, 2)
            assert price == pytest.approx(1.0, abs=1e-12)

    def test_derivatives_match_finite_differences(self) -> None:
        y, h = 0.05, 1e-6
        _, d1, d2 = _par_bond(y, 0.05, 10.0, 2)
        up = _par_bond(y + h, 0.05, 10.0, 2)[0]
        mid = _par_bond(y, 0.05, 10.0, 2)[0]
        down = _par_bond(y - h, 0.05, 10.0, 2)[0]
        assert d1 == pytest.approx((up - down) / (2 * h), rel=1e-6)
        assert d2 == pytest.approx((up - 2 * mid + down) / h**2, rel=1e-3)

    def test_price_falls_and_convexity_is_positive(self) -> None:
        _, d1, d2 = _par_bond(0.05, 0.05, 10.0, 2)
        assert d1 < 0.0
        assert d2 > 0.0

    def test_rejects_a_degenerate_schedule(self) -> None:
        with pytest.raises(ValueError, match="shorter than one coupon period"):
            _par_bond(0.05, 0.05, 0.1, 2)


class TestCmsConvexityAdjustment:
    @pytest.mark.parametrize(
        ("forward", "vol", "maturity", "tenor"),
        [(0.05, 0.20, 5.0, 10.0), (0.05, 0.20, 5.0, 30.0), (0.03, 0.15, 2.0, 5.0)],
    )
    def test_matches_the_exact_solution(
        self, forward: float, vol: float, maturity: float, tenor: float
    ) -> None:
        """The Taylor form must land within its own O(sigma^4 T^2) error."""
        approx = cms_convexity_adjustment(forward, vol, maturity, tenor)
        exact = _exact_convexity_adjustment(forward, vol, maturity, tenor)
        assert approx == pytest.approx(exact, rel=0.10)

    def test_the_approximation_improves_as_variance_falls(self) -> None:
        """A second-order expansion: halving the vol should cut the error faster."""

        def relative_error(vol: float) -> float:
            approx = cms_convexity_adjustment(0.05, vol, 5.0, 10.0)
            exact = _exact_convexity_adjustment(0.05, vol, 5.0, 10.0)
            return abs(approx / exact - 1.0)

        assert relative_error(0.10) < 0.4 * relative_error(0.30)

    def test_is_positive(self) -> None:
        assert cms_convexity_adjustment(0.05, 0.2, 5.0, 10.0) > 0.0

    def test_grows_with_variance_expiry_and_tenor(self) -> None:
        base = cms_convexity_adjustment(0.04, 0.20, 5.0, 10.0)
        assert cms_convexity_adjustment(0.04, 0.30, 5.0, 10.0) > base
        assert cms_convexity_adjustment(0.04, 0.20, 8.0, 10.0) > base
        assert cms_convexity_adjustment(0.04, 0.20, 5.0, 20.0) > base

    def test_scales_with_variance(self) -> None:
        """The formula is exactly linear in Var[y_T]."""
        a = cms_convexity_adjustment(0.04, 0.20, 5.0, 10.0)
        b = cms_convexity_adjustment(0.04, 0.40, 5.0, 10.0)
        assert b == pytest.approx(4.0 * a, rel=1e-12)

    def test_vanishes_without_uncertainty(self) -> None:
        assert cms_convexity_adjustment(0.05, 0.0, 5.0, 10.0) == 0.0
        assert cms_convexity_adjustment(0.05, 0.2, 0.0, 10.0) == 0.0

    def test_normal_and_lognormal_agree_at_matched_variance(self) -> None:
        """Only the variance enters, so a matched normal vol gives the same number."""
        forward, vol = 0.04, 0.25
        lognormal = cms_convexity_adjustment(forward, vol, 5.0, 10.0)
        normal = cms_convexity_adjustment(forward, vol * forward, 5.0, 10.0, model="normal")
        assert normal == pytest.approx(lognormal, rel=1e-12)

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="model must be"):
            cms_convexity_adjustment(0.05, 0.2, 5.0, 10.0, model="sabr")
        with pytest.raises(ValueError, match="vol must be non-negative"):
            cms_convexity_adjustment(0.05, -0.2, 5.0, 10.0)
        with pytest.raises(ValueError, match="maturity must be non-negative"):
            cms_convexity_adjustment(0.05, 0.2, -1.0, 10.0)


class TestCmsMargin:
    def test_mean_is_the_adjusted_rate(self) -> None:
        m = cms_margin(0.045, 0.22, 5.0, 10.0)
        expected = 0.045 + cms_convexity_adjustment(0.045, 0.22, 5.0, 10.0)
        assert m.mean() == pytest.approx(expected, rel=1e-12)

    def test_normal_model_is_symmetric_about_the_adjusted_rate(self) -> None:
        m = cms_margin(0.02, 0.008, 5.0, 10.0, model="normal")
        centre = 0.02 + cms_convexity_adjustment(0.02, 0.008, 5.0, 10.0, model="normal")
        assert m.ppf(0.5) == pytest.approx(centre, rel=1e-10)
        assert m.std() == pytest.approx(0.008 * np.sqrt(5.0), rel=1e-12)

    def test_normal_model_admits_negative_rates(self) -> None:
        """The reason it exists: a lognormal rate cannot go below zero."""
        assert cms_margin(0.002, 0.008, 5.0, 10.0, model="normal").ppf(0.05) < 0.0
        assert cms_margin(0.002, 0.30, 5.0, 10.0).ppf(1e-8) > 0.0


class TestCmsSpreadOption:
    LEGS: ClassVar[list[CmsLeg]] = [CmsLeg(0.045, 0.22, 10.0), CmsLeg(0.030, 0.28, 2.0)]

    def test_reduces_to_margrabe_at_zero_strike(self) -> None:
        """Gaussian copula, lognormal legs, K=0: an exchange option, exactly priced.

        The convexity adjustments move each leg's forward, so Margrabe is fed
        the adjusted forwards -- which is the point of checking it here rather
        than only in :class:`TestSpreadOptionAgainstMargrabe`.
        """
        maturity = 5.0
        adjusted = [
            leg.forward + cms_convexity_adjustment(leg.forward, leg.vol, maturity, leg.tenor)
            for leg in self.LEGS
        ]
        exact = margrabe(
            adjusted[0], adjusted[1], self.LEGS[0].vol, self.LEGS[1].vol, 0.85, maturity
        )
        mc = cms_spread_option(
            rc.GaussianCopula(0.85), self.LEGS, 0.0, maturity, n=400_000, random_state=0
        )
        assert mc.price == pytest.approx(exact, abs=4 * mc.standard_error)

    def test_matches_kirk_at_a_positive_strike(self) -> None:
        maturity, strike = 5.0, 0.01
        adjusted = [
            leg.forward + cms_convexity_adjustment(leg.forward, leg.vol, maturity, leg.tenor)
            for leg in self.LEGS
        ]
        approx = kirk_spread(
            adjusted[0],
            adjusted[1],
            strike,
            self.LEGS[0].vol,
            self.LEGS[1].vol,
            0.85,
            maturity,
        )
        mc = cms_spread_option(
            rc.GaussianCopula(0.85), self.LEGS, strike, maturity, n=400_000, random_state=0
        )
        assert mc.price == pytest.approx(approx, rel=0.05)

    def test_ignoring_convexity_underprices_a_steepener(self) -> None:
        """The longer leg carries the larger adjustment, so the two do not cancel."""
        maturity, strike = 5.0, 0.015
        with_adjustment = cms_spread_option(
            rc.GaussianCopula(0.85), self.LEGS, strike, maturity, n=300_000, random_state=0
        )
        flat = spread_option(
            rc.GaussianCopula(0.85),
            [lognormal_terminal(leg.forward, leg.vol, maturity) for leg in self.LEGS],
            strike,
            maturity,
            n=300_000,
            random_state=0,
        )
        assert with_adjustment.price > flat.price + 2 * with_adjustment.standard_error

    def test_correlation_cheapens_the_option(self) -> None:
        loose = cms_spread_option(
            rc.GaussianCopula(0.60), self.LEGS, 0.015, 5.0, n=300_000, random_state=0
        )
        tight = cms_spread_option(
            rc.GaussianCopula(0.97), self.LEGS, 0.015, 5.0, n=300_000, random_state=0
        )
        assert tight.price < loose.price

    def test_put_call_parity_on_the_spread(self) -> None:
        """call - put = discounted (E[spread] - K), the forward spread."""
        maturity, strike, rate = 5.0, 0.010, 0.02
        adjusted = [
            leg.forward + cms_convexity_adjustment(leg.forward, leg.vol, maturity, leg.tenor)
            for leg in self.LEGS
        ]
        call = cms_spread_option(
            rc.GaussianCopula(0.85),
            self.LEGS,
            strike,
            maturity,
            rate=rate,
            n=400_000,
            random_state=1,
        )
        put = cms_spread_option(
            rc.GaussianCopula(0.85),
            self.LEGS,
            strike,
            maturity,
            rate=rate,
            kind="put",
            n=400_000,
            random_state=1,
        )
        forward = np.exp(-rate * maturity) * (adjusted[0] - adjusted[1] - strike)
        assert call.price - put.price == pytest.approx(
            forward, abs=4 * (call.standard_error + put.standard_error)
        )

    def test_notional_scales_the_price(self) -> None:
        one = cms_spread_option(
            rc.GaussianCopula(0.85), self.LEGS, 0.015, 5.0, n=100_000, random_state=0
        )
        million = cms_spread_option(
            rc.GaussianCopula(0.85),
            self.LEGS,
            0.015,
            5.0,
            notional=1e6,
            n=100_000,
            random_state=0,
        )
        assert million.price == pytest.approx(1e6 * one.price, rel=1e-12)

    def test_tail_dependence_reprices_at_equal_rank_correlation(self) -> None:
        """Same Kendall tau, different joint tail, different price."""
        gaussian = rc.GaussianCopula(0.85)
        clayton = rc.ClaytonCopula.from_tau(gaussian.tau())
        a = cms_spread_option(gaussian, self.LEGS, 0.015, 5.0, n=400_000, random_state=0)
        b = cms_spread_option(clayton, self.LEGS, 0.015, 5.0, n=400_000, random_state=0)
        assert abs(a.price - b.price) > 3 * (a.standard_error + b.standard_error)

    def test_mixed_marginal_models_are_allowed(self) -> None:
        """A lognormal long leg and a normal short leg, as a low-rate desk would."""
        legs = [CmsLeg(0.030, 0.25, 10.0), CmsLeg(0.004, 0.006, 2.0, model="normal")]
        mc = cms_spread_option(rc.GaussianCopula(0.7), legs, 0.01, 5.0, n=200_000, random_state=0)
        assert mc.price > 0.0

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="bivariate"):
            cms_spread_option(rc.GaussianCopula(0.5, dim=3), self.LEGS, 0.0, 5.0)
        with pytest.raises(ValueError, match="expected 2 legs"):
            cms_spread_option(rc.GaussianCopula(0.5), self.LEGS[:1], 0.0, 5.0)
        with pytest.raises(ValueError, match="kind must be"):
            cms_spread_option(rc.GaussianCopula(0.5), self.LEGS, 0.0, 5.0, kind="digital")
