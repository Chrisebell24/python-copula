"""Multi-asset options that respect each underlying's own volatility smile.

This is the case where a copula earns its keep: a single correlation number
cannot price a basket whose components have non-lognormal marginals, because
there is no joint lognormal to correlate.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show

import rcopula as rc
from rcopula.derivatives import (
    CmsLeg,
    SmileMargin,
    basket_implied_vol,
    basket_option,
    cms_convexity_adjustment,
    cms_spread_option,
    kirk_spread,
    lognormal_terminal,
    margrabe,
    rainbow_option,
    spread_option,
)

heading("Validation first: Margrabe is exact, so the simulation must match it")

margins = [lognormal_terminal(100.0, 0.20, 1.0), lognormal_terminal(95.0, 0.30, 1.0)]
for correlation in (-0.5, 0.0, 0.5, 0.9):
    mc = spread_option(rc.GaussianCopula(correlation), margins, 0.0, 1.0, n=400_000, random_state=0)
    exact = margrabe(100.0, 95.0, 0.20, 0.30, correlation, 1.0)
    check(
        f"rho={correlation:+.1f}: MC {mc.price:.4f} vs Margrabe {exact:.4f} "
        f"(within {abs(mc.price - exact) / mc.standard_error:.1f} standard errors)",
        abs(mc.price - exact) < 4 * mc.standard_error,
    )

heading("And Kirk for a strike-bearing spread")

for strike in (5.0, 10.0):
    mc = spread_option(rc.GaussianCopula(0.5), margins, strike, 1.0, n=400_000, random_state=0)
    approx = kirk_spread(100.0, 95.0, strike, 0.20, 0.30, 0.5, 1.0)
    check(
        f"K={strike}: MC {mc.price:.4f} vs Kirk {approx:.4f}",
        abs(mc.price - approx) / approx < 0.03,
    )

heading("Basket and rainbow options")

three = [lognormal_terminal(100.0, 0.25, 1.0)] * 3
cop = rc.GaussianCopula(0.5, dim=3)
for strike in (90.0, 100.0, 110.0):
    price = basket_option(cop, three, strike, 1.0, n=200_000, random_state=0)
    print(f"  basket call, K = {strike:5.0f}:  {price.price:8.4f} +/- {price.standard_error:.4f}")

for on in ("best", "worst"):
    price = rainbow_option(cop, three, 100.0, 1.0, on=on, n=200_000, random_state=0)
    show(f"{on}-of-three call", price.price)

best = rainbow_option(cop, three, 100.0, 1.0, on="best", n=200_000, random_state=0).price
worst = rainbow_option(cop, three, 100.0, 1.0, on="worst", n=200_000, random_state=0).price
check("best-of is worth more than worst-of", best > worst)

# Correlation moves them in opposite directions -- the diagnostic that says
# a rainbow option is really a bet on dependence.
tight = rc.GaussianCopula(0.9, dim=3)
best_tight = rainbow_option(tight, three, 100.0, 1.0, on="best", n=200_000, random_state=0).price
worst_tight = rainbow_option(tight, three, 100.0, 1.0, on="worst", n=200_000, random_state=0).price
check("higher correlation cheapens best-of", best_tight < best)
check("and enriches worst-of", worst_tight > worst)

heading("The basket smile implied by the components' own smiles")

# Give each component the marginal its own smile implies (Breeden-Litzenberger),
# then choose the dependence. A single correlation cannot do this.
strikes = np.linspace(60, 150, 40)
skewed = SmileMargin(strikes, 0.25 + 0.0008 * (100 - strikes), forward=100.0, maturity=1.0)
flat = SmileMargin(strikes, np.full(strikes.size, 0.25), forward=100.0, maturity=1.0)

check(
    "a downward-sloping smile puts more mass in the left tail",
    skewed.cdf(70.0) > flat.cdf(70.0),
)

basket_strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
for label, margin_set in [("flat smiles", [flat] * 3), ("skewed smiles", [skewed] * 3)]:
    _, vols = basket_implied_vol(
        rc.GaussianCopula(0.5, dim=3), margin_set, basket_strikes, 1.0, n=200_000, random_state=0
    )
    shown = "  ".join(f"{k:.0f}: {v:.4f}" for k, v in zip(basket_strikes, vols, strict=True))
    print(f"  {label:<16} {shown}")

_, flat_vols = basket_implied_vol(
    rc.GaussianCopula(0.5, dim=3), [flat] * 3, basket_strikes, 1.0, n=200_000, random_state=0
)
_, skew_vols = basket_implied_vol(
    rc.GaussianCopula(0.5, dim=3), [skewed] * 3, basket_strikes, 1.0, n=200_000, random_state=0
)
check("flat component smiles give a nearly flat basket smile", flat_vols.std() < 0.02)
check("skewed ones give a skewed basket smile", skew_vols[0] > skew_vols[-1] + 0.02)

# And the dependence bends it too, independently of the margins.
_, gumbel_vols = basket_implied_vol(
    rc.GumbelCopula.from_tau(1 / 3, dim=3),
    [flat] * 3,
    basket_strikes,
    1.0,
    n=200_000,
    random_state=0,
)
check(
    "tail-dependent dependence bends the basket smile on its own",
    gumbel_vols.std() > flat_vols.std(),
)

heading("CMS spread options: a bet on the slope of the curve")

legs = [CmsLeg(0.045, 0.22, 10.0), CmsLeg(0.030, 0.28, 2.0)]
for leg in legs:
    adjustment = cms_convexity_adjustment(leg.forward, leg.vol, 5.0, leg.tenor)
    show(f"{leg.tenor:.0f}y convexity adjustment (bp)", 1e4 * adjustment)

check(
    "the longer tenor carries the larger adjustment, so they do not cancel",
    cms_convexity_adjustment(0.045, 0.22, 5.0, 10.0)
    > cms_convexity_adjustment(0.030, 0.28, 5.0, 2.0),
)

for correlation in (0.60, 0.85, 0.97):
    price = cms_spread_option(
        rc.GaussianCopula(correlation), legs, 0.015, 5.0, notional=1e6, n=200_000, random_state=0
    )
    print(f"  rho = {correlation:.2f}:  {price.price:10.2f} +/- {price.standard_error:.2f}")

loose = cms_spread_option(rc.GaussianCopula(0.60), legs, 0.015, 5.0, n=200_000, random_state=0)
tight = cms_spread_option(rc.GaussianCopula(0.97), legs, 0.015, 5.0, n=200_000, random_state=0)
check("correlation squeezes the spread and cheapens the option", tight.price < loose.price)
