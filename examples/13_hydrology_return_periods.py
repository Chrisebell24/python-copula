"""Kendall return periods, and why the univariate one is the wrong number.

A flood is not one number. Peak discharge and volume are both extreme, both
matter, and the design question -- how often is an event this severe -- has no
answer from either margin alone. The Kendall distribution function supplies one.

Follows Salvadori & De Michele (2004), *Water Resources Research* 40, W12511.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc

heading("A flood series: peak and volume")

# Gumbel margins for the peak, gamma for the volume -- the usual choices -- and
# an upper-tail-dependent copula, because large peaks and large volumes arrive
# together.
truth = rc.GumbelCopula(2.2)
model = rc.CopulaDistribution(
    truth, margins=[stats.gumbel_r(loc=800, scale=250), stats.gamma(3.0, scale=45)]
)
flood = model.rvs(600, random_state=3)
show("years of record", flood.shape[0])
show("mean peak discharge (m3/s)", float(flood[:, 0].mean()))
show("mean volume (Mm3)", float(flood[:, 1].mean()))
show("Kendall's tau", float(rc.cor_kendall(flood)[0, 1]))

heading("Fit the dependence without assuming the margins")

ranking = rc.select_copula(
    flood, families=["gumbel", "galambos", "husler_reiss", "clayton", "frank", "gaussian"]
)
print(ranking.summary())
check(
    "an upper-tail-dependent family is chosen",
    ranking.table.loc[ranking.best_name, "lambda_upper"] > 0.3,
)

# Gumbel and Galambos are near-indistinguishable at this sample size, and which
# of them tops the table varies with the sample. What does not vary is that the
# extreme-value families beat the tail-independent ones by a wide margin -- that
# is the finding, and it is the one that changes the answer later.
show(
    "AIC gap, best to Gaussian",
    float(ranking.table.loc["gaussian", "aic"] - ranking.table["aic"].iloc[0]),
)
check(
    "an extreme-value family tops the table",
    ranking.best_name in {"gumbel", "galambos", "husler_reiss"},
)
check(
    "and beats the tail-independent families by more than 20 AIC",
    ranking.table.loc["gaussian", "aic"] - ranking.table["aic"].iloc[0] > 20.0,
)

fitted = ranking.best
gof = rc.gof_test(type(fitted)(), flood, simulation="mult", n_rep=400, random_state=0)
show("goodness-of-fit p-value", float(gof.pvalue))
check("the chosen family is not rejected", gof.pvalue > 0.05)

print(
    "\n  A caution worth stating: this test rejects the TRUE family about 5% of\n"
    "  the time, by construction. A single p-value below 0.05 is not proof the\n"
    "  family is wrong, and one above it is not proof the family is right."
)

heading("Univariate return periods say one thing")

for years in (10, 50, 100):
    peak = stats.gumbel_r(loc=800, scale=250).ppf(1 - 1 / years)
    volume = stats.gamma(3.0, scale=45).ppf(1 - 1 / years)
    print(f"  {years:>4}-year: peak {peak:8.1f} m3/s     volume {volume:7.1f} Mm3")

heading("The Kendall return period says another")

# The critical layer {C(u) = t} is a curve, and every point on it is equally
# extreme. K measures the region beyond it.
print(f"  {'critical level t':<20}{'K(t)':>12}{'return period':>16}")
for t in (0.9, 0.99, 0.999):
    k = float(rc.kendall_cdf(fitted, t)[0])
    period = float(rc.kendall_return_period(fitted, t)[0])
    print(f"  {t:<20.3f}{k:>12.6f}{period:>16.1f}")

check(
    "the Kendall return period always exceeds the univariate one",
    rc.kendall_return_period(fitted, 0.99)[0] > 100.0,
)

heading("What a design standard actually asks for")

# "The 100-year event" is a return period; an engineer needs the level.
for years in (10, 50, 100, 500):
    level = float(rc.return_period_level(fitted, years)[0])
    # Points on that critical layer, i.e. equally extreme design pairs. Only
    # u1 > t can lie on it: C(u1, u2) <= min(u1, u2), so a coordinate below the
    # level cannot reach it however extreme the other one is.
    grid = np.linspace(level + (1 - level) * 0.05, 0.999, 5)
    pairs = []
    for u1 in grid:
        lo, hi = 1e-9, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if fitted.cdf([[u1, mid]])[0] < level:
                lo = mid
            else:
                hi = mid
        u2 = 0.5 * (lo + hi)
        if 0.0 < u2 < 1.0:
            pairs.append(
                (stats.gumbel_r(loc=800, scale=250).ppf(u1), stats.gamma(3.0, scale=45).ppf(u2))
            )
    show(f"{years}-year critical level t", level)
    if pairs:
        print(
            "      equally extreme design pairs (peak, volume): "
            + ", ".join(f"({p:.0f}, {v:.0f})" for p, v in pairs[:3])
        )

check(
    "the level and the period invert each other",
    abs(rc.kendall_return_period(fitted, rc.return_period_level(fitted, 100.0))[0] - 100.0) < 1e-6,
)

heading("The number that should decide the design")

# Two families with identical Kendall's tau, fitted to the same data, give
# return periods that differ by an order of magnitude.
tau = fitted.tau()
print(f"  {'family':<14}{'tau':>9}{'lambda_U':>11}{'T at t = 0.99':>16}")
for name, cop in [
    ("Gumbel", rc.GumbelCopula.from_tau(tau)),
    ("Galambos", rc.GalambosCopula.from_tau(tau)),
    ("Gaussian", rc.GaussianCopula.from_tau(tau)),
    ("Clayton", rc.ClaytonCopula.from_tau(tau)),
    ("Frank", rc.FrankCopula.from_tau(tau)),
]:
    period = float(rc.kendall_return_period(cop, 0.99)[0])
    print(f"  {name:<14}{cop.tau():>9.4f}{cop.lambda_().upper:>11.4f}{period:>16.1f}")

gumbel = float(rc.kendall_return_period(rc.GumbelCopula.from_tau(tau), 0.99)[0])
clayton = float(rc.kendall_return_period(rc.ClaytonCopula.from_tau(tau), 0.99)[0])
check("the spread across families exceeds a factor of ten", clayton / gumbel > 10)

print(
    "\n  Same rank correlation, same data, same margins. The design life differs\n"
    f"  by a factor of {clayton / gumbel:.0f}. Choosing the family is not a modelling detail --\n"
    "  it is the answer."
)

heading("The nonparametric check")

# Kn needs no fitted family, so it says whether the chosen one is plausible.
grid = np.array([0.2, 0.5, 0.8, 0.95])
empirical = rc.kendall_empirical(flood, grid)
modelled = rc.kendall_cdf(fitted, grid)
for t, e, m in zip(grid, empirical, modelled, strict=True):
    print(f"  t = {t:.2f}:  empirical K {e:.4f}   fitted K {m:.4f}")
check("the fitted K tracks the empirical one", np.max(np.abs(empirical - modelled)) < 0.06)
