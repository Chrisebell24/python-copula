"""Copulas outside finance: climate, reliability, and survival analysis.

Three settings from the literature where the copula choice changes the
conclusion, not merely the third decimal.

References
----------
Zscheischler, J. and Seneviratne, S. I. (2017). Dependence of drivers affects
    risks associated with compound events. *Science Advances* 3, e1700263.
Tang, X.-S. et al. (2013). Impact of copula selection on geotechnical
    reliability. *Computers and Geotechnics* 49, 264-278.
Clayton, D. G. (1978). A model for association in bivariate life tables.
    *Biometrika* 65(1), 141-151.
Lebrun, R. and Dutfoy, A. (2009). An innovating analysis of the Nataf
    transformation from the copula viewpoint.
    *Probabilistic Engineering Mechanics* 24(3), 312-320.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc

heading("Climate: compound hot-and-dry events")

# Hot and dry arrive together -- temperature and precipitation are negatively
# dependent -- so treating them as independent understates how often both bite.
tau = -0.35
margins = [stats.norm(loc=22.0, scale=3.0), stats.gamma(2.0, scale=25.0)]

hot_level, dry_level = 0.9, 0.1
print(f"  {'dependence':<24}{'P(hot AND dry)':>18}{'vs independence':>18}")
compound = {}
for name, cop in [
    ("independent", rc.IndependenceCopula(2)),
    ("Gaussian", rc.GaussianCopula.from_tau(tau)),
    ("Frank", rc.FrankCopula.from_tau(tau)),
    ("rotated Clayton (90)", rc.RotatedCopula(rc.ClaytonCopula.from_tau(-tau), 90)),
]:
    u = cop.rvs(400_000, random_state=0)
    # Margins are increasing transforms of the uniforms, so the event is a
    # statement about u alone -- the actual degrees and millimetres cancel.
    compound[name] = float(np.mean((u[:, 0] > hot_level) & (u[:, 1] < dry_level)))

for name, probability in compound.items():
    ratio = probability / compound["independent"]
    print(f"  {name:<24}{probability:>18.5f}{ratio:>17.2f}x")

multiplier = compound["Gaussian"] / compound["independent"]
show("compound event, independent", compound["independent"])
show("compound event, Gaussian dependence", compound["Gaussian"])
show("likelihood multiplier", multiplier)
check("negative dependence makes the compound event several times likelier", multiplier > 2.5)
show(
    "hottest / driest decile, in degrees C and mm",
    (float(margins[0].ppf(hot_level)), float(margins[1].ppf(dry_level))),
)
print(
    "\n  Zscheischler & Seneviratne report multipliers of 3.4 to 4.0 for observed\n"
    "  hot-dry dependence, which brackets the 3.4 above. A 100-year compound\n"
    "  event under an independence assumption is a 30-year event once the\n"
    "  dependence is admitted."
)

heading("Reliability: the copula moves the failure probability by an order of magnitude")

# Two correlated strength parameters, identical margins and identical Kendall's
# tau. Only the tail behaviour of the dependence differs. The system fails when
# BOTH are low at once, which is exactly what lower tail dependence governs and
# what a single correlation number cannot express.
strength = [stats.lognorm(0.3, scale=20.0), stats.lognorm(0.25, scale=15.0)]
CANDIDATES = [
    ("Gaussian", rc.GaussianCopula.from_tau(0.5)),
    ("Frank", rc.FrankCopula.from_tau(0.5)),
    ("Student(3)", rc.StudentCopula.from_tau(0.5, df=3.0)),
    ("Clayton", rc.ClaytonCopula.from_tau(0.5)),
    ("survival Gumbel", rc.survival(rc.GumbelCopula.from_tau(0.5))),
]


def failure_probability(cop: rc.Copula, load: float, draws: int = 2_000_000) -> float:
    u = cop.rvs(draws, random_state=0)
    capacity = strength[0].ppf(u[:, 0]) + strength[1].ppf(u[:, 1])
    return float(np.mean(capacity < load))


# Two design loads: a routine one, and one far enough into the tail that the
# joint behaviour of the two strengths is what decides the answer.
for load in (24.0, 17.0):
    print(f"\n  design load {load:.0f}")
    print(f"  {'copula':<24}{'tau':>8}{'lambda_L':>11}{'P(failure)':>13}{'vs least':>10}")
    probabilities = {name: failure_probability(cop, load) for name, cop in CANDIDATES}
    least = min(probabilities.values())
    for name, cop in CANDIDATES:
        p = probabilities[name]
        print(
            f"  {name:<24}{cop.tau():>8.3f}{cop.lambda_().lower:>11.4f}{p:>13.6f}{p / least:>9.1f}x"
        )
    spread = max(probabilities.values()) / least
    show("   worst / best", spread)
    if load == 17.0:
        deep_spread = spread

check("far enough into the tail, the copula choice moves it by more than 5x", deep_spread > 5)
print(
    "\n  Every row has Kendall's tau = 0.5 and identical margins. Near the mean\n"
    "  the five copulas barely differ; in the tail they differ by an order of\n"
    "  magnitude, and it is the tail that reliability analysis is about. Tang\n"
    "  et al. report the same effect in a geotechnical setting."
)

heading("Survival analysis: frailty and the Clayton copula are the same model")

# A shared gamma frailty acting multiplicatively on two hazards induces exactly
# a Clayton copula, with theta = 1/variance of the frailty. This is Clayton's
# 1978 paper, and it is an identity rather than an approximation.
rng = np.random.default_rng(0)
for frailty_variance in (0.5, 1.0, 2.0):
    shape = 1.0 / frailty_variance
    frailty = rng.gamma(shape, scale=frailty_variance, size=200_000)
    # Two lifetimes with unit baseline hazard, sharing the frailty.
    lifetimes = rng.exponential(1.0, size=(200_000, 2)) / frailty[:, None]
    observed = stats.kendalltau(lifetimes[:, 0], lifetimes[:, 1]).statistic
    implied = rc.ClaytonCopula(frailty_variance).tau()
    show(f"frailty variance {frailty_variance}: sample tau", float(observed))
    show(f"   Clayton(theta = {frailty_variance}) tau", implied)
    check("the frailty model IS a Clayton copula", abs(observed - implied) < 0.01)

heading("Reliability engineering: the Nataf transform is a Gaussian copula")

# The Nataf transformation, standard in structural reliability, is exactly a
# Gaussian copula with the margins transformed -- which means the whole method
# inherits the Gaussian copula's zero tail dependence, usually unremarked.
correlation = 0.7
nataf_margins = [stats.lognorm(0.4, scale=10.0), stats.weibull_min(2.0, scale=8.0)]
joint = rc.CopulaDistribution(rc.GaussianCopula(correlation), margins=nataf_margins)
x = joint.rvs(200_000, random_state=0)

# The Nataf construction: transform to standard normal, correlate, transform back.
z = rng.multivariate_normal([0, 0], [[1, correlation], [correlation, 1]], size=200_000)
nataf = np.column_stack([m.ppf(stats.norm.cdf(z[:, j])) for j, m in enumerate(nataf_margins)])

show("Kendall tau, copula construction", float(rc.cor_kendall(x)[0, 1]))
show("Kendall tau, Nataf construction", float(rc.cor_kendall(nataf)[0, 1]))
check(
    "the two agree -- they are the same model",
    abs(rc.cor_kendall(x)[0, 1] - rc.cor_kendall(nataf)[0, 1]) < 0.01,
)
show("and its tail dependence is exactly", rc.GaussianCopula(correlation).lambda_().upper)
print(
    "\n  So a Nataf-based reliability analysis has assumed away joint extremes,\n"
    "  whatever correlation was entered. That is a modelling choice; it is\n"
    "  rarely presented as one."
)
