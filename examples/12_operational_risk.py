"""Operational-risk capital by the loss-distribution approach.

Each business-line/event-type cell gets its own compound frequency-severity
distribution; the cells are then combined under a copula. Basel asks for the
99.9% one-year loss, which is deep enough that the dependence assumption
dominates the answer.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from _common import check, heading, show
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

heading("One cell: a compound distribution")

frequency, severity = stats.poisson(40), stats.lognorm(1.2, scale=5_000)
losses = aggregate_loss(frequency, severity, n=200_000, random_state=0)

# The compound moments are exact, which is the check that the simulation is
# right: E[S] = E[N] E[X] and Var[S] = E[N] Var[X] + Var[N] E[X]^2.
show("simulated mean", float(losses.mean()))
show("exact E[N] E[X]", float(frequency.mean() * severity.mean()))
exact_var = frequency.mean() * severity.var() + frequency.var() * severity.mean() ** 2
show("simulated variance", float(losses.var()))
show("exact variance", float(exact_var))
exact_mean = frequency.mean() * severity.mean()
check("the mean matches to 2%", abs(losses.mean() / exact_mean - 1) < 0.02)
check("the variance matches to 6%", abs(losses.var() / exact_var - 1) < 0.06)
check("the aggregate is right-skewed", stats.skew(losses) > 0)

heading("Seven cells, combined under different dependence")

cells = [
    (stats.poisson(30), stats.lognorm(1.2, scale=4_000)),
    (stats.poisson(60), stats.lognorm(0.9, scale=2_000)),
    (stats.poisson(15), stats.lognorm(1.8, scale=9_000)),
    (stats.poisson(45), stats.lognorm(1.1, scale=3_000)),
    (stats.poisson(20), stats.lognorm(1.5, scale=6_000)),
    (stats.poisson(80), stats.lognorm(0.7, scale=1_200)),
    (stats.poisson(10), stats.lognorm(2.0, scale=12_000)),
]
d = len(cells)
TAU = 0.3

STRUCTURES = {
    "independent": None,
    "Gaussian": rc.GaussianCopula.from_tau(TAU, dim=d),
    "Student(4)": rc.StudentCopula.from_tau(TAU, dim=d, df=4.0),
    "Gumbel": rc.GumbelCopula.from_tau(TAU, dim=d),
    "comonotone": rc.FrechetUpperCopula(d),
}

print(f"  {'dependence':<14}{'99.9% VaR':>14}{'expected loss':>16}{'capital':>14}{'benefit':>11}")
capital = {}
for name, cop in STRUCTURES.items():
    result = operational_risk_capital(cells, cop, alpha=0.999, n=120_000, random_state=0)
    capital[name] = result["capital"]
    print(
        f"  {name:<14}{result['var']:>14,.0f}{result['expected_loss']:>16,.0f}"
        f"{result['capital']:>14,.0f}{result['diversification_benefit']:>11,.0f}"
    )

check(
    "independence gives the largest diversification credit",
    capital["independent"] == min(capital.values()),
)
check("comonotonicity gives none", capital["comonotone"] == max(capital.values()))
show(
    "Gumbel vs Gaussian at the same tau: extra capital (%)",
    100 * (capital["Gumbel"] / capital["Gaussian"] - 1),
)

print(
    "\n  Gumbel and the Gaussian have IDENTICAL rank correlation here. The gap is\n"
    "  entirely tail dependence, and it is the number a capital committee would\n"
    "  argue about."
)

heading("Reinsurance: what a layer is worth")

portfolio = aggregate_loss(
    stats.poisson(40), stats.lognorm(1.3, scale=8_000), n=200_000, random_state=0
)
show("mean annual loss", float(portfolio.mean()))
show("99% VaR", float(rc.risk.value_at_risk(portfolio, 0.99)))

print(f"\n  {'layer':<22}{'P(attach)':>12}{'P(exhaust)':>13}{'exp. loss':>13}{'premium':>12}")
edges = [(400_000, 200_000), (600_000, 400_000), (1_000_000, 1_000_000)]
for attachment, limit in edges:
    stat = layer_statistics(portfolio, attachment, limit)
    premium = reinsurance_premium(portfolio, attachment, limit, method="expected_shortfall")
    label = f"{limit:,.0f} xs {attachment:,.0f}"
    print(
        f"  {label:<22}{stat.attachment_probability:>12.4f}"
        f"{stat.exhaustion_probability:>13.4f}{stat.expected_loss:>13,.0f}{premium:>12,.0f}"
    )

for attachment, limit in edges:
    stat = layer_statistics(portfolio, attachment, limit)
    premium = reinsurance_premium(portfolio, attachment, limit, method="expected_shortfall")
    check(
        f"premium exceeds expected loss for the {limit:,.0f} xs {attachment:,.0f} layer",
        premium > stat.expected_loss,
    )

# Disjoint layers must partition the loss exactly -- an identity, so it is a
# sharp check that the layer arithmetic is right.
boundaries = [0.0, 200_000.0, 500_000.0, 1_500_000.0, 1e15]
recovered = sum(excess_of_loss(portfolio, a, b - a) for a, b in pairwise(boundaries))
check("disjoint layers add up to the whole loss", np.allclose(recovered, portfolio, rtol=1e-12))

heading("A catastrophe bond")

print(f"  {'trigger':<26}{'exp. loss':>12}{'P(attach)':>12}{'multiple':>11}")
for attachment, exhaustion in [
    (600_000, 1_000_000),
    (1_000_000, 1_600_000),
    (1_600_000, 2_500_000),
]:
    bond = catastrophe_bond(portfolio, attachment, exhaustion, coupon=0.08, risk_free=0.03)
    label = f"{attachment:,.0f} - {exhaustion:,.0f}"
    print(
        f"  {label:<26}{bond['expected_loss']:>12.5f}"
        f"{bond['attachment_probability']:>12.4f}{bond['multiple']:>11.2f}"
    )

near = catastrophe_bond(portfolio, 600_000, 1_000_000)
far = catastrophe_bond(portfolio, 1_600_000, 2_500_000)
check("a remoter layer has a lower expected loss", far["expected_loss"] < near["expected_loss"])
check("and a higher multiple", far["multiple"] > near["multiple"])
