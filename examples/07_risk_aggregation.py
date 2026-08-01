"""Portfolio VaR and expected shortfall under different dependence.

The copula is not a detail here: at fixed margins and fixed rank correlation it
moves the 99% expected shortfall substantially.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc

TAU = 0.5
MARGINS = [
    stats.lognorm(1.0, scale=100),
    stats.lognorm(1.2, scale=80),
    stats.lognorm(0.8, scale=120),
]

heading("Three risk types, six dependence assumptions")

FAMILIES = {
    "independent": rc.IndependenceCopula(3),
    "Gaussian": rc.GaussianCopula.from_tau(TAU, dim=3),
    "Student(4)": rc.StudentCopula.from_tau(TAU, dim=3, df=4.0),
    "Clayton": rc.ClaytonCopula.from_tau(TAU, dim=3),
    "Gumbel": rc.GumbelCopula.from_tau(TAU, dim=3),
    "comonotone": rc.FrechetUpperCopula(3),
}

print(f"  {'dependence':<14}{'99% VaR':>12}{'99% ES':>12}{'99.9% ES':>12}{'benefit %':>10}")
results = {}
for name, cop in FAMILIES.items():
    total = rc.risk.simulate_losses(cop, MARGINS, n=200_000, random_state=0)
    var = rc.risk.value_at_risk(total, 0.99)
    es = rc.risk.expected_shortfall(total, 0.99)
    deep = rc.risk.expected_shortfall(total, 0.999)
    benefit = rc.risk.diversification_benefit(cop, MARGINS, alpha=0.99, n=100_000, random_state=0)
    results[name] = es
    print(f"  {name:<14}{var:>12.1f}{es:>12.1f}{deep:>12.1f}{benefit['benefit_pct']:>10.2f}")

check("independence is the most diversified", results["independent"] == min(results.values()))
check("comonotonicity is the least", results["comonotone"] == max(results.values()))
show(
    "Gaussian vs Student at the same tau: 99% ES gap (%)",
    100 * (results["Student(4)"] / results["Gaussian"] - 1),
)
show(
    "Clayton vs Gumbel at the same tau: 99% ES gap (%)",
    100 * (results["Gumbel"] / results["Clayton"] - 1),
)

heading("Where the capital is coming from")

cop = rc.StudentCopula.from_tau(TAU, dim=3, df=4.0)
contributions = rc.risk.risk_contributions(cop, MARGINS, alpha=0.99, n=200_000, random_state=0)
total_es = rc.risk.expected_shortfall(
    rc.risk.simulate_losses(cop, MARGINS, n=200_000, random_state=0), 0.99
)
for j, contribution in enumerate(contributions):
    show(f"risk type {j + 1} contributes", float(contribution))
check(
    "the contributions add up to the total (Euler allocation)",
    abs(contributions.sum() - total_es) < 1e-6 * total_es,
)

heading("Systemic-risk measures")

# Simulate the components, then the system, and measure how each firm relates
# to the whole under stress.
rng = np.random.default_rng(0)
u = cop.rvs(200_000, random_state=rng)
components = np.column_stack([m.ppf(u[:, j]) for j, m in enumerate(MARGINS)])
system = components.sum(axis=1)

print(f"  {'firm':<8}{'CoVaR':>12}{'dCoVaR':>12}{'MES':>12}")
for j in range(3):
    conditional = rc.risk.covar(system, components[:, j], alpha=0.95, beta=0.95)
    delta = rc.risk.delta_covar(system, components[:, j], alpha=0.95, beta=0.95)
    mes = rc.risk.marginal_expected_shortfall(components[:, j], system, alpha=0.95)
    print(f"  {j + 1:<8}{conditional:>12.1f}{delta:>12.1f}{mes:>12.1f}")

check(
    "distress raises the system's VaR",
    rc.risk.delta_covar(system, components[:, 0], alpha=0.95, beta=0.95) > 0,
)

heading("Stress testing by conditioning, not by shocking")

# A stress scenario that respects the fitted dependence: hold one factor at its
# 99th percentile and read off what the others do, rather than moving them all
# by hand.
stressed = rc.risk.stress_scenario(cop, MARGINS, {0: 0.99}, n=200_000, random_state=0)
baseline = components.mean(axis=0)
for j in range(3):
    print(
        f"  risk type {j + 1}: baseline mean {baseline[j]:9.1f}"
        f"   given type 1 stressed {stressed[:, j].mean():9.1f}"
    )
check("the stressed factor is elevated", stressed[:, 0].mean() > baseline[0])
check("and the others follow it up", stressed[:, 1].mean() > baseline[1])
