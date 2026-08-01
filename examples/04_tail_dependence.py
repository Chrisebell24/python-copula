"""Tail dependence: the property that separates families at the same tau.

Two copulas calibrated to identical Kendall's tau agree almost everywhere and
disagree exactly where it matters. This script quantifies that.

    ## R
    ## lambda(claytonCopula(2)); lambda(gumbelCopula(2)); lambda(tCopula(0.7, df=4))
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show

import rcopula as rc

TAU = 0.5
FAMILIES = {
    "Gaussian": rc.GaussianCopula.from_tau(TAU),
    "Student(4)": rc.StudentCopula.from_tau(TAU, df=4.0),
    "Clayton": rc.ClaytonCopula.from_tau(TAU),
    "Gumbel": rc.GumbelCopula.from_tau(TAU),
    "Frank": rc.FrankCopula.from_tau(TAU),
    "survival Clayton": rc.survival(rc.ClaytonCopula.from_tau(TAU)),
}

heading(f"Six families, all calibrated to Kendall's tau = {TAU}")

print(f"  {'family':<18}{'tau':>9}{'lambda_L':>11}{'lambda_U':>11}")
for name, cop in FAMILIES.items():
    lam = cop.lambda_()
    print(f"  {name:<18}{cop.tau():>9.4f}{lam.lower:>11.4f}{lam.upper:>11.4f}")
    check(f"{name} has tau = {TAU}", abs(cop.tau() - TAU) < 1e-9)

heading("The probability of a joint extreme, by level")

# lambda is a limit. What a practitioner meets is the joint exceedance at a
# quantile that actually occurs, so report that instead.
print(f"  {'family':<18}" + "".join(f"{q:>11.1%}" for q in (0.05, 0.01, 0.001)))
for name, cop in FAMILIES.items():
    row = []
    for q in (0.05, 0.01, 0.001):
        joint = float(cop.cdf([[1 - q, 1 - q]])[0])
        row.append((1 - 2 * (1 - q) + joint) / q)  # P(both exceed | one exceeds)
    print(f"  {name:<18}" + "".join(f"{v:>11.4f}" for v in row))

heading("What that costs on a two-line loss portfolio")


# Identical exponential margins, identical tau -- only the copula differs.
def expected_shortfall(cop: rc.Copula) -> float:
    losses = -np.log1p(-cop.rvs(400_000, random_state=0)).sum(axis=1)
    return float(rc.risk.expected_shortfall(losses, 0.99))


results = {name: expected_shortfall(cop) for name, cop in FAMILIES.items()}
for name, value in sorted(results.items(), key=lambda kv: kv[1]):
    show(f"99% expected shortfall, {name}", value)

worst, best = max(results.values()), min(results.values())
show("spread between best and worst", worst - best)
show("as a percentage of the best", 100 * (worst / best - 1))

heading("The finding that catches people out")

check(
    "plain Clayton gives a LOWER shortfall than the Gaussian",
    results["Clayton"] < results["Gaussian"],
)
check(
    "survival Clayton gives a higher one",
    results["survival Clayton"] > results["Gaussian"],
)
print(
    "\n  Aggregate loss risk is driven by the UPPER tail. Clayton binds in the\n"
    "  lower one, so 'we used Clayton because we care about the tail' is, for\n"
    "  loss aggregation, exactly backwards -- it is the survival Clayton that\n"
    "  belongs there. Kendall's tau is identical in both cases and reports\n"
    f"  nothing about it ({FAMILIES['Clayton'].tau():.4f} either way)."
)
