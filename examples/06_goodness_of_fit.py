"""Testing a family against the data, and choosing among families.

## R
## gofCopula(claytonCopula(), x, simulation = "pb")
## gofCopula(claytonCopula(), x, simulation = "mult")   # much faster
## xvCopula(claytonCopula(), x, k = 10)
"""

from __future__ import annotations

from _common import check, heading, show

import rcopula as rc

TRUTH = rc.ClaytonCopula(3.0)
data = TRUTH.rvs(500, random_state=0)

heading("Does the right family pass, and the wrong one fail?")

for name, candidate in [
    ("Clayton (correct)", rc.ClaytonCopula()),
    ("Gumbel (wrong tail)", rc.GumbelCopula()),
    ("Gaussian (no tail)", rc.GaussianCopula()),
]:
    result = rc.gof_test(candidate, data, simulation="mult", n_rep=500, random_state=0)
    verdict = "not rejected" if result.pvalue > 0.05 else "REJECTED"
    print(f"  {name:<22} Sn = {result.statistic:8.4f}   p = {result.pvalue:.4f}   {verdict}")

check(
    "the true family is not rejected",
    rc.gof_test(rc.ClaytonCopula(), data, simulation="mult", n_rep=500, random_state=0).pvalue
    > 0.05,
)
check(
    "the wrong tail is rejected",
    rc.gof_test(rc.GumbelCopula(), data, simulation="mult", n_rep=500, random_state=0).pvalue
    < 0.05,
)

heading("The multiplier bootstrap is the reason this is affordable")

# The parametric bootstrap refits the copula on every replicate; the multiplier
# bootstrap reweights instead. No other Python package implements the latter.
import time  # noqa: E402

for simulation in ("pb", "mult"):
    start = time.perf_counter()
    rc.gof_test(rc.ClaytonCopula(), data, simulation=simulation, n_rep=200, random_state=0)
    show(f"{simulation} with 200 replicates (seconds)", time.perf_counter() - start)

heading("Choosing among families, in one call")

ranking = rc.select_copula(data, families="all", criterion="aic")
print(ranking.summary())
show("winner", ranking.best_name)
check("the generating family wins", ranking.best_name == "clayton")
check("and is recovered", abs(ranking.best.theta - 3.0) < 0.5)

heading("Cross-validation, when AIC is not to be trusted")

# AIC is biased for copulas fitted to pseudo-observations (Gronneberg & Hjort
# 2014). Cross-validation scores on data the fit never saw.
for name, candidate in [
    ("Clayton", rc.ClaytonCopula()),
    ("Gumbel", rc.GumbelCopula()),
    ("Frank", rc.FrankCopula()),
    ("Gaussian", rc.GaussianCopula()),
]:
    score = rc.cross_validate(candidate, data, k=10, random_state=0)
    print(f"  {name:<10} cross-validated log-likelihood {score:9.3f}")

best = max(
    ["Clayton", "Gumbel", "Frank", "Gaussian"],
    key=lambda n: rc.cross_validate(
        {
            "Clayton": rc.ClaytonCopula(),
            "Gumbel": rc.GumbelCopula(),
            "Frank": rc.FrankCopula(),
            "Gaussian": rc.GaussianCopula(),
        }[n],
        data,
        k=10,
        random_state=0,
    ),
)
check("cross-validation agrees with AIC here", best == "Clayton")

heading("Testing structural assumptions rather than a family")

symmetric = rc.ClaytonCopula(3.0).rvs(400, random_state=1)
for label, test in [
    ("exchangeable?", rc.exch_test),
    ("radially symmetric?", rc.rad_sym_test),
    ("independent?", rc.indep_test),
    ("extreme-value?", rc.ev_test),
]:
    result = test(symmetric, n_rep=300, random_state=0)
    print(f"  {label:<22} p = {result.pvalue:.4f}")

check(
    "Clayton is exchangeable, so that test does not reject",
    rc.exch_test(symmetric, n_rep=300, random_state=0).pvalue > 0.05,
)
check(
    "Clayton is not radially symmetric, so that one does",
    rc.rad_sym_test(
        rc.ClaytonCopula(6.0).rvs(800, random_state=2), n_rep=300, random_state=0
    ).pvalue
    < 0.05,
)
