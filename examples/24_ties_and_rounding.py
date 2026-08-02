"""Ties: what rounding does to a copula, and how to notice.

Rank-based copula inference assumes continuous margins, which guarantees no two
observations are ever equal. Real data is rounded -- to the cent, to the
millimetre, to the nearest whole day -- and ties appear. Nothing raises an
error; the numbers just quietly change.

This measures the damage, shows which tie-breaking rule to use for what, and
draws the line where rounding stops being a nuisance and becomes a different
problem, handled in 20_discrete_margins.py.

References
----------
Kojadinovic, I. (2017). Some copula inference procedures adapted to the
    presence of ties. *Computational Statistics and Data Analysis* 112, 24-41.
    The paper this example follows.
Genest, C., Neslehova, J. and Remillard, B. (2013). On the estimation of
    Spearman's rho and related tests of independence for possibly discontinuous
    multivariate data. *J. Multivariate Analysis* 117, 214-228.
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc

heading("Rounding moves the rank statistic the wrong way")

# Same data, coarser and coarser. Nothing else changes.
truth = rc.ClaytonCopula.from_tau(0.5)
joint = rc.CopulaDistribution(truth, [stats.norm(), stats.norm()])
exact = joint.rvs(4000, random_state=0)


def to_grid(x: np.ndarray, step: float) -> np.ndarray:
    """Round to a multiple of `step` -- a rounding rule, not a decimal count."""
    return x if step == 0.0 else np.round(x / step) * step


print(f"  {'grid':>8}{'distinct':>10}{'ties':>8}{'tau-b':>9}{'tau-c':>9}{'fitted theta':>14}")
measured = {}
for step in (0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0):
    x = to_grid(exact, step)
    distinct = int(np.mean([len(np.unique(x[:, j])) for j in range(2)]))
    tied = 1.0 - distinct / len(x)
    tau_b = float(stats.kendalltau(x[:, 0], x[:, 1]).statistic)
    tau_c = float(stats.kendalltau(x[:, 0], x[:, 1], variant="c").statistic)
    theta = float(rc.fit(rc.ClaytonCopula(1.0), rc.pseudo_obs(x), method="mpl").params[0])
    measured[step] = (tau_b, tau_c, theta)
    print(f"  {step:>8}{distinct:>10}{tied:>8.1%}{tau_b:>9.4f}{tau_c:>9.4f}{theta:>14.4f}")

show("\ntrue tau", truth.tau())
show("true theta", float(truth.params[0]))

check(
    "tau-b RISES with ties rather than falling",
    measured[1.0][0] > measured[0.0][0] + 0.05,
)
check("while tau-c eventually falls", measured[2.0][1] < measured[0.0][1] - 0.15)
print(
    "\n  This is the opposite of what 'ties destroy information' suggests, and it\n"
    "  is worth understanding rather than memorising. Kendall's tau-b divides by\n"
    "  the number of *untied* pairs. Rounding removes pairs from that denominator\n"
    "  faster than it removes concordance from the numerator, so the statistic\n"
    "  climbs -- 0.507 to 0.571 here -- while the dependence has not changed at\n"
    "  all. Reading that as 'the assets became more dependent' would be exactly\n"
    "  backwards. tau-c, which does not renormalise, falls as expected."
)

show("\nfitted theta at 4000 distinct values", measured[0.0][2])
show("fitted theta at 7 distinct values", measured[1.0][2])
check(
    "the fitted parameter is far more robust than the rank statistic",
    abs(measured[1.0][2] - 2.0) < 0.2,
)
print(
    "\n  The fit barely moves: theta stays within 10% of the truth down to seven\n"
    "  distinct values per margin. It is the summary statistic that misleads,\n"
    "  not the estimator -- which is an argument for reporting the fitted\n"
    "  parameter rather than a sample tau whenever ties are present."
)

heading("Which tie-breaking rule, and what each one costs")

# pseudo_obs must decide what rank a group of equal values receives, and the
# choice splits into two kinds with very different consequences.
coarse = to_grid(exact, 1.0)
print(f"  {'ties_method':>12}{'tau-b':>9}{'fitted theta':>14}   what it does")
descriptions = {
    "average": "shares the rank -- the default",
    "min": "the whole group takes the lowest rank",
    "max": "the whole group takes the highest",
    "dense": "shares, but without gaps afterwards",
    "ordinal": "breaks ties by position in the file",
    "random": "breaks ties by coin flip",
}
taus, thetas = {}, {}
for method, note in descriptions.items():
    u = rc.pseudo_obs(coarse, ties_method=method)
    taus[method] = float(rc.cor_kendall(u)[0, 1])
    thetas[method] = float(rc.fit(rc.ClaytonCopula(1.0), u, method="mpl").params[0])
    print(f"  {method:>12}{taus[method]:>9.4f}{thetas[method]:>14.4f}   {note}")

check(
    "the tie-preserving rules all give the same tau",
    max(abs(taus[m] - taus["average"]) for m in ("min", "max", "dense")) < 1e-12,
)
show(
    "fitted theta, smallest and largest",
    (round(min(thetas.values()), 3), round(max(thetas.values()), 3)),
)
check(
    "but wildly different fitted parameters",
    max(thetas.values()) > 5 * min(thetas.values()),
)
print(
    "\n  Two separate things are happening. `average`, `min`, `max` and `dense`\n"
    "  leave the ties tied, so every rank statistic is identical -- and yet the\n"
    "  fitted theta runs from 1.12 to 8.05, a factor of seven, because the\n"
    "  pseudo-observations themselves sit at different heights. `dense` is the\n"
    "  worst offender: closing the gaps after each tied group compresses the\n"
    "  whole sample towards the diagonal, which reads as enormous dependence.\n"
    "  A likelihood sees all of that; a rank correlation sees none of it."
)

# Reordering the data must not change a rank statistic. Two of these fail that.
rng = np.random.default_rng(0)
order = rng.permutation(len(coarse))
print(f"\n  {'ties_method':>12}{'original order':>17}{'rows shuffled':>16}")
for method in ("average", "ordinal", "random"):
    original = float(rc.cor_kendall(rc.pseudo_obs(coarse, ties_method=method))[0, 1])
    reordered = float(rc.cor_kendall(rc.pseudo_obs(coarse[order], ties_method=method))[0, 1])
    print(f"  {method:>12}{original:>17.6f}{reordered:>16.6f}")
    if method == "average":
        check("'average' is invariant to row order", abs(original - reordered) < 1e-12)
    if method == "ordinal":
        check(
            "'ordinal' is not -- it reads order that is not there", abs(original - reordered) > 1e-6
        )
print(
    "\n  `ordinal` resolves ties by row position, so sorting the file by date\n"
    "  rather than by name changes the estimate. `random` at least admits it is\n"
    "  guessing, and averaging over several draws is defensible -- that is the\n"
    "  distributional transform, in 20_discrete_margins.py. `average` is the\n"
    "  default because it is the only rule that adds nothing."
)

heading("Ties break the goodness-of-fit test, quietly")

# The bootstrap resamples from a *continuous* fitted copula, so its null
# distribution has no ties while the data does. The test sees the difference.
print(f"  {'data':>16}{'gof p-value':>14}{'verdict':>12}")
for label, x in [
    ("exact", exact),
    ("grid 0.25", to_grid(exact, 0.25)),
    ("grid 1.0", to_grid(exact, 1.0)),
]:
    u = rc.pseudo_obs(x)
    fitted = rc.fit(rc.ClaytonCopula(1.0), u, method="mpl").copula
    p = rc.gof_test(fitted, u, n_rep=200, random_state=0).pvalue
    print(f"  {label:>16}{p:>14.4f}{'ok' if p > 0.05 else 'REJECTED':>12}")

exact_p = rc.gof_test(
    rc.fit(rc.ClaytonCopula(1.0), rc.pseudo_obs(exact), method="mpl").copula,
    rc.pseudo_obs(exact),
    n_rep=200,
    random_state=0,
).pvalue
coarse_p = rc.gof_test(
    rc.fit(rc.ClaytonCopula(1.0), rc.pseudo_obs(coarse), method="mpl").copula,
    rc.pseudo_obs(coarse),
    n_rep=200,
    random_state=0,
).pvalue
show("\np-value on exact data", float(exact_p))
show("p-value on the same data, rounded to a unit grid", float(coarse_p))
check("the test survives the correct model on exact data", exact_p > 0.05)
check("and rejects it once the data is tied", coarse_p < 0.05)
print(
    "\n  If a rounded sample rejects, the honest reading is 'the data has ties',\n"
    "  not 'Clayton is wrong'. The copula is the same one that generated it. A\n"
    "  test that cannot tell those apart should not be used to choose a family."
)

heading("Where rounding stops being a nuisance")

# There is no threshold, but there is a regime change: once the number of
# distinct values is small relative to n, the model is discrete and the tools in
# 20_discrete_margins.py apply instead.
print(f"  {'distinct values per margin':>28}{'tau-b':>10}{'change':>14}")
baseline = float(rc.cor_kendall(exact)[0, 1])
for step in (0.1, 0.5, 1.0, 2.0):
    x = to_grid(exact, step)
    distinct = int(np.mean([len(np.unique(x[:, j])) for j in range(2)]))
    tau = float(rc.cor_kendall(x)[0, 1])
    print(f"  {distinct:>28}{tau:>10.4f}{tau / baseline - 1:>+13.1%}")

# The extreme case: a binary margin, where the ceiling is a property of the
# margins rather than of the dependence.
binary = (exact > 0).astype(float)
show("\ntau-b after collapsing to two values", float(rc.cor_kendall(binary)[0, 1]))
show(
    "the most two matched Bernoulli margins allow",
    rc.discrete.tau_upper_bound([stats.bernoulli(0.5), stats.bernoulli(0.5)]),
)
show(
    "...but mismatched ones, say 0.1 and 0.9, allow only",
    rc.discrete.tau_upper_bound([stats.bernoulli(0.1), stats.bernoulli(0.9)]),
)
check(
    "the ceiling is a property of the margins, not of the dependence",
    rc.discrete.tau_upper_bound([stats.bernoulli(0.1), stats.bernoulli(0.9)]) < 0.2,
)
print(
    "\n  At two values per margin nothing here applies any more: the copula is no\n"
    "  longer identified from the data, and 20_discrete_margins.py takes over\n"
    "  with the exact likelihood. The transition is gradual, which is exactly\n"
    "  why it is worth checking the number of distinct values before trusting a\n"
    "  rank-based fit."
)
