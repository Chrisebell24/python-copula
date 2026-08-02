"""Counts, categories, and the copula that is only half identified.

Copulas are usually taught with continuous margins, where Sklar's theorem gives
a unique copula and rank-based inference is exact. Neither survives contact with
count data -- and count data is what genomics, ecology, insurance claim
frequencies and crash-severity models are made of.

This example is as much about what stops working as about what replaces it.

References
----------
Genest, C. and Neslehova, J. (2007). A primer on copulas for count data.
    *ASTIN Bulletin* 37(2), 475-515.
Denuit, M. and Lambert, P. (2005). Constraints on concordance measures in
    bivariate discrete data. *J. Multivariate Analysis* 93(1), 40-57.
Ruschendorf, L. (2009). On the distributional transform, Sklar's theorem, and
    the empirical copula process. *JSPI* 139(11), 3921-3927.
Sun, T., Song, X. and Zhang, X. (2021). scDesign2. *Genome Biology* 22, 163.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc
from rcopula.discrete import (
    discrete_pmf,
    distributional_transform,
    fit_discrete,
    mixed_pdf,
    tau_upper_bound,
)

heading("The likelihood becomes a difference, not a derivative")

# With atoms there is no density to differentiate. The probability of a cell is
# the copula's volume over the rectangle the atom spans -- an inclusion-exclusion
# sum, exact and cheap in two dimensions.
margins = [stats.poisson(2.0), stats.poisson(3.0)]
copula = rc.ClaytonCopula(2.0)

lattice = np.array([[i, j] for i in range(45) for j in range(45)], dtype=float)
mass = discrete_pmf(copula, lattice, margins).reshape(45, 45)

show("total mass over the lattice", float(mass.sum()))
check("it is a probability distribution", abs(mass.sum() - 1.0) < 1e-9)
show(
    "max error in the first margin",
    float(np.max(np.abs(mass.sum(axis=1) - margins[0].pmf(np.arange(45))))),
)
show(
    "max error in the second margin",
    float(np.max(np.abs(mass.sum(axis=0) - margins[1].pmf(np.arange(45))))),
)
check(
    "both margins come back exactly",
    np.allclose(mass.sum(axis=1), margins[0].pmf(np.arange(45)), atol=1e-9),
)

print(f"\n  {'cell':>8}{'independent':>14}{'Clayton(2)':>13}{'ratio':>9}")
for cell in [(0, 0), (2, 3), (5, 6), (8, 10)]:
    point = np.array([cell], dtype=float)
    free = float(discrete_pmf(rc.IndependenceCopula(2), point, margins)[0])
    tied = float(discrete_pmf(copula, point, margins)[0])
    print(f"  {cell!s:>8}{free:>14.6f}{tied:>13.6f}{tied / free:>9.2f}x")

heading("Ranks stop meaning what they meant")

# Ties compress concordance, and the ceiling depends on the margins. Inverting a
# sample tau -- exact for continuous margins -- is simply wrong here.
print(f"  {'margins':<38}{'max attainable tau':>20}")
for label, pair in [
    ("Poisson(2), Poisson(2)", [stats.poisson(2.0)] * 2),
    ("Poisson(3), NegBin(4, 0.5)", [stats.poisson(3.0), stats.nbinom(4, 0.5)]),
    ("Bernoulli(0.3), Bernoulli(0.5)", [stats.bernoulli(0.3), stats.bernoulli(0.5)]),
    ("Bernoulli(0.1), Bernoulli(0.9)", [stats.bernoulli(0.1), stats.bernoulli(0.9)]),
]:
    print(f"  {label:<38}{tau_upper_bound(pair):>20.4f}")

skewed = [stats.bernoulli(0.1), stats.bernoulli(0.9)]
ceiling = tau_upper_bound(skewed)
x = rc.CopulaDistribution(rc.GaussianCopula(0.95), skewed).rvs(40_000, random_state=0)
observed = float(stats.kendalltau(x[:, 0], x[:, 1]).statistic)
show("\nrho = 0.95, an almost comonotone copula", 0.95)
show("   the tau it actually produces", observed)
show("   the ceiling these margins impose", ceiling)
check("no copula can beat the ceiling", observed <= ceiling + 0.01)
print(
    "\n  Read that 0.1 as weak dependence and you have it exactly backwards: it\n"
    "  is as strong as these margins permit. Identical margins are the benign\n"
    "  case -- tau-b still reaches 1 there -- and mismatched ones are not."
)

heading("Fitting: likelihood, because the rank shortcut is unavailable")

truth = rc.GaussianCopula(0.6)
count_margins = [stats.poisson(4.0), stats.poisson(4.0)]
counts = rc.CopulaDistribution(truth, count_margins).rvs(3000, random_state=0)

result = fit_discrete(counts, rc.GaussianCopula(0.0), count_margins)
print()
print(result.summary())

show("\ntrue rho", 0.6)
show("estimated rho", float(result.params[0]))
check("recovered", abs(result.params[0] - 0.6) < 0.06)

# What the naive shortcut would have given. One sample cannot separate bias
# from noise, so repeat it.
replicates = 15
likelihood_estimates, rank_estimates = [], []
for seed in range(replicates):
    draw = rc.CopulaDistribution(truth, count_margins).rvs(1500, random_state=seed)
    fitted = fit_discrete(draw, rc.GaussianCopula(0.0), count_margins)
    likelihood_estimates.append(float(fitted.params[0]))
    rank_estimates.append(
        float(rc.GaussianCopula.from_tau(float(rc.cor_kendall(draw)[0, 1])).params[0])
    )

print(f"\n  {replicates} replicates at n = 1500, true rho = 0.6")
print(f"  {'estimator':<22}{'mean':>9}{'bias':>10}{'sd':>9}{'RMSE':>9}")
summaries = {}
for label, estimates in [
    ("exact likelihood", np.array(likelihood_estimates)),
    ("invert sample tau", np.array(rank_estimates)),
]:
    bias = float(estimates.mean() - 0.6)
    rmse = float(np.sqrt(np.mean((estimates - 0.6) ** 2)))
    summaries[label] = (bias, float(estimates.std()), rmse)
    print(f"  {label:<22}{estimates.mean():>9.4f}{bias:>+10.4f}{estimates.std():>9.4f}{rmse:>9.4f}")

check(
    "inverting the sample tau is biased, not merely noisy",
    abs(summaries["invert sample tau"][0]) > 2 * summaries["invert sample tau"][1],
)
check(
    "and the exact likelihood is not",
    abs(summaries["exact likelihood"][0]) < summaries["exact likelihood"][1],
)
check(
    "which costs the shortcut a factor of three in RMSE",
    summaries["invert sample tau"][2] > 3 * summaries["exact likelihood"][2],
)
print(
    "\n  The bias does not shrink with n: the sample tau of tied data does not\n"
    "  estimate the copula's tau, so inverting it converges to the wrong number.\n"
    "  This is why fit_discrete offers likelihood and nothing else."
)

heading("Mixed margins: a continuous response beside a discrete one")

# The transportation and biostatistics case -- joining a discrete choice to a
# continuous outcome. Differentiate along one coordinate, difference along the
# other.
mixed_margins = [stats.norm(loc=30.0, scale=8.0), stats.poisson(1.2)]
# Five standard deviations either side: the truncation error is then 6e-7, well
# below the quadrature error, so a shortfall in the total means a real bug.
grid = np.linspace(-10.0, 70.0, 4001)

total = 0.0
print(f"  {'count':>7}{'P(count)':>12}{'from the mixed density':>26}")
for count in range(8):
    rows = np.column_stack([grid, np.full_like(grid, float(count))])
    recovered = float(np.trapezoid(mixed_pdf(copula, rows, mixed_margins, [False, True]), grid))
    total += recovered
    if count < 5:
        print(f"  {count:>7}{mixed_margins[1].pmf(count):>12.6f}{recovered:>26.6f}")
for count in range(8, 40):
    rows = np.column_stack([grid, np.full_like(grid, float(count))])
    total += float(np.trapezoid(mixed_pdf(copula, rows, mixed_margins, [False, True]), grid))

show("\nthe mixed density integrates and sums to", total)
check("it is a probability distribution", abs(total - 1.0) < 1e-5)

# And the dependence is real: the continuous variable's mean shifts with the count.
conditional = []
for count in range(4):
    rows = np.column_stack([grid, np.full_like(grid, float(count))])
    density = mixed_pdf(copula, rows, mixed_margins, [False, True])
    conditional.append(float(np.trapezoid(grid * density, grid) / np.trapezoid(density, grid)))
print(f"\n  {'count':>7}{'E[continuous | count]':>26}")
for count, mean in enumerate(conditional):
    print(f"  {count:>7}{mean:>26.4f}")
check("the conditional mean increases with the count", conditional == sorted(conditional))

heading("The distributional transform, and what it costs")

# Randomising within each atom gives exactly uniform pseudo-observations, which
# any continuous-margin method can then consume. The randomisation becomes part
# of the answer.
poisson = stats.poisson(3.0)
sample = poisson.rvs(50_000, random_state=0).reshape(-1, 1).astype(float)
u = distributional_transform(sample, [poisson], random_state=0)

show("mean (exactly 1/2 for a uniform)", float(u.mean()))
show("variance (exactly 1/12)", float(u.var()))
show("   1/12 is", 1 / 12)
show("Kolmogorov-Smirnov p-value against uniform", float(stats.kstest(u.ravel(), "uniform").pvalue))
check("it is exactly uniform, not approximately", stats.kstest(u.ravel(), "uniform").pvalue > 0.01)

# The cost: run it twice and get two different answers.
estimates = [
    float(rc.cor_kendall(distributional_transform(counts, count_margins, random_state=seed))[0, 1])
    for seed in range(12)
]
show(
    "\ntau from the transform, across 12 seeds",
    (round(min(estimates), 4), round(max(estimates), 4)),
)
show("   spread", max(estimates) - min(estimates))
check("the randomisation is visible in the answer", max(estimates) - min(estimates) > 1e-4)
print(
    "\n  That spread is not sampling error -- the data never changed. It is the\n"
    "  identifiability problem made visible: each draw picks a different member\n"
    "  of the class of copulas compatible with these counts, and they genuinely\n"
    "  differ. The likelihood approach avoids the randomisation but does not\n"
    "  avoid the underlying non-identifiability; it only stops pretending."
)
