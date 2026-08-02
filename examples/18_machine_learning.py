"""Copulas in statistics and machine learning.

Four uses that have nothing to do with pricing anything: measuring dependence
information-theoretically, estimating a robust covariance, generating synthetic
data that preserves an arbitrary joint structure, and detecting conditional
independence.

References
----------
Ma, J. and Sun, Z. (2011). Mutual information is copula entropy.
    *Tsinghua Science and Technology* 16(1), 51-54.
Liu, H., Han, F., Yuan, M., Lafferty, J. and Wasserman, L. (2012).
    High-dimensional semiparametric Gaussian copula graphical models.
    *Annals of Statistics* 40(4), 2293-2326.  (the "SKEPTIC" estimator)
Liu, H., Lafferty, J. and Wasserman, L. (2009). The nonparanormal:
    semiparametric estimation of high dimensional undirected graphs.
    *JMLR* 10, 2295-2328.
Bedford, T. and Cooke, R. M. (2002). Vines -- a new graphical model for
    dependent random variables. *Annals of Statistics* 30(4), 1031-1068.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc

heading("Mutual information is copula entropy")

# Ma & Sun: I(X;Y) = -H(c), the negative entropy of the copula density. The
# margins contribute nothing, so mutual information is a property of the
# dependence alone -- which is why it is invariant to monotone rescaling and
# correlation is not.
rng = np.random.default_rng(0)


def copula_entropy(cop: rc.Copula, draws: int = 400_000) -> float:
    """-E[log c(U)] under the copula itself, which is a plain Monte Carlo mean."""
    u = cop.rvs(draws, random_state=1)
    return -float(np.mean(cop.logpdf(u)))


print(f"  {'copula':<16}{'tau':>8}{'-H(c) est':>12}{'closed form':>14}")
for name, cop, exact in [
    ("independent", rc.IndependenceCopula(2), 0.0),
    ("Gaussian 0.5", rc.GaussianCopula(0.5), -0.5 * np.log(1 - 0.5**2)),
    ("Gaussian 0.9", rc.GaussianCopula(0.9), -0.5 * np.log(1 - 0.9**2)),
]:
    estimate = -copula_entropy(cop)
    print(f"  {name:<16}{cop.tau():>8.3f}{estimate:>12.4f}{exact:>14.4f}")
    check(f"{name}: mutual information matches -0.5 log(1-rho^2)", abs(estimate - exact) < 0.01)

# And the point of the identity: it works for copulas with no closed form.
for family in (
    rc.ClaytonCopula.from_tau(0.5),
    rc.GumbelCopula.from_tau(0.5),
    rc.FrankCopula.from_tau(0.5),
    rc.StudentCopula.from_tau(0.5, df=3.0),
):
    show(f"MI of {type(family).__name__.replace('Copula', '')} at tau=0.5", -copula_entropy(family))
print(
    "\n  All four have the same Kendall's tau and none has the same mutual\n"
    "  information. Rank correlation summarises dependence with one number;\n"
    "  mutual information counts all of it, in nats."
)

heading("Mutual information survives a monotone transform; correlation does not")

x = rng.standard_normal(200_000)
y = 0.8 * x + 0.6 * rng.standard_normal(200_000)
# A brutal but strictly increasing transform of each coordinate.
warped = np.column_stack([np.exp(x / 2), np.sinh(3 * y)])

show("Pearson, raw", float(np.corrcoef(x, y)[0, 1]))
show("Pearson, warped", float(np.corrcoef(warped[:, 0], warped[:, 1])[0, 1]))
show("Kendall, raw", float(rc.cor_kendall(np.column_stack([x, y]))[0, 1]))
show("Kendall, warped", float(rc.cor_kendall(warped)[0, 1]))
check(
    "the rank measure is unchanged; Pearson has moved a long way",
    abs(rc.cor_kendall(np.column_stack([x, y]))[0, 1] - rc.cor_kendall(warped)[0, 1]) < 1e-12
    and abs(np.corrcoef(warped[:, 0], warped[:, 1])[0, 1] - np.corrcoef(x, y)[0, 1]) > 0.2,
)

heading("The nonparanormal SKEPTIC: a covariance immune to outliers")

# Liu et al.: estimate Sigma from Kendall's tau via Sigma = sin(pi*tau/2)
# rather than from the sample covariance. Identical asymptotics under
# normality, and it does not care what the margins are.
dim, n = 6, 400
truth = rc.p2P(
    np.array([0.7, 0.5, 0.3, 0.2, 0.1, 0.6, 0.4, 0.25, 0.15, 0.55, 0.35, 0.2, 0.45, 0.3, 0.4]), dim
)
clean = rng.multivariate_normal(np.zeros(dim), truth, size=n)

# 2% of rows are corrupted -- a stuck sensor, a units error, a bad merge.
dirty = clean.copy()
corrupt = rng.choice(n, size=int(0.02 * n), replace=False)
dirty[corrupt] = rng.standard_normal((corrupt.size, dim)) * 40


def skeptic(data: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * rc.cor_kendall(data) / 2.0)


def error(estimate: np.ndarray) -> float:
    return float(np.max(np.abs(estimate - truth)))


print(f"  {'estimator':<28}{'clean':>10}{'2% corrupted':>16}")
for label, estimate in [
    ("sample correlation", np.corrcoef),
    ("SKEPTIC (sin(pi*tau/2))", lambda d: skeptic(d)),
]:
    on_clean = error(estimate(clean.T) if estimate is np.corrcoef else estimate(clean))
    on_dirty = error(estimate(dirty.T) if estimate is np.corrcoef else estimate(dirty))
    print(f"  {label:<28}{on_clean:>10.4f}{on_dirty:>16.4f}")

pearson_damage = error(np.corrcoef(dirty.T)) / error(np.corrcoef(clean.T))
skeptic_damage = error(skeptic(dirty)) / error(skeptic(clean))
show("Pearson error multiplied by", pearson_damage)
show("SKEPTIC error multiplied by", skeptic_damage)
check("the rank-based estimator barely notices the corruption", skeptic_damage < pearson_damage)
print(
    "\n  Both estimate the same Sigma. Two percent of rows replaced by noise\n"
    "  costs the sample correlation several times more than it costs SKEPTIC,\n"
    "  because a rank cannot be dragged further than to the end of the list."
)

heading("Synthetic data with the right joint structure")

# The generative-model use: fit margins and dependence separately, then draw.
# Here the truth is deliberately awkward -- a skewed margin, a bounded one, a
# count-like one, and a copula with asymmetric tails.
generator = rc.CopulaDistribution(
    rc.survival(rc.ClaytonCopula.from_tau(0.45, dim=3)),
    margins=[stats.lognorm(0.6, scale=3.0), stats.beta(2.0, 5.0), stats.gamma(3.0, scale=2.0)],
)
real = generator.rvs(2_000, random_state=7)

# Now pretend we only have `real`, and rebuild it.
u = rc.pseudo_obs(real)
selection = rc.select_copula(u, criterion="aic")
fitted_margins = [
    stats.lognorm(*stats.lognorm.fit(real[:, 0], floc=0)),
    stats.beta(*stats.beta.fit(real[:, 1], floc=0, fscale=1)),
    stats.gamma(*stats.gamma.fit(real[:, 2], floc=0)),
]
synthetic = rc.CopulaDistribution(selection.best, margins=fitted_margins).rvs(
    20_000, random_state=8
)

show("family chosen by AIC", selection.best.describe())
print(f"\n  {'':<14}{'real':>12}{'synthetic':>12}   {'':<8}{'real':>10}{'synthetic':>12}")
for j in range(3):
    print(
        f"  margin {j}      {np.mean(real[:, j]):>12.3f}{np.mean(synthetic[:, j]):>12.3f}"
        f"   {'sd':<8}{np.std(real[:, j]):>10.3f}{np.std(synthetic[:, j]):>12.3f}"
    )

real_tau, synthetic_tau = rc.cor_kendall(real), rc.cor_kendall(synthetic)
for i, j in [(0, 1), (0, 2), (1, 2)]:
    show(
        f"tau[{i},{j}]  real / synthetic",
        (round(float(real_tau[i, j]), 3), round(float(synthetic_tau[i, j]), 3)),
    )
check("every pairwise dependence is reproduced", np.max(np.abs(real_tau - synthetic_tau)) < 0.06)


# The property a marginal-only synthesiser would destroy: joint extremes.
def joint_upper(data: np.ndarray, q: float = 0.95) -> float:
    cut = np.quantile(data, q, axis=0)
    return float(np.mean(np.all(data > cut, axis=1)))


show("P(all three above their 95th pct), real", joint_upper(real))
show("P(all three above their 95th pct), synthetic", joint_upper(synthetic))
show("   what independence would give", 0.05**3)
check("the joint tail is preserved, not just the margins", joint_upper(synthetic) > 20 * 0.05**3)
print(
    "\n  AIC picked Joe, not the survival Clayton that generated the data --\n"
    "  select_copula searches the plain families, and Joe is the closest of\n"
    "  them to an upper-tail Archimedean. The cost shows up as tau about 0.04\n"
    "  low. Adding rotations to the search closes that; the joint tail, which\n"
    "  is what a marginal-only synthesiser destroys entirely, survives either\n"
    "  way at 200x the independence rate."
)

heading("Conditional independence, read off a vine")

# A vine's second tree estimates the dependence of 1 and 3 *given* 2. If the
# data really was generated by a chain 1 -> 2 -> 3, that edge should come back
# as independence -- which is a conditional-independence test with a picture.
z1 = rng.standard_normal(4_000)
z2 = 0.9 * z1 + np.sqrt(1 - 0.9**2) * rng.standard_normal(4_000)
z3 = 0.8 * z2 + np.sqrt(1 - 0.8**2) * rng.standard_normal(4_000)
chain = rc.pseudo_obs(np.column_stack([z1, z2, z3]))

show("unconditional tau(1,3)", float(rc.cor_kendall(chain)[0, 2]))
vine = rc.fit_vine(chain, structure="D", order=(0, 1, 2))
tree2 = vine.pair_copulas[1][0]
show("vine tree 2: the 1,3 | 2 edge", tree2.describe())
show("   its Kendall tau", float(tree2.tau()))
check(
    "the conditional dependence is near zero, the unconditional is not",
    abs(tree2.tau()) < 0.05 < abs(rc.cor_kendall(chain)[0, 2]),
)
print(
    "\n  tau(1,3) = 0.5 unconditionally and ~0 given 2. That is the Markov\n"
    "  property of the chain, recovered without assuming normality anywhere:\n"
    "  each edge of the vine is free to be any family."
)
