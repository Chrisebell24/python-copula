"""Sklar's theorem, both directions, and the bounds every copula obeys.

Ported from *Elements of Copula Modeling with R*, chapters 2.2-2.4.

    ## R
    ## mv <- mvdc(claytonCopula(2), c("norm", "exp"),
    ##            list(list(mean = 1, sd = 2), list(rate = 3)))
    ## x  <- rMvdc(1000, mv); dMvdc(x, mv); pMvdc(x, mv)
    ## pCopula(u, lowfhCopula()) ; pCopula(u, upfhCopula())
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc

heading("Direction 1: any copula plus any margins is a joint distribution")

joint = rc.CopulaDistribution(
    rc.ClaytonCopula(2.0),
    margins=[stats.norm(loc=1, scale=2), stats.expon(scale=1 / 3)],
)
x = joint.rvs(4000, random_state=0)
show("sample shape", x.shape)

# The margins come out exactly as specified -- that is the point of the
# construction: dependence and margins are chosen independently of each other.
for j, (name, margin) in enumerate(
    [("normal(1, 2)", stats.norm(1, 2)), ("exponential(rate 3)", stats.expon(scale=1 / 3))]
):
    p = stats.kstest(x[:, j], margin.cdf).pvalue
    check(f"margin {j} is {name} (KS p = {p:.3f})", p > 0.01)

heading("Direction 2: any joint distribution decomposes into margins and a copula")

# Recovering the copula needs no knowledge of the margins -- only ranks.
u = rc.pseudo_obs(x)
recovered = rc.fit(rc.ClaytonCopula(), u, method="mpl")
show("true theta", 2.0)
show("recovered from ranks alone", float(recovered.params[0]))
check("the copula survives the margins being unknown", abs(recovered.params[0] - 2.0) < 0.2)

heading("Invariance: monotone transforms of the margins change nothing")

# Squaring an exponential and exponentiating a normal are wildly nonlinear, yet
# the dependence is identical, because it lives in the ranks.
warped = np.column_stack([np.exp(x[:, 0]), x[:, 1] ** 3])
show("Kendall's tau, original", float(rc.cor_kendall(x)[0, 1]))
show("Kendall's tau, transformed", float(rc.cor_kendall(warped)[0, 1]))
show("Pearson correlation, original", float(np.corrcoef(x.T)[0, 1]))
show("Pearson correlation, transformed", float(np.corrcoef(warped.T)[0, 1]))
check(
    "rank correlation is unchanged to machine precision",
    abs(rc.cor_kendall(x)[0, 1] - rc.cor_kendall(warped)[0, 1]) < 1e-12,
)
check(
    "Pearson correlation is not",
    abs(np.corrcoef(x.T)[0, 1] - np.corrcoef(warped.T)[0, 1]) > 0.1,
)

heading("The Frechet-Hoeffding bounds")

# Every copula lies between them, so they bracket what dependence can be.
rng = np.random.default_rng(0)
grid = rng.uniform(0.01, 0.99, size=(2000, 2))
lower = rc.FrechetLowerCopula().cdf(grid)
upper = rc.FrechetUpperCopula(2).cdf(grid)

for cop in (
    rc.ClaytonCopula(4.0),
    rc.GumbelCopula(3.0),
    rc.FrankCopula(-6.0),
    rc.GaussianCopula(0.9),
    rc.IndependenceCopula(2),
):
    values = cop.cdf(grid)
    check(
        f"W <= {cop.name} <= M",
        bool(np.all(values >= lower - 1e-12) and np.all(values <= upper + 1e-12)),
    )

show("W is countermonotone, tau", rc.FrechetLowerCopula().tau())
show("M is comonotone, tau", rc.FrechetUpperCopula(2).tau())

# In three or more dimensions W is not a copula at all: its C-volume can be
# negative. R restricts lowfhCopula the same way.
try:
    rc.FrechetLowerCopula(dim=3)
except ValueError as exc:
    check(f"W is refused in dim 3 ({exc})", True)
