"""Build a copula, evaluate it, sample it.

Ported from *Elements of Copula Modeling with R*, chapter 2.1.

    ## R
    ## library(copula)
    ## cop <- claytonCopula(2, dim = 3)
    ## u   <- rCopula(1000, cop)
    ## dCopula(u[1:3, ], cop); pCopula(u[1:3, ], cop)
    ## tau(cop); rho(cop); lambda(cop)
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc

heading("A Clayton copula in three dimensions")

cop = rc.ClaytonCopula(theta=2.0, dim=3)
print(f"  {cop!r}")
show("Kendall's tau", cop.tau())
show("Spearman's rho", cop.rho())
show("lower tail dependence", cop.lambda_().lower)
show("upper tail dependence", cop.lambda_().upper)

u = cop.rvs(2000, random_state=0)
show("sample shape", u.shape)
show("density at the first point", float(cop.pdf(u[:1])[0]))
show("CDF at the first point", float(cop.cdf(u[:1])[0]))

heading("What makes it a copula")

# Uniform margins: this is the definition, not a coincidence of the sampler.
for j in range(3):
    p = stats.kstest(u[:, j], "uniform").pvalue
    check(f"margin {j} is uniform (KS p = {p:.3f})", p > 0.01)

# C(u, 1, 1) = u -- the margin property, on the CDF this time.
grid = np.linspace(0.05, 0.95, 19)
points = np.ones((grid.size, 3))
points[:, 0] = grid
check(
    "C(u, 1, 1) = u to machine precision",
    np.allclose(cop.cdf(points), grid, atol=1e-12),
)

# Every box has non-negative probability -- the d-increasing property.
rng = np.random.default_rng(1)
volumes = [
    cop.prob(a := rng.uniform(0.05, 0.8, 3), np.minimum(a + rng.uniform(0.05, 0.15, 3), 0.99))
    for _ in range(200)
]
check("every box has non-negative probability", min(volumes) >= -1e-12)

heading("The parameter is recovered from the sample")

fitted = rc.fit(rc.ClaytonCopula(dim=3), u, method="mpl")
show("true theta", 2.0)
show("estimate", float(fitted.params[0]))
show("standard error", float(fitted.bse[0]))
low, high = fitted.conf_int()[0]
check(f"95% interval [{low:.3f}, {high:.3f}] contains the truth", low < 2.0 < high)

print(f"\n{fitted.summary()}")
