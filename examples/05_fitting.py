"""Five estimators, and the standard errors no other Python package reports.

## R
## fitCopula(claytonCopula(), u, method = "mpl")
## fitCopula(claytonCopula(), u, method = "itau")
## coef(fit); vcov(fit); logLik(fit)
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc

TRUTH = rc.ClaytonCopula(2.5)
raw = rc.CopulaDistribution(TRUTH, margins=[stats.norm(0, 1), stats.gamma(2, scale=3)]).rvs(
    1500, random_state=0
)

heading("The five methods R offers, on the same data")

print(f"  {'method':<10}{'estimate':>11}{'std. error':>13}{'log-lik':>11}")
for method in ("mpl", "ml", "itau", "irho"):
    res = rc.fit(rc.ClaytonCopula(), raw, method=method)
    se = "n/a" if res.bse is None else f"{res.bse[0]:.6f}"
    print(f"  {method:<10}{res.params[0]:>11.6f}{se:>13}{res.loglik:>11.3f}")

heading("mpl and ml give the SAME estimate, different variances")

mpl = rc.fit(rc.ClaytonCopula(), raw, method="mpl")
ml = rc.fit(rc.ClaytonCopula(), raw, method="ml")
check("the point estimates agree exactly", abs(mpl.params[0] - ml.params[0]) < 1e-8)
show("mpl standard error", float(mpl.bse[0]))
show("ml standard error", float(ml.bse[0]))
print(
    "\n  The gap is the Genest-Ghoudi-Rivest correction for having estimated the\n"
    "  margins by ranks. Use 'ml' only when the data really ARE copula\n"
    "  observations; on raw data like this, 'mpl' is the honest one."
)

heading("Are the standard errors any good?")

# The only real test: do they predict the actual spread of the estimator?
estimates, errors = [], []
for seed in range(200):
    sample = TRUTH.rvs(1500, random_state=seed)
    fitted = rc.fit(rc.ClaytonCopula(), rc.pseudo_obs(sample), method="mpl")
    estimates.append(fitted.params[0])
    errors.append(fitted.bse[0])

ratio = float(np.mean(errors)) / float(np.std(estimates, ddof=1))
show("mean reported standard error", float(np.mean(errors)))
show("actual spread over 200 replications", float(np.std(estimates, ddof=1)))
show("ratio", ratio)
check("the standard error is calibrated within 10%", 0.9 < ratio < 1.1)

coverage = np.mean([abs(e - 2.5) < 1.96 * s for e, s in zip(estimates, errors, strict=True)])
show("95% interval coverage", float(coverage))
check("coverage is close to nominal", 0.90 < coverage < 0.99)

heading("Holding a parameter fixed (R's fixParam)")

u = rc.StudentCopula(0.6, df=4.0).rvs(2000, random_state=0)
free = rc.fit(rc.StudentCopula(), u, method="mpl")
pinned = rc.fit(rc.StudentCopula(0.5, df=4.0, df_fixed=True), u, method="mpl")
show("df estimated", float(free.params[1]))
show("df pinned at 4, correlation", float(pinned.params[0]))
check("pinning df costs one parameter", pinned.n_params == free.n_params - 1)
check("and cannot improve the fit", pinned.loglik <= free.loglik + 1e-9)
