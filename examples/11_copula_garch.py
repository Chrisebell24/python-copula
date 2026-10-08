"""Volatility in time, dependence in the cross-section.

Fitting a copula straight to returns confuses volatility clustering with
dependence. This script shows the confusion, then removes it. It ends with the
leverage effect -- volatility rising more after falls than after rallies --
captured by GJR-GARCH margins (``vol="gjr"``).

    ## R (the copula_GARCH vignette uses rugarch for the margins)
    ## fit <- lapply(1:2, function(j) ugarchfit(spec, x[, j]))
    ## z   <- sapply(fit, residuals, standardize = TRUE)
    ## fitCopula(tCopula(), pobs(z), method = "mpl")
    ## GJR margins: ugarchspec(variance.model = list(model = "gjrGARCH"))
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc
from rcopula.garch import CopulaGarch, fit_garch

heading("Two INDEPENDENT series sharing a volatility process")

rng = np.random.default_rng(1)
n = 4000
log_vol = np.zeros(n)
shocks = rng.standard_normal(n)
for t in range(1, n):
    log_vol[t] = 0.98 * log_vol[t - 1] + 0.25 * shocks[t]
returns = rng.standard_normal((n, 2)) * (0.01 * np.exp(log_vol))[:, None]

show("Kendall's tau of the raw returns", float(rc.cor_kendall(returns)[0, 1]))
show(
    "Spearman rho of the SQUARED returns",
    float(stats.spearmanr(returns[:, 0] ** 2, returns[:, 1] ** 2).statistic),
)

print(
    "\n  Rank correlation says nothing -- the signs are independent. The squared\n"
    "  returns co-move strongly, because volatility is shared. That shows up as\n"
    "  apparent TAIL DEPENDENCE, which is the part that matters for risk."
)

heading("What a copula fitted to the raw returns concludes")

raw_fit = rc.fit(rc.StudentCopula(0.0, df=8.0), rc.pseudo_obs(returns), method="mpl")
show("degrees of freedom", float(raw_fit.params[1]))
show("implied tail dependence", float(raw_fit.copula.lambda_().upper))

heading("And what it concludes after filtering")

model = CopulaGarch.fit(returns, rc.StudentCopula(0.0, df=8.0, dim=2))
show("degrees of freedom", float(model.copula.df))
show("implied tail dependence", float(model.copula.lambda_().upper))

check(
    "the raw fit sees far heavier joint tails than are there",
    raw_fit.copula.lambda_().upper > 3 * model.copula.lambda_().upper,
)
check(
    "and rank correlation is near zero either way, so it never flagged it",
    abs(raw_fit.params[0]) < 0.06 and abs(model.copula.params[0]) < 0.06,
)

heading("The marginal models")

print(model.summary().to_string(float_format=lambda v: f"{v:.5f}"))
for margin in model.margins:
    check(f"persistence {margin.persistence:.4f} is below one", margin.persistence < 1.0)

heading("Forecasting a joint distribution")

# Now with genuine dependence, so the forecast has something to say.
truth = rc.StudentCopula(0.6, df=4.0)
u = truth.rvs(2500, random_state=0)
z = stats.norm.ppf(u)


def _apply(series: np.ndarray, omega=2e-6, alpha=0.08, beta=0.90) -> np.ndarray:
    out = np.empty_like(series)
    var, shock = omega / (1 - alpha - beta), 0.0
    for i, innovation in enumerate(series):
        var = omega + alpha * shock**2 + beta * var
        shock = np.sqrt(var) * innovation
        out[i] = shock
    return out


data = np.column_stack([_apply(z[:, 0]), _apply(z[:, 1])])
fitted = CopulaGarch.fit(data, rc.StudentCopula(0.0, df=8.0, dim=2))
show("recovered correlation", float(fitted.copula.params[0]))
show("recovered degrees of freedom", float(fitted.copula.df))
check("the injected dependence is recovered", abs(fitted.copula.params[0] - 0.6) < 0.08)

for horizon in (1, 5, 20):
    risk = fitted.forecast_risk(alpha=0.99, horizon=horizon, n=40_000, random_state=0)
    print(
        f"  {horizon:>2}-day: VaR {risk['var']:.5f}   ES {risk['expected_shortfall']:.5f}"
        f"   vol {risk['volatility']:.5f}"
    )

one = fitted.forecast_risk(alpha=0.99, horizon=1, n=40_000, random_state=0)
twenty = fitted.forecast_risk(alpha=0.99, horizon=20, n=40_000, random_state=0)
check("risk grows with the horizon", twenty["var"] > one["var"])
check("expected shortfall exceeds VaR", one["expected_shortfall"] > one["var"])

heading("Tail dependence costs capital at the same rank correlation")

margins = [fit_garch(data[:, j]) for j in range(2)]
tau = 0.4
gaussian = CopulaGarch(margins, rc.GaussianCopula.from_tau(tau))
student = CopulaGarch(margins, rc.StudentCopula.from_tau(tau, df=3.0))
a = gaussian.forecast_risk(alpha=0.995, n=60_000, random_state=0)
b = student.forecast_risk(alpha=0.995, n=60_000, random_state=0)
show("99.5% VaR, Gaussian dependence", a["var"])
show("99.5% VaR, Student(3) dependence", b["var"])
check("the tail-dependent copula demands more", b["var"] > a["var"])

heading("The leverage effect: bad news raises volatility more (GJR-GARCH)")


def _leverage(seed: int, n: int = 4000) -> np.ndarray:
    """A falling market feeds 0.15 of a squared shock into next period's
    variance, a rising one only 0.03 -- the asymmetry equity indices show."""
    out = np.empty(n)
    var, shock = 7e-5, 0.0
    for i, innovation in enumerate(np.random.default_rng(seed).standard_normal(n)):
        var = 2e-6 + (0.03 + 0.12 * (shock < 0)) * shock**2 + 0.88 * var
        shock = np.sqrt(var) * innovation
        out[i] = shock
    return out


asym = np.column_stack([_leverage(7), _leverage(8)])
plain = fit_garch(asym[:, 0])
gjr = fit_garch(asym[:, 0], vol="gjr")
print(f"  {plain!r}\n  {gjr!r}")
show("alpha (any shock), true 0.03", gjr.alpha)
show("gamma (extra for a negative shock), true 0.12", gjr.gamma)
show("AIC, GARCH", plain.aic)
show("AIC, GJR-GARCH", gjr.aic)
check("the fit finds the asymmetry", gjr.gamma > 0.05)
check("and GJR beats plain GARCH by AIC", gjr.aic < plain.aic)
check("persistence alpha + gamma/2 + beta stays below one", gjr.persistence < 1.0)

# The same switch works for the joint model; simulation then uses the GJR
# recursion, so a sell-off raises the simulated volatility more than a rally.
lev_model = CopulaGarch.fit(asym, rc.StudentCopula(0.0, df=8.0, dim=2), vol="gjr")
print(lev_model.summary()[["alpha", "gamma", "beta", "persistence"]].to_string())
check("both margins show the leverage effect", bool((lev_model.summary()["gamma"] > 0.05).all()))
