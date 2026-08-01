"""Dependence that moves: Patton's recursion, GAS, and DCC.

Every other example in this gallery fits one parameter to a whole sample. That
is a strong assumption and usually a false one. Here the copula parameter is
allowed to follow the data.

The trap this example exists to demonstrate is that **unfiltered margins produce
time-varying dependence that is not there**. Volatility clustering left in the
series is picked up by the recursion and reported as a moving copula, which is
the same confusion `11_copula_garch.py` exists to prevent -- except that here it
produces a number that looks like evidence.

References
----------
Patton, A. J. (2006). Modelling asymmetric exchange rate dependence.
    *International Economic Review* 47(2), 527-556.
Creal, D., Koopman, S. J. and Lucas, A. (2013). Generalized autoregressive
    score models with applications. *J. Applied Econometrics* 28(5), 777-795.
Engle, R. F. (2002). Dynamic conditional correlation.
    *J. Business and Economic Statistics* 20(3), 339-350.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show

import rcopula as rc
from rcopula.dynamic import DynamicCopula, fit_dcc, fit_dynamic

heading("Recovering a dependence path that really does move")

# R's copula package has nothing like this: the parameter is a deterministic
# function of the past, so the likelihood stays exact and there is no latent
# state to integrate out.
truth = DynamicCopula(rc.GaussianCopula(0.0), coefficients=(0.30, 1.0, 0.80))
u = truth.simulate(1500, random_state=0)
true_path = truth.filter(u).path

show("true rho, range", (round(float(true_path.min()), 3), round(float(true_path.max()), 3)))
show("true rho, standard deviation", float(true_path.std()))

fitted = fit_dynamic(u, rc.GaussianCopula(0.0))
print()
print(fitted.summary())

correlation = float(np.corrcoef(true_path, fitted.path)[0, 1])
show("\ncorrelation between true and filtered path", correlation)
check("the path is recovered", correlation > 0.9)
check("and the maximiser beat the generating coefficients", fitted.loglik >= truth.loglik(u))
check("a constant copula is decisively worse", fitted.loglik > fitted.constant_loglik + 20)

# A single number fitted to the same data is an average of a moving thing.
show("constant rho fitted to it all", fitted.constant_param)
show("but rho was actually above 0.6 for", f"{np.mean(true_path > 0.6):.1%} of the sample")

heading("Does unfiltered data fool the recursion? Test it rather than assume")

# Two genuinely independent series that share a persistent volatility. The
# natural worry is that this manufactures a moving dependence parameter. It
# does not -- and the reason is worth knowing, because the failure it *does*
# cause is a different one.
rng = np.random.default_rng(1)
n = 2000
log_volatility = np.zeros(n)
for t in range(1, n):
    log_volatility[t] = 0.98 * log_volatility[t - 1] + 0.15 * rng.standard_normal()
volatility = np.exp(log_volatility)

innovations = rng.standard_normal((n, 2))  # independent by construction
returns = innovations * volatility[:, None]  # sharing one volatility path

raw = rc.pseudo_obs(returns)
clean = rc.pseudo_obs(innovations)  # what a GARCH filter would hand back


def joint_tail(data: np.ndarray, q: float = 0.95) -> float:
    """P(both above q) divided by 1-q. One under independence at any q."""
    return float(np.mean(np.all(data > q, axis=1))) / (1 - q)


print(f"  {'':<24}{'tau':>9}{'LR vs constant':>17}{'p':>8}{'joint upper tail':>19}")
results = {}
for label, data in [("raw returns", raw), ("filtered innovations", clean)]:
    result = fit_dynamic(data, rc.GaussianCopula(0.0))
    statistic, pvalue = result.constancy_test()
    results[label] = (result, joint_tail(data), rc.select_copula(data, criterion="aic"))
    print(
        f"  {label:<24}{rc.cor_kendall(data)[0, 1]:>9.4f}{statistic:>17.2f}"
        f"{pvalue:>8.3f}{joint_tail(data):>19.3f}"
    )

check(
    "neither fit rejects constancy -- the recursion is not fooled",
    all(results[k][0].constancy_test()[1] > 0.05 for k in results),
)
print(
    "\n  It cannot be fooled, and the reason is structural: pseudo-observations\n"
    "  are ranks, ranks are invariant to any common increasing transform of a\n"
    "  coordinate, and independent series stay independent under one. No amount\n"
    "  of marginal misspecification can conjure rank dependence out of nothing."
)

show("family selected on raw returns", results["raw returns"][2].best.describe())
show("family selected on filtered", results["filtered innovations"][2].best.describe())
show("joint upper tail, raw", results["raw returns"][1])
show("joint upper tail, filtered", results["filtered innovations"][1])
check(
    "what the shared volatility really creates is tail dependence",
    results["raw returns"][1] > 5 * results["filtered innovations"][1],
)
print(
    "\n  Both series are large together whenever volatility is high, so the raw\n"
    "  pair joins its 95th percentiles nine times more often than independence\n"
    "  allows and a t copula is selected at tau near zero. The damage from\n"
    "  skipping the filter is a wrong tail, not a wrong path -- which is the\n"
    "  failure 11_copula_garch.py is about. Filter anyway; just know which\n"
    "  mistake you are avoiding."
)

heading("Patton's forcing term against the likelihood score")

# The two recursions are different models, not two estimators of one. Patton
# needs a forcing term chosen by hand; GAS derives its own from the family, so
# a tail observation moves a Clayton copula differently from a Gaussian one.
# Fitting each to data generated by each is the only comparison that is fair to
# both.
generators = {
    "patton": DynamicCopula(rc.ClaytonCopula(1.0), coefficients=(-1.2, -1.5, 0.85)),
    "gas": DynamicCopula(rc.ClaytonCopula(1.0), coefficients=(-0.2, 0.05, 0.90), driver="gas"),
}

print(f"  {'generated by':<16}{'fitted by':<12}{'loglik':>11}{'AIC':>11}{'corr w/ truth':>16}")
agreement: dict[tuple[str, str], float] = {}
fits: dict[tuple[str, str], object] = {}
for generating, model in generators.items():
    v = model.simulate(1000, random_state=0)
    truth_path = model.filter(v).path
    for fitting in ("patton", "gas"):
        result = fit_dynamic(v, rc.ClaytonCopula(1.0), driver=fitting)
        agreement[generating, fitting] = float(np.corrcoef(truth_path, result.path)[0, 1])
        fits[generating, fitting] = result
        print(
            f"  {generating:<16}{fitting:<12}{result.loglik:>11.3f}{result.aic:>11.3f}"
            f"{agreement[generating, fitting]:>16.4f}"
        )
    if generating == "patton":
        patton_result = fits["patton", "patton"]

check(
    "each recursion tracks its own generating process best",
    agreement["patton", "patton"] > agreement["patton", "gas"]
    and agreement["gas", "gas"] > agreement["gas", "patton"],
)
print(
    "\n  The off-diagonal is the interesting part: a misspecified recursion loses\n"
    "  little log-likelihood and a great deal of the path. Comparing two drivers\n"
    "  on fit alone would call them nearly equivalent, and they are not."
)

show("Patton's alpha (negative: |u-v| falls as dependence rises)", patton_result.coefficients[1])
check("and the sign is the one the forcing term implies", patton_result.coefficients[1] < 0)
show(
    "GAS alpha (a score, so the usual sign convention applies)", fits["gas", "gas"].coefficients[1]
)

heading("Forecasting the dependence, not just the level")

v = generators["patton"].simulate(1000, random_state=0)
ahead = patton_result.model.forecast(v, horizon=10, draws=800, random_state=3)
print(f"  {'h':>3}{'mean theta':>14}{'5th':>10}{'95th':>10}{'implied tau':>14}")
for h in (0, 2, 4, 9):
    theta = float(ahead["mean"][h])
    print(
        f"  {h + 1:>3}{theta:>14.4f}{ahead['lower'][h]:>10.4f}"
        f"{ahead['upper'][h]:>10.4f}{rc.ClaytonCopula(theta).tau():>14.4f}"
    )
width = ahead["upper"] - ahead["lower"]
check("uncertainty about the parameter widens with the horizon", width[-1] > width[0])
print(
    "\n  Beyond one step the parameter is a random variable, because the\n"
    "  recursion is driven by observations that have not happened. Reporting a\n"
    "  point forecast for it alone would be a category error."
)

heading("Many assets at once: dynamic conditional correlation")

# For d > 2 the object that moves is a whole matrix, and the standard recursion
# for that is Engle's. Regime switch built in: independent, then dependent.
early = rc.GaussianCopula(0.05, dim=4, dispstr="ex").rvs(700, random_state=0)
late = rc.GaussianCopula(0.80, dim=4, dispstr="ex").rvs(700, random_state=1)
panel = np.vstack([early, late])

dcc = fit_dcc(panel)
print()
print(dcc.summary())

path = dcc.pair(0, 1)
before = float(path[600:700].mean())
after = float(path[1300:1400].mean())
show("\nmean correlation, quiet regime", before)
show("mean correlation, dependent regime", after)
check("the filter finds the switch", after > before + 0.4)
check(
    "every filtered matrix is a correlation matrix",
    all(np.all(np.linalg.eigvalsh(m) > 0) for m in dcc.correlations[::100]),
)
show("unconditional target it reverts to", float(dcc.unconditional[0, 1]))
print(
    "\n  Note the target is the whole-sample correlation, which describes\n"
    "  neither regime. That is the point: correlation targeting fixes the level\n"
    "  the recursion reverts to, and the recursion supplies everything else."
)
