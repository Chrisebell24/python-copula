"""Criticising a fitted copula, and reshaping one.

Fitting is the easy half. This is the other half: the questions a fitted model
should be made to survive, and the transformations that give you somewhere to
go when it does not.

Six tools, none of which appears elsewhere in this gallery, and each of which
answers a question a log-likelihood cannot.

References
----------
Remillard, B. and Scaillet, O. (2009). Testing for equality between two
    copulas. *J. Multivariate Analysis* 100(3), 377-386.
Genest, C., Remillard, B. and Beaudoin, D. (2009). Goodness-of-fit tests for
    copulas: a review and a power study.
    *Insurance: Mathematics and Economics* 44(2), 199-213.
McNeil, A. J. and Neslehova, J. (2009). Multivariate Archimedean copulas,
    d-monotone functions and l1-norm symmetric distributions.
    *Annals of Statistics* 37(5B), 3059-3097.
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc
from rcopula.gof import gof_two_sample
from rcopula.htest import serial_indep_test
from rcopula.plots import pairs_rosenblatt
from rcopula.transforms import radial_cdf, radial_ppf, radial_simplex

heading("Where a fit fails, not whether")

# A goodness-of-fit test returns one number. The Rosenblatt transform turns the
# question into d(d-1)/2 of them, one per pair, and names the culprit.
truth = rc.ClaytonCopula(3.0, dim=3)
data = truth.rvs(2000, random_state=0)


def panel_pvalues(model: rc.Copula, sample: np.ndarray) -> list[float]:
    """Independence p-value per pair of the Rosenblatt-transformed data."""
    z = np.asarray(rc.rosenblatt(model, sample))
    n, d = z.shape
    scale = np.sqrt(9.0 * n * (n - 1) / (2.0 * (2 * n + 5)))
    return [
        float(2.0 * stats.norm.sf(abs(scale * stats.kendalltau(z[:, i], z[:, j]).statistic)))
        for i in range(d)
        for j in range(i + 1, d)
    ]


print(f"  {'fitted model':<26}{'(1,2)':>9}{'(1,3)':>9}{'(2,3)':>9}   verdict")
verdicts = {}
for label, model in [
    ("Clayton(3) -- the truth", truth),
    ("Gaussian", rc.GaussianCopula(0.6, dim=3, dispstr="ex")),
    ("Gumbel", rc.GumbelCopula(2.5, dim=3)),
]:
    values = panel_pvalues(model, data)
    verdicts[label] = values
    flag = "ok" if min(values) > 0.05 else "REJECTED"
    print(f"  {label:<26}" + "".join(f"{p:>9.4f}" for p in values) + f"   {flag}")

check("the true model survives every panel", min(verdicts["Clayton(3) -- the truth"]) > 0.05)
check("a plainly wrong family fails every panel", max(verdicts["Gaussian"]) < 0.01)
check(
    "and Gumbel passes one panel while failing others",
    max(verdicts["Gumbel"]) > 0.05 and min(verdicts["Gumbel"]) < 0.01,
)
print(
    "\n  The Gumbel row is the whole argument for a picture over a scalar. It\n"
    "  matches the first pair perfectly well and fails the two conditional ones,\n"
    "  which a single test statistic collapses into one uninformative rejection.\n"
    "  Because the Rosenblatt transform conditions successively, a failure at\n"
    "  (2,3) is a failure of *conditional* dependence -- exactly what a vine's\n"
    "  higher trees exist to fix."
)

axes = pairs_rosenblatt(rc.GumbelCopula(2.5, dim=3), data)
show("the same thing as a figure", f"{axes.shape[0]}x{axes.shape[1]} panel grid")

heading("Did the dependence change?")

# No model at all: two samples compared to each other. The question behind
# "did correlations break in the crisis?".
before = rc.GaussianCopula.from_tau(0.35).rvs(700, random_state=0)
after_same = rc.GaussianCopula.from_tau(0.35).rvs(700, random_state=1)
after_shifted = rc.GaussianCopula.from_tau(0.60).rvs(700, random_state=2)
after_reshaped = rc.ClaytonCopula.from_tau(0.35).rvs(700, random_state=3)

print(f"  {'second period':<34}{'tau':>8}{'p-value':>10}   conclusion")
for label, second in [
    ("same copula", after_same),
    ("same family, stronger (tau .35 -> .60)", after_shifted),
    ("same tau, different shape (Clayton)", after_reshaped),
]:
    result = gof_two_sample(before, second, n_rep=400, random_state=0)
    print(
        f"  {label:<34}{rc.cor_kendall(second)[0, 1]:>8.3f}{result.pvalue:>10.4f}"
        f"   {'unchanged' if result.pvalue > 0.05 else 'CHANGED'}"
    )

unchanged = gof_two_sample(before, after_same, n_rep=400, random_state=0)
check("no change is not flagged", unchanged.pvalue > 0.05)
check(
    "a shift in strength is",
    gof_two_sample(before, after_shifted, n_rep=400, random_state=0).pvalue < 0.05,
)
check(
    "and so is a change of shape at identical tau",
    gof_two_sample(before, after_reshaped, n_rep=400, random_state=0).pvalue < 0.05,
)

# The design guarantee: it cannot see the margins.
rescaled = np.column_stack(
    [stats.expon(scale=50).ppf(after_same[:, 0]), stats.norm(-9, 0.01).ppf(after_same[:, 1])]
)
plain = gof_two_sample(before, after_same, n_rep=400, random_state=0)
warped = gof_two_sample(before, rescaled, n_rep=400, random_state=0)
show("\np-value, plain second sample", plain.pvalue)
show("p-value after mangling its margins", warped.pvalue)
check("identical -- the test cannot see the margins", warped.pvalue == plain.pvalue)
print(
    "\n  That third row is the one worth dwelling on. Kendall's tau is the same\n"
    "  0.35 in both periods, so any correlation-based monitor reports nothing,\n"
    "  and the dependence has changed from no tail dependence to 0.71 in the\n"
    "  lower tail. This is what breaks a portfolio."
)

heading("Dependence in time, which a correlogram can miss")

# Serial dependence, located by lag structure. The case that matters is a
# series uncorrelated in level and dependent in magnitude.
rng = np.random.default_rng(3)
volatility = np.exp(0.5 * rng.standard_normal(2000).cumsum() / 30)
returns = volatility * rng.standard_normal(2000)

show("lag-1 autocorrelation of returns", float(np.corrcoef(returns[:-1], returns[1:])[0, 1]))
level = serial_indep_test(returns, n_rep=300, random_state=1)
magnitude = serial_indep_test(np.abs(returns), n_rep=300, random_state=1)
show("serial independence of returns, p", level.global_pvalue)
show("serial independence of |returns|, p", magnitude.global_pvalue)
check("the levels look independent", abs(np.corrcoef(returns[:-1], returns[1:])[0, 1]) < 0.06)
check("the magnitudes plainly are not", magnitude.global_pvalue < 0.05)

print(f"\n  {'lag subset':>14}{'statistic':>13}{'p-value':>10}")
order = np.argsort(magnitude.pvalues)
for k in order[:4]:
    label = "{" + ",".join(str(j) for j in magnitude.subsets[k]) + "}"
    print(f"  {label:>14}{magnitude.statistics[k]:>13.6f}{magnitude.pvalues[k]:>10.4f}")
print(
    "\n  Index 0 is today and k is k days back, so {0,1} is consecutive-day\n"
    "  dependence. Being rank-based this needs no moments at all, which matters\n"
    "  for returns whose autocorrelation may not even be defined."
)

heading("The radial law, where an Archimedean family keeps its identity")

# Split a sample into radius and angle. The angular half is the same for every
# Archimedean copula in every dimension -- so everything that distinguishes one
# family from another is in the radial law.
print(f"  {'copula':<26}{'angular mean':>14}{'1/d':>7}{'radial median':>16}")
for copula in (
    rc.ClaytonCopula(2.0, dim=3),
    rc.GumbelCopula(2.0, dim=3),
    rc.FrankCopula(5.0, dim=3),
):
    radii, angles = radial_simplex(copula, copula.rvs(60_000, random_state=0))
    print(
        f"  {copula.describe()[:26]:<26}{float(angles.mean()):>14.4f}"
        f"{1 / copula.dim:>7.4f}{float(radial_ppf(copula, 0.5)[0]):>16.4f}"
    )

check(
    "every angular part has mean 1/d, whatever the family",
    all(
        abs(float(radial_simplex(c, c.rvs(40_000, random_state=0))[1].mean()) - 1 / c.dim) < 0.005
        for c in (rc.ClaytonCopula(2.0, dim=3), rc.GumbelCopula(2.0, dim=3))
    ),
)

# And the closed form matches what the sample actually does.
copula = rc.ClaytonCopula(2.0, dim=3)
radii, _ = radial_simplex(copula, copula.rvs(100_000, random_state=0))
print(f"\n  {'level':>8}{'empirical quantile':>22}{'radial_ppf':>14}")
worst = 0.0
for level in (0.1, 0.5, 0.9, 0.99):
    empirical = float(np.quantile(radii, level))
    exact = float(radial_ppf(copula, level)[0])
    worst = max(worst, abs(radial_cdf(copula, empirical)[0] - level))
    print(f"  {level:>8}{empirical:>22.4f}{exact:>14.4f}")
show("worst deviation of radial_cdf from the empirical level", worst)
check("the closed form matches the sample", worst < 0.01)
print(
    "\n  The radial quantiles differ by orders of magnitude between families while\n"
    "  the angular means are identical. That is the McNeil-Neslehova split: all\n"
    "  the family-specific information is in the radius, and it is why a\n"
    "  nonparametric test of *Archimedeanity* can exist at all -- the angular\n"
    "  part has nothing family-specific left to disagree with."
)

heading("Reshaping a family that will not fit")

# When the tails are wrong, the outer power transformation supplies a second
# parameter without changing family.
base = rc.ClaytonCopula(2.0)
print(f"  {'copula':<34}{'tau':>8}{'lambda_L':>11}{'lambda_U':>11}")
print(
    f"  {'Clayton(2)':<34}{base.tau():>8.4f}"
    f"{base.lambda_().lower:>11.4f}{base.lambda_().upper:>11.4f}"
)
for alpha in (1.5, 2.0, 3.0):
    lifted = rc.opower(base, alpha)
    print(
        f"  {f'opower(Clayton(2), alpha={alpha})':<34}{lifted.tau():>8.4f}"
        f"{lifted.lambda_().lower:>11.4f}{lifted.lambda_().upper:>11.4f}"
    )

check("Clayton has no upper tail dependence", base.lambda_().upper == 0.0)
check("the transformation creates some", rc.opower(base, 2.0).lambda_().upper > 0.5)
check(
    "and tau follows Nelsen's closed form",
    all(
        abs(rc.opower(base, a).tau() - (1 - (1 - base.tau()) / a)) < 1e-12 for a in (1.5, 2.0, 3.0)
    ),
)
print(
    "\n  A one-parameter family gives you one shape. If the data has both tails\n"
    "  and Clayton only has the lower one, this is a second dial that does not\n"
    "  require abandoning the family -- and applied to the independence\n"
    "  generator it *is* the Gumbel copula, which is the cleanest statement of\n"
    "  what it does."
)

heading("Simulating with the data's own margins")

# The last step of a study that wants the copula's dependence and the sample's
# marginal shapes: no parametric margin is fitted at all.
# Genuinely dependent, with awkward margins: skewed and heavy-tailed. The
# dependence is real so there is something to reproduce.
latent = rc.ClaytonCopula.from_tau(0.45).rvs(1500, random_state=0)
history = np.column_stack(
    [
        stats.skewnorm(-6, loc=0.01, scale=0.03).ppf(latent[:, 0]),
        stats.t(3, loc=0.005, scale=0.02).ppf(latent[:, 1]),
    ]
)
fitted = rc.select_copula(rc.pseudo_obs(history), criterion="aic").best
simulated = rc.to_emp_margins(fitted.rvs(50_000, random_state=0), history)

show("family selected", fitted.describe())
print(f"\n  {'':<12}{'history':>12}{'simulated':>12}")
for name, fn in (("median", np.median), ("5th pct", lambda a, **k: np.percentile(a, 5, **k))):
    h, s = fn(history, axis=0), fn(simulated, axis=0)
    print(f"  {name:<12}{h[0]:>12.4f}{s[0]:>12.4f}")
show("tau, history", float(rc.cor_kendall(history)[0, 1]))
show("tau, simulated", float(rc.cor_kendall(simulated)[0, 1]))
check(
    "the dependence is reproduced",
    abs(rc.cor_kendall(simulated)[0, 1] - rc.cor_kendall(history)[0, 1]) < 0.03,
)
show("worst simulated loss", float(simulated.min()))
show("worst loss in the history", float(history.min()))
check("nothing worse than history can ever come out", simulated.min() >= history.min())
print(
    "\n  That last line is the trap. An empirical quantile function cannot\n"
    "  extrapolate, so a 99.9% capital number computed this way can never exceed\n"
    "  the worst loss already observed, however many paths are simulated. For a\n"
    "  historical-simulation study that is exactly right; for a tail study it is\n"
    "  a hard ceiling, and parametric margins are the answer."
)
