"""Getting the same answer with fewer draws.

Every Monte Carlo number in this gallery has a standard error, and the default
way to halve it is to quadruple the sample. These are the cheaper ways -- and,
just as importantly, the cases where they do not work, because the literature
tends to report only the first kind.

References
----------
Cambou, M., Hofert, M. and Lemieux, C. (2017). Quasi-random numbers for copula
    models. *Statistics and Computing* 27(5), 1307-1329.
Owen, A. B. (1997). Scrambled net variance for integrals of smooth functions.
    *Annals of Statistics* 25(4), 1541-1562.
Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show

import rcopula as rc
from rcopula.sampling import antithetic_rvs, latin_hypercube_rvs, quasi_rvs, variance_ratio
from rcopula.transforms import inverse_rosenblatt, rosenblatt

heading("The machinery: an exact inverse, both ways")

# Quasi-random copula sampling is not a different sampler -- it is the ordinary
# inverse Rosenblatt transform fed a better set of uniforms. So the transform
# has to be exactly invertible first.
copula = rc.ClaytonCopula(3.0, dim=3)
u = copula.rvs(2000, random_state=0)

z = np.random.default_rng(0).uniform(size=(2000, 3))
forward_back = float(np.max(np.abs(inverse_rosenblatt(copula, rosenblatt(copula, u)) - u)))
back_forward = float(np.max(np.abs(rosenblatt(copula, inverse_rosenblatt(copula, z)) - z)))
show("u -> z -> u, max error", forward_back)
show("z -> u -> z, max error", back_forward)
check("the two transforms invert each other to machine precision", forward_back < 1e-9)
print(
    "\n  The inverse is found by bisection against the forward transform itself,\n"
    "  not derived separately, so the two cannot drift apart. That matters more\n"
    "  than it sounds: two independent derivations agreeing is evidence, but two\n"
    "  that are the same object by construction cannot disagree at all."
)

heading("A low-discrepancy point set fills the space on purpose")

target = rc.GumbelCopula(2.0)
quasi = quasi_rvs(target, 4096, random_state=0)
plain = target.rvs(4096, random_state=0)

show("Kendall tau, quasi-random", float(rc.cor_kendall(quasi)[0, 1]))
show("Kendall tau, plain", float(rc.cor_kendall(plain)[0, 1]))
show("   the truth", target.tau())
check("both have the right distribution", abs(rc.cor_kendall(quasi)[0, 1] - target.tau()) < 0.02)


# The direct measure of "evenly filled": the biggest hole in a margin.
def gap(sample: np.ndarray) -> float:
    """Largest distance between consecutive order statistics of a margin."""
    return float(np.max(np.diff(np.sort(sample[:, 0]))))


show("largest gap in a margin, quasi-random", gap(quasi))
show("largest gap in a margin, plain", gap(plain))
show("   ratio", gap(plain) / gap(quasi))
check("the quasi-random set leaves smaller holes", gap(quasi) < gap(plain))

heading("What each method is actually worth, by payoff")

# The honest way to report a variance reduction: run the estimator many times
# under each scheme and compare the spread of the answers.
base = rc.ClaytonCopula(2.0)
payoffs = {
    "smooth: u1 * u2": lambda x: x[:, 0] * x[:, 1],
    "kinked: call on u1+u2": lambda x: np.maximum(x.sum(axis=1) - 1.0, 0.0),
    "symmetric: |u1 - u2|": lambda x: np.abs(x[:, 0] - x[:, 1]),
    "indicator: both < 0.1": lambda x: np.all(x < 0.1, axis=1).astype(float),
}

print(f"  {'payoff':26}{'method':13}{'SE ratio':>10}{'equivalent draws':>19}")
REPLICATES = 20
results: dict[tuple[str, str], float] = {}
worst_z = 0.0
for name, payoff in payoffs.items():
    for method in ("sobol", "antithetic", "lhs"):
        out = variance_ratio(
            base, payoff, 1024, method=method, replicates=REPLICATES, random_state=0
        )
        results[name, method] = out["ratio"]
        # A method that lowers the standard error and moves the mean has not
        # reduced variance; it has introduced bias. The scale to judge a shift
        # against is the standard error of the *difference* of the two means,
        # not an absolute number -- both are themselves averages of 20 noisy
        # runs, and the reduced one is not always the tighter of the two.
        spread = np.sqrt((out["plain_se"] ** 2 + out["reduced_se"] ** 2) / REPLICATES)
        worst_z = max(worst_z, abs(out["plain_mean"] - out["reduced_mean"]) / spread)
        verdict = "x" if out["ratio"] < 1.0 else " "
        print(
            f"  {name:26}{method:13}{out['ratio']:>9.2f}{verdict}"
            f"{out['equivalent_sample_factor']:>19,.0f}"
        )

show("largest shift, in standard errors of the difference", worst_z)
check(
    "no method moved the estimate -- these are variance reductions, not biases",
    worst_z < 4.0,
)
check(
    "quasi-random helps most where the payoff is smoothest",
    results["smooth: u1 * u2", "sobol"] > 20 * results["indicator: both < 0.1", "sobol"],
)
check(
    "and antithetic pairing HURTS a symmetric payoff",
    results["symmetric: |u1 - u2|", "antithetic"] < 1.0,
)
print(
    "\n  Rows marked x are worse than plain Monte Carlo. Antithetic pairing is\n"
    "  usually described as free; it is free in cost and not in variance. For a\n"
    "  payoff symmetric about the middle of the distribution, z and 1-z move the\n"
    "  same way and the pairing adds variance instead of cancelling it.\n\n"
    "  Sobol survives the kink -- an option payoff is still worth sampling this\n"
    "  way -- but loses an order of magnitude against the discontinuous default\n"
    "  indicator, which is exactly the integrand a CDO tranche is made of."
)

heading("What that buys on a real calculation")

# A basket option under a t copula: smooth-ish, five-dimensional.
basket = rc.StudentCopula(0.5, df=5.0, dim=5, dispstr="ex")
strike = 0.5


def basket_payoff(sample: np.ndarray) -> np.ndarray:
    """A call on the average of five uniforms -- kinked, moderately smooth."""
    return np.maximum(sample.mean(axis=1) - strike, 0.0)


for size in (512, 2048, 8192):
    out = variance_ratio(basket, basket_payoff, size, method="sobol", replicates=20, random_state=0)
    print(
        f"  n = {size:>5}   plain SE {out['plain_se']:.2e}   "
        f"Sobol SE {out['reduced_se']:.2e}   ratio {out['ratio']:>6.1f}x"
    )

final = variance_ratio(basket, basket_payoff, 8192, method="sobol", replicates=20, random_state=0)
show("\nprice, plain Monte Carlo", final["plain_mean"])
show("price, quasi-random", final["reduced_mean"])
show("they agree to", abs(final["plain_mean"] - final["reduced_mean"]))
check("same answer, smaller error bar", abs(final["plain_mean"] - final["reduced_mean"]) < 1e-3)
check("and the error bar is genuinely smaller", final["ratio"] > 3.0)

heading("Latin hypercube, when the margins are what matter")

# LHS guarantees marginal coverage and says nothing about the joint structure.
stratified = latin_hypercube_rvs(rc.FrankCopula(5.0, dim=3), 2000, random_state=0)
ordinary = rc.FrankCopula(5.0, dim=3).rvs(2000, random_state=0)

show("mean of coordinate 1, Latin hypercube", float(stratified[:, 0].mean()))
show("mean of coordinate 1, plain", float(ordinary[:, 0].mean()))
show("   the truth", 0.5)
check(
    "stratification pins the marginal mean far tighter",
    abs(stratified[:, 0].mean() - 0.5) < abs(ordinary[:, 0].mean() - 0.5),
)
show("Kendall tau, Latin hypercube", float(rc.cor_kendall(stratified)[0, 1]))
show("   the truth", rc.FrankCopula(5.0).tau())
check(
    "and the dependence survives it",
    abs(rc.cor_kendall(stratified)[0, 1] - rc.FrankCopula(5.0).tau()) < 0.03,
)

antithetic = antithetic_rvs(rc.ClaytonCopula(2.0), 2000, random_state=0)
show(
    "\nantithetic pairs are exact on the first coordinate",
    float(np.max(np.abs(antithetic[:1000, 0] + antithetic[1000:, 0] - 1.0))),
)
check("z and 1-z, as advertised", np.allclose(antithetic[:1000, 0] + antithetic[1000:, 0], 1.0))
