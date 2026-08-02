"""Inside an Archimedean copula: generators, frailties, and 100 dimensions.

Every Archimedean copula in this package is one function and its derivatives.
This opens the box -- and then pushes it to d = 100, where the arithmetic that
works fine in two dimensions stops working at all.

The last section is the point. A d-dimensional Archimedean density needs the
d-th derivative of the generator, and by d = 100 that number is far outside
double precision in either direction depending on the parameter. Everything here
is computed in logs for that reason, not for tidiness.

References
----------
McNeil, A. J. and Neslehova, J. (2009). Multivariate Archimedean copulas,
    d-monotone functions and l1-norm symmetric distributions.
    *Annals of Statistics* 37(5B), 3059-3097.
Marshall, A. W. and Olkin, I. (1988). Families of multivariate distributions.
    *JASA* 83(403), 834-841.  The frailty sampling algorithm.
Hofert, M. (2011). Efficiently sampling nested Archimedean copulas.
    *Computational Statistics and Data Analysis* 55(1), 57-70.
Hofert, M., Machler, M. and McNeil, A. J. (2012). Likelihood inference for
    Archimedean copulas in high dimensions under known margins.
    *J. Multivariate Analysis* 110, 133-150.
    Why the log scale is not optional past a handful of dimensions.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show

import rcopula as rc
from rcopula.special.stable import retstable, rstable_positive

heading("The generator, and its inverse")

# An Archimedean copula is C(u) = psi(ipsi(u_1) + ... + ipsi(u_d)). Everything
# else -- tau, tail dependence, the sampler, the density -- follows from psi.
rng = np.random.default_rng(0)
grid = np.linspace(0.01, 0.99, 99)

print(f"  {'family':10}{'psi(ipsi(u)) - u':>22}{'psi(0)':>10}{'psi(inf)':>11}")
worst = 0.0
for copula in (rc.ClaytonCopula(2.0), rc.GumbelCopula(2.0), rc.FrankCopula(5.0), rc.JoeCopula(2.0)):
    generator, theta = copula.generator, float(copula.params[0])
    error = float(np.max(np.abs(generator.psi(generator.ipsi(grid, theta), theta) - grid)))
    worst = max(worst, error)
    at_zero = float(generator.psi(np.array([0.0]), theta)[0])
    at_infinity = float(generator.psi(np.array([1e12]), theta)[0])
    print(f"  {generator.name:10}{error:>22.2e}{at_zero:>10.4f}{at_infinity:>11.4f}")

check("every generator inverts to machine precision", worst < 1e-12)
print(
    "\n  psi(0) = 1 and psi(inf) = 0 are not decoration: they are what force the\n"
    "  margins to be uniform. A function without them generates something that\n"
    "  is not a copula."
)

heading("An Archimedean copula is a mixture, and the mixing variable has a name")


# Marshall and Olkin: draw a frailty V whose Laplace transform is psi, then set
# U_j = psi(E_j / V) with independent exponentials. That *is* the copula.
def frailty_sample(draw, generator, theta, size):
    """The Marshall-Olkin algorithm, written out."""
    v = draw(size)
    e = rng.exponential(1.0, size=(size, 2))
    return generator.psi(e / v[:, None], theta)


print(f"  {'construction':34}{'sample tau':>12}{'copula tau':>12}")
cases = [
    (
        "gamma frailty -> Clayton",
        rc.ClaytonCopula(2.0),
        lambda n: rng.gamma(1 / 2.0, scale=1.0, size=n),
    ),
    (
        "positive stable -> Gumbel",
        rc.GumbelCopula(2.0),
        lambda n: rstable_positive(n, 1 / 2.0, rng),
    ),
]
for label, copula, draw in cases:
    u = frailty_sample(draw, copula.generator, float(copula.params[0]), 200_000)
    observed = float(rc.cor_kendall(u)[0, 1])
    print(f"  {label:34}{observed:>12.4f}{copula.tau():>12.4f}")
    check(f"{label} reproduces the copula", abs(observed - copula.tau()) < 0.01)

show("\nand the generator IS the frailty's Laplace transform, E[exp(-tV)]:", "")
frailty = rng.gamma(0.5, scale=1.0, size=2_000_000)
for t in (0.25, 1.0, 4.0):
    transform = float(np.mean(np.exp(-t * frailty)))
    exact = float(rc.ClaytonCopula(2.0).generator.psi(np.array([t]), 2.0)[0])
    print(f"    t = {t:<5} E[exp(-tV)] = {transform:.5f}   psi(t) = {exact:.5f}")
    check(f"they agree at t={t}", abs(transform - exact) < 0.002)

heading("The tilted stable sampler, which nested Clayton needs")

# retstable draws from an exponentially tilted stable law. Its whole purpose is
# the Laplace transform below; nothing else in NumPy or SciPy produces it, and
# without it a nested Clayton copula cannot be sampled at all.
alpha, v0, h = 0.5, 1.3, 1.0
draws = retstable(400_000, alpha, np.full(400_000, v0), h, rng)
print(f"  {'t':>6}{'E[exp(-tV)] sampled':>24}{'exact':>14}")
for t in (0.5, 1.0, 3.0):
    sampled = float(np.mean(np.exp(-t * draws)))
    exact = float(np.exp(-v0 * ((t + h) ** alpha - h**alpha)))
    print(f"  {t:>6}{sampled:>24.6f}{exact:>14.6f}")
    check(f"tilted stable transform at t={t}", abs(sampled - exact) < 0.003)
print(
    "\n  The naive way to draw this is rejection sampling, which accepts with\n"
    "  probability exp(-V0 h^alpha). For a Gamma-distributed V0 the *expected*\n"
    "  number of attempts is infinite -- the moment generating function diverges\n"
    "  -- so it does not run slowly, it sometimes does not finish. Splitting the\n"
    "  draw by infinite divisibility makes the cost linear instead."
)

heading("One hundred dimensions, where the arithmetic runs out")

# The d-dimensional density needs the d-th derivative of psi. In two dimensions
# nobody thinks about it. At d = 100 it is the whole problem.
dimension, sample_size = 100, 200
print(
    f"  {'theta':>7}{'log|psi^(100)| range over the 200 rows':>42}"
    f"{'rows lost to exp()':>21}{'log-likelihood':>17}"
)
lost_total = 0
for theta in (1.25, 1.5, 1.75, 2.0):
    copula = rc.JoeCopula(theta, dim=dimension)
    u = copula.rvs(sample_size, random_state=0)
    generator = copula.generator
    total = np.sum(generator.ipsi(u, theta), axis=1)
    log_derivative = generator.log_abs_dpsi_d(total, theta, dimension)
    with np.errstate(over="ignore"):
        naive = np.exp(log_derivative)
    lost = int(np.sum(~np.isfinite(naive)))
    lost_total += lost
    loglik = float(np.sum(copula.logpdf(u)))
    span = f"[{log_derivative.min():.1f}, {log_derivative.max():.1f}]"
    print(f"  {theta:>7}{span:>42}{lost:>21}{loglik:>17.2f}")
    check(f"the log-likelihood is finite at theta={theta}", np.isfinite(loglik))

show("rows that overflow exp() across the four fits", lost_total)
check("the naive linear-space version does lose rows", lost_total > 0)
print(
    "\n  The second column is the exponent an implementation holding psi^(100) as\n"
    "  a plain number would need. It spans hundreds of orders of magnitude within\n"
    "  a single sample, and at the larger thetas some rows land past the top of\n"
    "  double precision -- exp() returns inf and the likelihood becomes nan. The\n"
    "  last column is finite throughout, because it never leaves the log scale."
)

heading("And the model still fits at that size")

truth = rc.JoeCopula(1.75, dim=dimension)
data = truth.rvs(sample_size, random_state=1)
fitted = rc.fit(rc.JoeCopula(1.2, dim=dimension), data, method="mpl")
show("true theta", 1.75)
show("fitted theta", float(fitted.params[0]))
show("standard error", float(np.ravel(fitted.bse)[0]))
check(
    "recovered in 100 dimensions from 200 observations", abs(float(fitted.params[0]) - 1.75) < 0.1
)
check("with a usable standard error", 0.0 < float(np.ravel(fitted.bse)[0]) < 0.2)

# The Kendall tau of a 100-dimensional Archimedean copula is the same for every
# pair, which is exactly the limitation nested copulas and vines exist to lift.
pairs = rc.cor_kendall(data)
off_diagonal = pairs[np.triu_indices(dimension, 1)]
show(
    "\npairwise tau: min and max over 4950 pairs",
    (round(float(off_diagonal.min()), 3), round(float(off_diagonal.max()), 3)),
)
show("what the fitted copula says every pair is", fitted.copula.tau())
check(
    "one parameter means one dependence for all 4950 pairs",
    abs(float(np.median(off_diagonal)) - fitted.copula.tau()) < 0.05,
)
print(
    "\n  That is the ceiling of a flat Archimedean copula in high dimensions, and\n"
    "  the reason for 15_vine_copulas.py and the nested trees in 23_*.py: one\n"
    "  parameter cannot describe 4950 different pairs, however well it is fitted."
)
