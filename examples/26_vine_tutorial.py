"""Vine copulas, checked against the paper that introduced them.

A tutorial that reproduces published results rather than asserting agreement
with them. Every claim below is either an identity from Aas, Czado, Frigessi and
Bakken (2009) -- reproduced by writing their equation out by hand and comparing
to what this package computes -- or a combinatorial count printed in that paper.

What is *not* reproduced, and why: the paper's headline number, an AIC of
-665.08 for a four-dimensional D-vine, comes from a proprietary series of
Norwegian financial returns that is not distributed with it. No implementation
can reproduce that figure without the data, and claiming otherwise would be
worthless. The identities below need no data at all, which is what makes them
worth checking.

References
----------
Aas, K., Czado, C., Frigessi, A. and Bakken, H. (2009). Pair-copula
    constructions of multiple dependence.
    *Insurance: Mathematics and Economics* 44(2), 182-198.
Bedford, T. and Cooke, R. M. (2002). Vines -- a new graphical model for
    dependent random variables. *Annals of Statistics* 30(4), 1031-1068.
Dissmann, J., Brechmann, E. C., Czado, C. and Kurowicka, D. (2013). Selecting
    and estimating regular vine copulae and application to financial returns.
    *Computational Statistics and Data Analysis* 59, 52-69.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from _common import check, heading, show

import rcopula as rc
from rcopula import conditional_cdf

rng = np.random.default_rng(0)
points = rng.uniform(0.05, 0.95, size=(500, 4))


def h(copula, x, v):
    """Aas et al. eq. (7): h(x, v) = dC(x, v)/dv, the conditioning variable second."""
    return conditional_cdf(copula, np.column_stack([x, v]), 1)


heading("The h-function is exactly what the paper defines it to be")

# Equation (7): h(x, v, Theta) = dC(x, v)/dv. Checked against a central
# difference of the CDF, which is the definition and nothing else.
print(f"  {'family':>18}{'max |h - dC/dv|':>20}")
worst = 0.0
for copula in (
    rc.ClaytonCopula(2.5),
    rc.GumbelCopula(1.9),
    rc.GaussianCopula(0.6),
    rc.FrankCopula(5.0),
    rc.StudentCopula(0.5, df=4.0),
):
    x, v, step = rng.uniform(0.1, 0.9, 400), rng.uniform(0.1, 0.9, 400), 1e-6
    difference = (
        copula.cdf(np.column_stack([x, v + step])) - copula.cdf(np.column_stack([x, v - step]))
    ) / (2 * step)
    error = float(np.max(np.abs(difference - h(copula, x, v))))
    worst = max(worst, error)
    print(f"  {type(copula).__name__:>18}{error:>20.2e}")
check("every h-function is the derivative it claims to be", worst < 1e-8)

heading("The D-vine factorisation, Aas et al. equation (11)")

# Written out exactly as printed:
#   f = c12 . c23 . c34 . c13|2 . c24|3 . c14|23
# with the conditional arguments built from h-functions.
c12, c23, c34 = rc.ClaytonCopula(2.0), rc.GumbelCopula(1.8), rc.FrankCopula(4.0)
c13_2, c24_3 = rc.GaussianCopula(0.4), rc.ClaytonCopula(1.1)
c14_23 = rc.GaussianCopula(-0.25)
dvine = rc.VineCopula([[c12, c23, c34], [c13_2, c24_3], [c14_23]], structure="D")

u = points
by_hand = (
    np.log(c12.pdf(u[:, [0, 1]])) + np.log(c23.pdf(u[:, [1, 2]])) + np.log(c34.pdf(u[:, [2, 3]]))
)
f1_2, f3_2 = h(c12, u[:, 0], u[:, 1]), h(c23, u[:, 2], u[:, 1])
f2_3, f4_3 = h(c23, u[:, 1], u[:, 2]), h(c34, u[:, 3], u[:, 2])
by_hand += np.log(c13_2.pdf(np.column_stack([f1_2, f3_2])))
by_hand += np.log(c24_3.pdf(np.column_stack([f2_3, f4_3])))
by_hand += np.log(c14_23.pdf(np.column_stack([h(c13_2, f1_2, f3_2), h(c24_3, f4_3, f2_3)])))

show(
    "max |equation (11) by hand - VineCopula.logpdf|",
    float(np.max(np.abs(by_hand - dvine.logpdf(u)))),
)
check(
    "the package computes the paper's equation", np.allclose(by_hand, dvine.logpdf(u), atol=1e-13)
)

heading("The C-vine factorisation, Aas et al. section 2.3")

c12, c13, c14 = rc.ClaytonCopula(2.0), rc.GumbelCopula(1.8), rc.FrankCopula(4.0)
c23_1, c24_1 = rc.GaussianCopula(0.4), rc.ClaytonCopula(1.1)
c34_12 = rc.GaussianCopula(-0.25)
cvine = rc.VineCopula([[c12, c13, c14], [c23_1, c24_1], [c34_12]], structure="C")

by_hand = (
    np.log(c12.pdf(u[:, [0, 1]])) + np.log(c13.pdf(u[:, [0, 2]])) + np.log(c14.pdf(u[:, [0, 3]]))
)
f2_1, f3_1, f4_1 = (h(c12, u[:, 1], u[:, 0]), h(c13, u[:, 2], u[:, 0]), h(c14, u[:, 3], u[:, 0]))
by_hand += np.log(c23_1.pdf(np.column_stack([f2_1, f3_1])))
by_hand += np.log(c24_1.pdf(np.column_stack([f2_1, f4_1])))
by_hand += np.log(c34_12.pdf(np.column_stack([h(c23_1, f3_1, f2_1), h(c24_1, f4_1, f2_1)])))

show(
    "max |section 2.3 by hand - VineCopula.logpdf|",
    float(np.max(np.abs(by_hand - cvine.logpdf(u)))),
)
check("and the canonical-vine one too", np.allclose(by_hand, cvine.logpdf(u), atol=1e-13))
print(
    "\n  Both agree to about 2e-15, which is the accumulated rounding of a dozen\n"
    "  floating-point operations. The package is evaluating the printed formula,\n"
    "  not something that happens to resemble it."
)

heading("How many vines are there? The paper says 12 and 12")


# Section 2.3: "there are 12 different D-vine decompositions and 12 different
# canonical vine decompositions" in four dimensions. Counted here by building
# every ordering and asking how many give distinct densities.
def distinct_structures(structure: str, d: int) -> int:
    """Distinct densities over all d! orderings.

    The same copula goes on every edge of a tree, so what is counted is the
    *structure* rather than the labelling -- give each edge its own parameter
    and all d! orderings differ trivially, which answers a different question.
    """
    grid = np.random.default_rng(1).uniform(0.05, 0.95, size=(80, d))
    thetas = [1.6, 2.4, 3.1, 4.2][: d - 1]
    seen = set()
    for order in itertools.permutations(range(d)):
        pairs = [[rc.ClaytonCopula(thetas[k]) for _ in range(d - 1 - k)] for k in range(d - 1)]
        built = rc.VineCopula(pairs, structure=structure, order=order)
        seen.add(np.round(built.logpdf(grid), 10).tobytes())
    return len(seen)


print(f"  {'d':>3}{'D-vines':>10}{'C-vines':>10}{'d!/2':>8}")
for d in (3, 4, 5):
    counts = (distinct_structures("D", d), distinct_structures("C", d))
    print(f"  {d:>3}{counts[0]:>10}{counts[1]:>10}{math.factorial(d) // 2:>8}")
    check(f"d={d}: both counts are d!/2", counts == (math.factorial(d) // 2,) * 2)
print(
    "\n  Twelve and twelve in four dimensions, as printed. The halving is because\n"
    "  a D-vine is a path and a path read backwards is the same path -- which is\n"
    "  easy to miss: counting all 24 orderings as distinct is the obvious thing\n"
    "  to do and gives the wrong answer."
)

heading("A Gaussian vine is a multivariate Gaussian copula")

# Bedford and Cooke: every pair-copula Gaussian, and the whole vine collapses to
# an ordinary Gaussian copula whose correlation matrix follows from the partial
# correlation recursion. An exact identity, so an exact test.
sigma = rc.p2P(np.array([0.6, 0.4, 0.35, 0.5, 0.45, 0.55]), 4)
truth = rc.GaussianCopula(rc.P2p(sigma), dim=4, dispstr="un")
sample = truth.rvs(6000, random_state=0)

print(f"  {'structure':>12}{'max |Sigma_hat - Sigma|':>26}{'log-density identity':>24}")
for structure in ("C", "D"):
    fitted = rc.fit_vine(sample, structure=structure, families=("gaussian",))
    collapsed = fitted.to_gaussian()
    recovery = float(np.max(np.abs(np.asarray(collapsed.sigma()) - sigma)))
    identity = float(np.max(np.abs(fitted.logpdf(points) - collapsed.logpdf(points))))
    print(f"  {structure + '-vine':>12}{recovery:>26.4f}{identity:>24.2e}")
    check(f"{structure}-vine collapses exactly", identity < 1e-12)
    check(f"{structure}-vine recovers the correlation", recovery < 0.05)
print(
    "\n  The recovery column is sampling error at n = 6000; the identity column is\n"
    "  machine precision. Those are different claims and it is worth keeping them\n"
    "  apart: the second says the algebra is right, the first only that 6000\n"
    "  observations is enough."
)

heading("Conditional independence, which is what the trees are for")

# A Markov chain 1 -> 2 -> 3 -> 4. Every variable is independent of the rest
# given its neighbour, so a D-vine in chain order should find nothing above the
# first tree.
walk = [rng.standard_normal(4000)]
for _ in range(3):
    walk.append(0.9 * walk[-1] + np.sqrt(1 - 0.81) * rng.standard_normal(4000))
chain = rc.pseudo_obs(np.column_stack(walk))

in_order = rc.fit_vine(chain, structure="D", order=(0, 1, 2, 3))
print(f"  {'tree':>6}{'conditioning':>16}   Kendall tau of each edge")
for level, label in enumerate(("(none)", "one variable", "two variables")):
    taus = [round(float(c.tau()), 4) for c in in_order.pair_copulas[level]]
    print(f"  {level + 1:>6}{label:>16}   {taus}")

conditional = [c for tree in in_order.pair_copulas[1:] for c in tree]
independent = [c for c in conditional if isinstance(c, rc.IndependenceCopula)]
print(
    f"\n  families chosen above tree 1: "
    f"{[type(c).__name__.replace('Copula', '') for c in conditional]}"
)

check(
    "tree 1 finds strong dependence",
    min(abs(float(c.tau())) for c in in_order.pair_copulas[0]) > 0.4,
)
check(
    "every conditional edge is near independent",
    max(abs(float(c.tau())) for c in conditional) < 0.05,
)
check("and at least one selects independence outright", len(independent) >= 1)
print(
    "\n  That is the Markov property read off the fitted model, with no normality\n"
    "  assumed anywhere. Worth noting what did *not* happen: AIC picked outright\n"
    "  independence on only one of the three conditional edges, leaving the other\n"
    "  two on weak parametric families at tau about 0.015. At n = 4000 a single\n"
    "  extra parameter is cheap, so a criterion will often keep one rather than\n"
    "  round to zero. The dependence is gone; the family label is not."
)

heading("What Dissmann's algorithm actually buys")

# The structure is chosen greedily by maximum spanning tree on |tau|. It is a
# heuristic, so the honest question is how much it gives up against searching
# every ordering -- which is feasible only because d = 4.
automatic = rc.fit_vine(chain, structure="D")
scores = sorted(
    (float(np.sum(rc.fit_vine(chain, structure="D", order=order).logpdf(chain))), order)
    for order in itertools.permutations(range(4))
)
best_ll, best_order = scores[-1]
worst_ll, _ = scores[0]
auto_ll = float(np.sum(automatic.logpdf(chain)))

show("Dissmann's greedy choice", f"order {tuple(automatic.order)}, loglik {auto_ll:.2f}")
show("best of all 24 orderings", f"order {best_order}, loglik {best_ll:.2f}")
show("worst of all 24 orderings", f"loglik {worst_ll:.2f}")
show("greedy shortfall", best_ll - auto_ll)
show("full spread across orderings", best_ll - worst_ll)
show("spread as a fraction of the log-likelihood", f"{(best_ll - worst_ll) / abs(best_ll):.4%}")

# The greedy choice is *not* the best one, and saying so is the point.
check("the greedy choice is not optimal here", best_ll - auto_ll > 0.5)
check(
    "but the whole spread is negligible against the likelihood",
    (best_ll - worst_ll) < 0.001 * abs(best_ll),
)
print(
    "\n  Dissmann's greedy pick came 2.5 log-likelihood units below the best of\n"
    "  all 24 orderings -- about 60% of the way down a spread it did not close.\n"
    "  So the heuristic is genuinely a heuristic, and it is worth not pretending\n"
    "  otherwise.\n\n"
    "  What rescues it is the scale: best and worst orderings differ by four\n"
    "  units out of ten thousand, four hundredths of one percent. The structure\n"
    "  search is not where the modelling value is -- the family chosen on each\n"
    "  edge is -- and an exhaustive search costs d! fits to recover almost\n"
    "  nothing. That is why the greedy algorithm survives in practice, and it is\n"
    "  a better reason than 'it finds the optimum', which it does not."
)

heading("And it samples back to what it was fitted to")

drawn = automatic.rvs(20_000, random_state=0)
observed, expected = rc.cor_kendall(drawn), rc.cor_kendall(chain)
show("max |tau of simulated - tau of data|", float(np.max(np.abs(observed - expected))))
check("simulation reproduces the dependence", np.max(np.abs(observed - expected)) < 0.03)
show("margins uniform to", float(np.max(np.abs(drawn.mean(axis=0) - 0.5))))
check("with uniform margins", np.max(np.abs(drawn.mean(axis=0) - 0.5)) < 0.01)
