"""Nested Archimedean copulas: dependence that varies by branch.

A flat Archimedean copula has one parameter, so it says every pair of variables
is equally dependent. In 100 dimensions that is one number describing 4950
pairs, which 22_archimedean_internals.py ends by measuring.

A nested Archimedean copula is a tree. Each node carries its own generator, two
variables meet at exactly one node, and that node's parameter is their
dependence. Sectors within an index, business lines within a firm, catchments
within a basin -- the structure is usually already known, and this is how to use
it.

The cost is a genuine constraint: a parent may not be more dependent than its
child. That is not a numerical convenience, it is what keeps the result a
copula, and it is why the fitted tree below has increasing thetas going down.

References
----------
Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman & Hall.
McNeil, A. J. (2008). Sampling nested Archimedean copulas.
    *J. Statistical Computation and Simulation* 78(6), 567-581.
Hofert, M. (2011). Efficiently sampling nested Archimedean copulas.
    *Computational Statistics and Data Analysis* 55(1), 57-70.
Hofert, M. and Machler, M. (2011). Nested Archimedean copulas meet R:
    the nacopula package. *J. Statistical Software* 39(9), 1-20.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show

import rcopula as rc

heading("A tree, not a number")

# Four variables: one on its own, three in a tightly coupled block. The nesting
# condition requires the parent's theta to be no larger than any child's.
tree = rc.NestedArchimedean(
    rc.ClaytonCopula(1.0),
    components=[0],
    children=[rc.NestedArchimedean(rc.ClaytonCopula(4.0), components=[1, 2, 3])],
)
print(tree.describe())

tau = tree.tau_matrix()
print(f"\n  {'pair':>10}{'tau':>10}   which node governs it")
for i, j, label in [
    (0, 1, "root  (theta 1)"),
    (0, 3, "root  (theta 1)"),
    (1, 2, "child (theta 4)"),
    (2, 3, "child (theta 4)"),
]:
    print(f"  {f'({i},{j})':>10}{tau[i, j]:>10.4f}   {label}")

within = tau[np.ix_([1, 2, 3], [1, 2, 3])][np.triu_indices(3, 1)]
across = np.array([tau[0, j] for j in (1, 2, 3)])
check("the block is internally tight", np.all(within > 0.6))
check("and loosely tied to the outsider", np.all(across < 0.4))
check("a flat copula could not do this", float(within.min()) - float(across.max()) > 0.3)

# Two variables meet at exactly one node, so the pairwise tau is that node's own
# -- no integration anywhere.
show("\ntau of the child's generator alone", rc.ClaytonCopula(4.0).tau())
show("tau_matrix entry for the pair (1,2)", float(tau[1, 2]))
check("they are the same number", abs(tau[1, 2] - rc.ClaytonCopula(4.0).tau()) < 1e-12)

heading("The nesting condition is a real constraint")

# Parent theta <= child theta. Violate it and the construction stops being a
# copula, so it is refused rather than returned with a warning.
try:
    rc.NestedArchimedean(
        rc.ClaytonCopula(5.0),
        components=[0],
        children=[rc.NestedArchimedean(rc.ClaytonCopula(1.0), components=[1, 2])],
    )
    refused = False
except ValueError as error:
    refused = True
    show("refused, as it should be", str(error)[:68])
check("a parent more dependent than its child is rejected", refused)
print(
    "\n  The reason is not bookkeeping. Sampling works by drawing an outer frailty\n"
    "  and then an inner one conditioned on it, and the inner draw only exists as\n"
    "  a distribution when the parameters are ordered that way."
)

heading("Sampling, which needs the tilted stable law")

sample = tree.rvs(200_000, random_state=0)
observed = rc.cor_kendall(sample)
print(f"  {'pair':>10}{'sampled tau':>14}{'exact tau':>12}")
for i, j in [(0, 1), (0, 2), (1, 2), (2, 3)]:
    print(f"  {f'({i},{j})':>10}{observed[i, j]:>14.4f}{tau[i, j]:>12.4f}")
check("the sampler reproduces every pair", np.max(np.abs(observed - tau)) < 0.01)
show(
    "margins uniform, max deviation of the mean from 1/2",
    float(np.max(np.abs(sample.mean(axis=0) - 0.5))),
)
check("and the margins are uniform", np.max(np.abs(sample.mean(axis=0) - 0.5)) < 0.005)
print(
    "\n  The inner frailty is an exponentially tilted stable draw. No other Python\n"
    "  package has one, which is the reason no other Python package has nested\n"
    "  Archimedean copulas."
)

heading("Estimating it without a density")

# A nested Archimedean copula has no usable density: it would need high-order
# derivatives of a *composition* of generators, which is an open problem past
# small dimensions. The package says so rather than returning something wrong.
data = tree.rvs(2000, random_state=1)
try:
    tree.logpdf(data[:5])
    has_density = True
except NotImplementedError as error:
    has_density = False
    show("no density, and it explains why", str(error)[:66] + "...")
check("the limitation is refused, not fudged", not has_density)

# Everything else is available, so estimation and assessment go through the CDF.
# Cramer-von Mises distance to the empirical copula is the natural criterion.
print(f"\n  {'root theta':>12}{'Sn':>12}     (child held at 4.0)")


def criterion(root_theta: float, child_theta: float) -> float:
    candidate = rc.NestedArchimedean(
        rc.ClaytonCopula(root_theta),
        components=[0],
        children=[rc.NestedArchimedean(rc.ClaytonCopula(child_theta), components=[1, 2, 3])],
    )
    return float(rc.gof_statistic(data, candidate))


root_grid = [0.4, 0.7, 1.0, 1.3, 1.8]
root_values = [criterion(t, 4.0) for t in root_grid]
for theta, value in zip(root_grid, root_values, strict=True):
    print(f"  {theta:>12}{value:>12.4f}{'  <- truth' if theta == 1.0 else ''}")
check(
    "the criterion is smallest at the true root parameter",
    root_grid[int(np.argmin(root_values))] == 1.0,
)

print(f"\n  {'child theta':>12}{'Sn':>12}     (root held at 1.0)")
child_grid = [2.5, 3.2, 4.0, 5.0, 6.5]
child_values = [criterion(1.0, t) for t in child_grid]
for theta, value in zip(child_grid, child_values, strict=True):
    print(f"  {theta:>12}{value:>12.4f}{'  <- truth' if theta == 4.0 else ''}")
check("and at the true child parameter", child_grid[int(np.argmin(child_values))] == 4.0)
print(
    "\n  Both profiles are minimised at the truth, which is why fit_nested can\n"
    "  estimate a tree it cannot write a likelihood for. It inverts pairwise\n"
    "  Kendall tau node by node -- exact, because two variables meet at exactly\n"
    "  one node and that node's generator is their bivariate copula."
)

heading("Fitting the tree")

fitted = rc.fit_nested(tree, data)
print(fitted.describe())
# A tree's parameters live in its nodes, not in a flat vector -- `params` is
# empty by design, because a tree has no canonical ordering to flatten into.
root_theta = float(fitted.generator_copula.params[0])
child_theta = float(fitted.children[0].generator_copula.params[0])
show("\nroot theta: true / fitted", (1.0, round(root_theta, 4)))
show("child theta: true / fitted", (4.0, round(child_theta, 4)))
check("both recovered", abs(root_theta - 1.0) < 0.2 and abs(child_theta - 4.0) < 0.5)
check("and the fit respects the nesting condition", root_theta <= child_theta)

# What a flat copula fitted to the same data would have to say.
flat = rc.fit(rc.ClaytonCopula(1.0, dim=4), data, method="mpl").copula
wrong = rc.NestedArchimedean(
    rc.ClaytonCopula(0.4),
    components=[0],
    children=[rc.NestedArchimedean(rc.ClaytonCopula(8.0), components=[1, 2, 3])],
)
print(f"\n  {'model':34}{'Sn':>10}")
for label, model in [
    ("fitted tree", fitted),
    ("flat Clayton, best fit", flat),
    ("a tree with the wrong thetas", wrong),
]:
    print(f"  {label:34}{rc.gof_statistic(data, model):>10.4f}")

show("\nflat Clayton's single theta", float(flat.params[0]))
show("   the one tau it implies", flat.tau())
show("   but the truth ranges over", (round(float(across.min()), 3), round(float(within.max()), 3)))
check(
    "the tree fits far better than the flat copula",
    rc.gof_statistic(data, fitted) < 0.25 * rc.gof_statistic(data, flat),
)
check(
    "and the criterion still rejects a wrong tree",
    rc.gof_statistic(data, fitted) < rc.gof_statistic(data, wrong),
)
print(
    "\n  The flat copula splits the difference and describes neither group. That\n"
    "  is not a fitting failure -- it is the only thing one parameter can do, and\n"
    "  the reason to spend two on a structure you already know."
)

heading("Tail dependence also varies by branch")

lower_tail, upper_tail = tree.lambda_matrix()
print(f"  {'pair':>10}{'lambda_L':>12}{'lambda_U':>12}   governed by")
for i, j, label in [(0, 1, "root  (theta 1)"), (1, 2, "child (theta 4)")]:
    print(f"  {f'({i},{j})':>10}{lower_tail[i, j]:>12.4f}{upper_tail[i, j]:>12.4f}   {label}")
show("within the block, lower tail", float(lower_tail[1, 2]))
show("across to the outsider", float(lower_tail[0, 1]))
check("joint crashes are far likelier inside the block", lower_tail[1, 2] > 1.5 * lower_tail[0, 1])
off = np.triu_indices(4, 1)
check("and Clayton has no upper tail dependence on any pair", np.all(upper_tail[off] == 0.0))
print(
    "\n  For a credit portfolio that is the whole question: names in one sector\n"
    "  default together far more readily than names across sectors, and a single\n"
    "  correlation cannot express it."
)
