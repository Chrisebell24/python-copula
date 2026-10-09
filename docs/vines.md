# Vine copulas: a tutorial, checked against the source

An Archimedean copula in $d$ dimensions has one parameter, so it says every pair
of variables is equally dependent. In 100 dimensions that is one number
describing 4,950 pairs, and
[`examples/22`](https://github.com/Chrisebell24/python-copula/blob/main/examples/22_archimedean_internals.py)
ends by measuring exactly that ceiling.

A **vine** lifts it. The joint density is factored into $d(d-1)/2$ *pair*
copulas, each free to be a different family with a different parameter, arranged
in a nested sequence of trees. Nothing is assumed to be Gaussian, nothing is
assumed to be exchangeable, and the dimension no longer constrains the shape.

This page reproduces the results of the paper that introduced the construction —
Aas, Czado, Frigessi and Bakken (2009) — rather than asserting agreement with
it. Everything below is executable:
[`examples/26_vine_tutorial.py`](https://github.com/Chrisebell24/python-copula/blob/main/examples/26_vine_tutorial.py)
runs it all and asserts every number.

!!! note "What is not reproduced, and why"
    The paper's headline figure is an AIC of **−665.08** for a four-dimensional
    D-vine on Norwegian financial returns. That series is not distributed with
    the paper, so no implementation can reproduce the number, and claiming to
    would be worthless. The same applies to R `VineCopula`'s `daxreturns`
    example: the dataset is GPL-3 and this package is MIT, so it is not
    vendored — see [How parity is verified](parity.md).

    The identities below need **no data at all**, which is precisely what makes
    them worth checking.

## The h-function is what the paper defines

Everything in a vine is built from one object. Aas et al. equation (7):

$$h(x, v, \Theta) \;=\; F(x \mid v) \;=\; \frac{\partial C_{x,v}(x, v, \Theta)}{\partial v}$$

with the **conditioning variable second**. Checked against a central difference
of the CDF, which is the definition and nothing more:

| family | max &#124;h − ∂C/∂v&#124; |
|---|---|
| Clayton | 9.1e-11 |
| Gumbel | 8.1e-11 |
| Gaussian | 1.5e-10 |
| Frank | 1.4e-10 |
| Student | 1.2e-10 |

```python
from rcopula import conditional_cdf

h = conditional_cdf(copula, np.column_stack([x, v]), 1)  # 1 = condition on v
```

## The factorisation, written out by hand

Aas et al. equation (11) gives the four-dimensional D-vine density as

$$f = c_{12}\,c_{23}\,c_{34}\;c_{13|2}\,c_{24|3}\;c_{14|23}$$

with each conditional argument produced by an h-function. Typing that formula
out and comparing it to `VineCopula.logpdf` is a real check on whether the
package computes the published equation or merely something resembling it:

```python
by_hand = (
    np.log(c12.pdf(u[:, [0, 1]])) + np.log(c23.pdf(u[:, [1, 2]])) + np.log(c34.pdf(u[:, [2, 3]]))
)
f1_2, f3_2 = h(c12, u[:, 0], u[:, 1]), h(c23, u[:, 2], u[:, 1])
f2_3, f4_3 = h(c23, u[:, 1], u[:, 2]), h(c34, u[:, 3], u[:, 2])
by_hand += np.log(c13_2.pdf(np.column_stack([f1_2, f3_2])))
by_hand += np.log(c24_3.pdf(np.column_stack([f2_3, f4_3])))
by_hand += np.log(c14_23.pdf(np.column_stack([h(c13_2, f1_2, f3_2), h(c24_3, f4_3, f2_3)])))

vine = rc.VineCopula([[c12, c23, c34], [c13_2, c24_3], [c14_23]], structure="D")
np.max(np.abs(by_hand - vine.logpdf(u)))  # 1.8e-15
```

The canonical-vine factorisation from the paper's section 2.3 agrees to the same
**1.8e-15**, which is the accumulated rounding of a dozen floating-point
operations and nothing else.

## How many vines are there?

Section 2.3 states that in four dimensions there are *"12 different D-vine
decompositions and 12 different canonical vine decompositions"*. Counted by
building every ordering and asking how many give distinct densities:

| $d$ | D-vines | C-vines | $d!/2$ |
|---|---|---|---|
| 3 | 3 | 3 | 3 |
| 4 | **12** | **12** | 12 |
| 5 | 60 | 60 | 60 |

The halving is easy to get wrong. A D-vine is a *path*, and a path read
backwards is the same path — so all $d!$ orderings collapse in pairs. Counting
every ordering as distinct is the obvious thing to do and gives 24, not 12. The
count above only comes out right if the same copula is placed on every edge of a
tree, so that what is being counted is the structure rather than the labelling.

## A Gaussian vine is a multivariate Gaussian copula

Make every pair-copula Gaussian and the whole construction collapses to an
ordinary Gaussian copula, whose correlation matrix follows from the
partial-correlation recursion (Bedford and Cooke). That is an exact identity, so
it admits an exact test:

| structure | max &#124;Σ̂ − Σ&#124; | log-density identity |
|---|---|---|
| C-vine | 0.0179 | 5.8e-15 |
| D-vine | 0.0184 | 3.6e-15 |

```python
fitted = rc.fit_vine(sample, structure="D", families=("gaussian",))
fitted.is_gaussian  # True
fitted.to_gaussian()  # the equivalent GaussianCopula
```

Those two columns are different claims and worth keeping apart: the second says
the algebra is right, the first only that 6,000 observations is enough.

## Conditional independence, which is what the trees are for

For a Markov chain $1 \to 2 \to 3 \to 4$, every variable is independent of the
rest given its neighbour. A D-vine in chain order should therefore find strong
dependence in the first tree and nothing above it:

| tree | conditioning | Kendall's τ per edge |
|---|---|---|
| 1 | — | 0.716, 0.720, 0.719 |
| 2 | one variable | 0.000, 0.015 |
| 3 | two variables | 0.015 |

The Markov property, read straight off the fitted model, with no normality
assumed anywhere — each edge was free to choose among six families.

!!! warning "What did *not* happen"
    AIC selected outright independence on only **one** of the three conditional
    edges; the other two kept weak parametric families at τ ≈ 0.015. At
    $n = 4000$ one extra parameter is cheap, so a criterion will often retain a
    family rather than round to zero. The dependence is gone; the family label
    is not. Read the τ values, not the family names.

## What Dissmann's algorithm actually buys

The structure is chosen greedily, by maximum spanning tree on |τ| at each level
(Dissmann et al. 2013). With $d = 4$ an exhaustive search over all 24 orderings
is feasible, so the heuristic can be graded:

```
Dissmann's greedy choice   order (1, 2, 3, 0), loglik 10162.65
best of all 24 orderings   order (0, 2, 3, 1), loglik 10165.17
worst of all 24 orderings                     loglik 10161.18
```

The greedy pick came **2.5 units below the best** — about 60% of the way down a
spread it did not close. It is genuinely a heuristic and it is worth not
pretending otherwise.

What rescues it is the scale. Best and worst differ by 4.0 log-likelihood units
out of 10,163: **0.04%**. The structure search is not where the modelling value
sits — the family chosen on each edge is — and an exhaustive search costs $d!$
fits to recover almost nothing. That is a better argument for the greedy
algorithm than "it finds the optimum", which it does not.

## Regular vines: any tree shape

A C-vine forces every tree to be a star and a D-vine forces a path. Real
dependence is rarely either. In a **regular vine** (R-vine) the first tree can
be any tree on the variables, and each later tree any tree on the edges of the
one before, with one rule — the *proximity condition*: two edges may be joined
only if they share a variable. C- and D-vines are the two extreme cases.

An R-vine is written down as a small lower-triangular matrix, in the same
convention as R's `VineCopula` package, except that the labels start at 0.
Read column `i` from the bottom up: it lists the variables that the diagonal
entry is paired with in tree 1, tree 2, and so on. The pair-copula for column
`i` in tree `k` is `pair_copulas[k][i]`.

```python
import numpy as np
import rcopula as rc

# Variable 0 is a hub joined to 1, 2 and 3; variable 4 hangs off 3.
matrix = np.array(
    [[4, 0, 0, 0, 0], [1, 1, 0, 0, 0], [2, 2, 2, 0, 0], [0, 3, 3, 3, 0], [3, 0, 0, 0, 0]]
)
truth = rc.VineCopula(
    [
        [
            rc.ClaytonCopula(2.0),  # 3,4
            rc.RotatedCopula(rc.GumbelCopula(1.8), 270),  # 0,1: negative
            rc.BB1Copula(0.6, 1.5),  # 0,2: both tails
            rc.StudentCopula(0.7, df=5.0),  # 0,3
        ],
        [rc.GaussianCopula(0.3), rc.FrankCopula(-3.0), rc.FrankCopula(2.0)],
        [rc.IndependenceCopula(2)] * 2,
        [rc.IndependenceCopula(2)],
    ],
    structure="R",
    matrix=matrix,
)
truth.truncation_level  # 2: trees 3 and 4 carry no dependence
u = truth.rvs(3000, random_state=0)

fitted = rc.fit_vine(u, structure="R", families=rc.EXTENDED_FAMILIES, truncate=2)
print(fitted.describe())
```

```
R-vine copula, dim 5, order [3, 0, 4, 2, 1]
  tree 1  3,1            270-degree rotated BB1 copula, dim 2, theta=0.185676, delta=1.66304
  tree 1  0,2            BB7 copula, dim 2, theta=1.67767, delta=1.11867
  tree 1  3,4            Clayton copula, dim 2, theta=1.93415
  tree 1  3,0            Student copula, dim 2, rho=0.684017, df=4.30972
  tree 2  0,1|3          270-degree rotated Gumbel copula, dim 2, theta=1.26609
  tree 2  3,2|0          Frank copula, dim 2, theta=1.9456
  tree 2  0,4|3          Gaussian copula, dim 2, rho=0.296769
  tree 3  2,1|0,3        Independence copula, dim 2
  ...
```

`fit_vine(..., structure="R")` chooses the trees with Dissmann's algorithm: tree
1 is the tree that links all the variables with the largest total |Kendall's
τ|, and each later tree does the same on the transformed data, among the pairs
the proximity condition allows. Every edge then gets its own family.
`EXTENDED_FAMILIES` adds Joe, the 90/180/270-degree rotations and the
two-parameter BB1 and BB7 to the defaults. The rotations are how a family that
only does positive dependence (Gumbel, BB1) can fit a negative pair.

Three of the four true tree-1 edges come back. The fourth shows the greedy
step at work: in this sample |τ| is 0.464 for the pair 1–3 and 0.433 for the
true edge 0–1, so the algorithm links 1 to 3 and picks up 0–1 in tree 2. The
fitted log-likelihood is 4666.3 against the truth's 4750.8. A C-vine fitted to
the same data gets 4605.1 and a D-vine 3700.8, because neither can draw the
true shape.

Two checks pin the R-vine code down. An all-Gaussian R-vine is exactly the
Gaussian copula with the implied correlation matrix (`to_gaussian()` works for
any structure). A C- or D-vine rewritten as its R-vine matrix (`to_rvine()`,
or `structure="R", matrix=vine.matrix`) gives the same log-density to 1e-12.
Against R `VineCopula` 2.6.1, the density, log-likelihood and Rosenblatt
transform of a five-variable mixed R-vine agree to 1e-8. Dissmann selection on
600 R-simulated rows picks the same matrix and the same 10 families as R's
`RVineStructureSelect`, with log-likelihood 1183.5 against R's 1184.8 (the two
use different optimisers).

Sampling uses the exact inverse h-function wherever one exists: Gaussian,
Student t, Clayton, Frank, all their rotations, and BB1/BB7 through a Newton
solve. Only Gumbel and Joe still fall back to bisection. A ten-variable full
vine draws 20,000 rows 15 to 40 times faster than rcopula 0.4.0. Seeded draws
agree with 0.4.0 to about 1e-13, except where 0.4.0's h-function was a
numerical derivative, which is about 1e-7.

## Using it

```python
import rcopula as rc

vine = rc.fit_vine(u, structure="D")  # Dissmann selection, AIC per edge
vine.logpdf(u)  # density
vine.rvs(10_000, random_state=0)  # simulate
vine.rosenblatt(u)  # to independent uniforms
rc.plots.vine_trees(vine)  # draw the trees
rc.marginal_copula  # ...and see api/structural
```

Simulating from a fitted vine reproduces what it was fitted to — pairwise τ to
within 0.007 at n = 20,000, margins uniform to 7e-4.

For choosing *which* variables to put in a vine in the first place, see
[`rcopula.statarb.select_partners`](api/applications.md), which implements the
four partner-selection approaches of Stübinger, Mangold and Krauss (2018).

## References

- Aas, K., Czado, C., Frigessi, A. and Bakken, H. (2009). Pair-copula
  constructions of multiple dependence. *Insurance: Mathematics and Economics*
  44(2), 182–198.
- Bedford, T. and Cooke, R. M. (2002). Vines — a new graphical model for
  dependent random variables. *Annals of Statistics* 30(4), 1031–1068.
- Dissmann, J., Brechmann, E. C., Czado, C. and Kurowicka, D. (2013). Selecting
  and estimating regular vine copulae and application to financial returns.
  *Computational Statistics and Data Analysis* 59, 52–69.
- Joe, H. (1996). Families of $m$-variate distributions with given margins and
  $m(m-1)/2$ bivariate dependence parameters. In *Distributions with Fixed
  Marginals*, IMS Lecture Notes 28, 120–141.
