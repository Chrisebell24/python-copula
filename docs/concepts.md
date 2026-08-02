# What copulas are for

**Sklar's theorem.** Every joint distribution splits into its margins and a
copula, and any margins and any copula recombine into a valid joint
distribution:

$$H(x_1, \dots, x_d) = C\bigl(F_1(x_1), \dots, F_d(x_d)\bigr).$$

That is the whole practical appeal. Fit each margin however suits it — a fitted
Gamma for claim sizes, a Student-t for returns, a kernel estimate for something
awkward — and choose the dependence structure separately.

```python
import rcopula as rc
from scipy import stats

joint = rc.CopulaDistribution(
    rc.ClaytonCopula(2.0),
    margins=[stats.norm(loc=1, scale=2), stats.expon(scale=1 / 3)],
)
x = joint.rvs(4000, random_state=0)
```

Going the other way needs no knowledge of the margins at all — only ranks:

```python
u = rc.pseudo_obs(x)
rc.fit(rc.ClaytonCopula(), u, method="mpl")
```

## Why rank correlation, not Pearson

Rank measures are **invariant under increasing transforms of the margins**.
Squaring one variable and exponentiating another leaves Kendall's τ unchanged to
machine precision, and moves Pearson's correlation substantially. Since the
choice of marginal scale is usually arbitrary, a dependence measure that
responds to it is measuring the wrong thing.

The narrower lesson matters too: **no scalar summarises dependence**. For
$y = x^2$ with symmetric $x$, the two are perfectly dependent — knowing $x$
determines $y$ — and Pearson, Kendall and Spearman all report approximately
zero, because the relationship is not monotone.

## Tail dependence

$$\lambda_L = \lim_{u \downarrow 0}\frac{C(u,u)}{u},\qquad
  \lambda_U = \lim_{u \uparrow 1}\frac{1 - 2u + C(u,u)}{1-u}.$$

A non-zero value means joint extremes occur with a probability that does *not*
vanish relative to marginal extremes. This is the property that separates
families calibrated to the same τ, and it is the property that decides risk
numbers.

!!! warning "The Gaussian copula's zero is not reassuring"

    Its tail dependence is exactly zero, and it reaches that limit
    *logarithmically*. At τ = 0.5 the tail concentration is still **0.27 at the
    1-in-100 level and 0.10 at 1-in-10,000**, while a Student-t at the same τ
    has settled to its limit by 1-in-100. "No tail dependence" describes
    behaviour at quantiles nobody observes; at the levels a risk report uses,
    the Gaussian copula looks substantially tail-dependent.

    `rc.plots.tail_concentration(u, [c1, c2])` draws exactly this.

## The Kendall distribution function

For a copula $C$ and $\mathbf U \sim C$, $K(t) = P(C(\mathbf U) \le t)$ is the
univariate summary of a multivariate dependence structure. It determines
Kendall's τ through $\tau = 3 - 4\int_0^1 K$, and it answers a question a joint
quantile cannot: *how often is an event this severe*, when "severe" is a region
rather than a point.

The **Kendall return period** $1/(1 - K(t))$ is the multivariate analogue of a
return period, and it is never shorter than the univariate one.
