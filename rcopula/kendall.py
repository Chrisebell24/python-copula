r"""The Kendall distribution function.

For a copula :math:`C` and :math:`\mathbf U \sim C`, the **Kendall distribution
function** is the law of the copula evaluated at its own draw:

.. math::  K(t) = P\bigl(C(\mathbf U) \le t\bigr).

It is the univariate summary of a multivariate dependence structure: a single
curve on :math:`[0, 1]` that determines Kendall's tau
(:math:`\tau = 3 - 4\int_0^1 K`), identifies the generator of an Archimedean
copula uniquely, and answers the question a joint quantile cannot.

That last point is what makes it useful rather than merely elegant. Ask *how
often is a flood this bad*, and there is no single answer from the margins: the
set :math:`\{C(\mathbf u) > t\}` is a region, not a point, and every point on its
boundary is "equally extreme". :math:`K` measures that region. The **Kendall
return period** :math:`1/(1 - K(t))` is therefore the honest multivariate
analogue of a return period, and it is systematically longer than the
univariate one -- treating a compound event as if only its worst component
mattered understates how rare it is.

=============================  ===============================================
:func:`kendall_cdf`            :math:`K(t)`; R's ``pK``.
:func:`kendall_pdf`            Its density; R's ``dK``.
:func:`kendall_ppf`            Its quantile function; R's ``qK``.
:func:`kendall_rvs`            Draws of :math:`C(\mathbf U)`; R's ``rK``.
:func:`kendall_empirical`      The nonparametric estimator; R's ``Kn``.
:func:`kendall_return_period`  :math:`1/(1 - K(t))`, and its inverse.
=============================  ===============================================

For Archimedean copulas everything is closed form in any dimension. For other
bivariate copulas :math:`K` is obtained by integrating the conditional
distribution along the level curve of :math:`C`, which needs only the CDF.

References
----------
Genest, C. and Rivest, L.-P. (1993). Statistical inference procedures for
    bivariate Archimedean copulas. *JASA* 88(423), 1034-1043.
    The bivariate form and the nonparametric estimator.
Barbe, P., Genest, C., Ghoudi, K. and Remillard, B. (1996). On Kendall's
    process. *Journal of Multivariate Analysis* 58(2), 197-229.
    The multivariate generalisation used here.
Salvadori, G. and De Michele, C. (2004). Frequency analysis via copulas.
    *Water Resources Research* 40, W12511.
    Kendall return periods, and why the univariate one is the wrong number.
Nappo, G. and Spizzichino, F. (2009). Kendall distributions and level sets in
    bivariate exchangeable survival models.
    *Information Sciences* 179(17), 2878-2890.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaln

from rcopula.core.archimedean import ArchimedeanCopula
from rcopula.core.base import Copula
from rcopula.core.other import FrechetUpperCopula, IndependenceCopula

__all__ = [
    "kendall_cdf",
    "kendall_empirical",
    "kendall_pdf",
    "kendall_ppf",
    "kendall_return_period",
    "kendall_rvs",
    "return_period_level",
]

#: Gauss-Legendre nodes for the generic (non-Archimedean) route. The integrand
#: is a conditional distribution function, bounded in [0, 1] and smooth.
_NODES = 256

#: Bisection steps when inverting either the copula along a level curve or K
#: itself. 60 halvings take a unit bracket below 1e-18.
_BISECTIONS = 60


def _archimedean_terms(
    copula: ArchimedeanCopula, t: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """``s = psi^{-1}(t)`` and the generator's parameter, for the closed forms."""
    theta = float(copula.theta)
    return np.asarray(copula.generator.ipsi(t, theta)), theta


def kendall_cdf(copula: Copula, t: ArrayLike) -> NDArray[np.float64]:
    r"""The Kendall distribution function :math:`K(t) = P(C(\mathbf U) \le t)`.

    For an Archimedean copula in :math:`d` dimensions (Barbe et al. 1996),

    .. math::
        K(t) = \sum_{k=0}^{d-1} \frac{(-1)^k}{k!}\,
               s^k\,\psi^{(k)}(s), \qquad s = \psi^{-1}(t),

    which is a sum of **non-negative** terms once the alternating sign is
    absorbed into :math:`|\psi^{(k)}|`, so it never cancels. In two dimensions it
    reduces to the familiar :math:`K(t) = t - s\,\psi'(s)`.

    For any other bivariate copula, integrating along the level curve gives

    .. math::
        K(t) = t + \int_t^1 \frac{\partial C}{\partial u}\bigl(u, v_t(u)\bigr)\,du,

    where :math:`v_t(u)` solves :math:`C(u, v_t) = t`. Below :math:`u = t` the
    event is certain, which is where the leading :math:`t` comes from.

    Examples
    --------
    Independence has the closed form :math:`K(t) = t(1 - \log t)`:

    >>> import numpy as np
    >>> from rcopula import IndependenceCopula
    >>> from rcopula.kendall import kendall_cdf
    >>> t = np.array([0.1, 0.3, 0.7])
    >>> bool(np.allclose(kendall_cdf(IndependenceCopula(2), t), t * (1 - np.log(t))))
    True

    Comonotonicity leaves nothing to summarise -- :math:`C(\mathbf U) = U`, so
    :math:`K` is uniform:

    >>> from rcopula import FrechetUpperCopula
    >>> bool(np.allclose(kendall_cdf(FrechetUpperCopula(2), t), t))
    True

    ``K`` dominates the uniform for every copula, which is why the Kendall
    return period always exceeds the univariate one:

    >>> from rcopula import ClaytonCopula
    >>> bool(np.all(kendall_cdf(ClaytonCopula(3.0), t) >= t))
    True

    It recovers Kendall's tau through :math:`\tau = 3 - 4\int_0^1 K`:

    >>> from scipy import integrate
    >>> cop = ClaytonCopula(3.0)
    >>> area = integrate.quad(lambda x: kendall_cdf(cop, x)[0], 0, 1)[0]
    >>> bool(abs((3 - 4 * area) - cop.tau()) < 1e-6)
    True
    """
    grid = np.atleast_1d(np.asarray(t, dtype=np.float64))
    out = np.empty_like(grid)
    out[grid <= 0.0] = 0.0
    out[grid >= 1.0] = 1.0
    inside = (grid > 0.0) & (grid < 1.0)
    if not inside.any():
        return out

    x = grid[inside]

    # Two families have K in closed form in every dimension without being
    # instances of ArchimedeanCopula. Independence is Archimedean in substance
    # (psi(t) = e^{-t}) and its K is a Poisson tail; comonotonicity makes
    # C(U) = min_j U_j = U_1, so K is uniform and there is nothing to summarise.
    if isinstance(copula, IndependenceCopula):
        order = np.arange(copula.dim)
        terms = np.exp(order * np.log(-np.log(x))[:, None] - gammaln(order + 1.0))
        out[inside] = np.clip(x * terms.sum(axis=1), 0.0, 1.0)
        return out
    if isinstance(copula, FrechetUpperCopula):
        out[inside] = x
        return out

    if isinstance(copula, ArchimedeanCopula):
        s, theta = _archimedean_terms(copula, x)
        gen = copula.generator
        with np.errstate(divide="ignore", invalid="ignore"):
            log_s = np.log(s)
            total = np.array(x)  # the k = 0 term is psi(s) = t
            for k in range(1, copula.dim):
                total = total + np.exp(
                    k * log_s - gammaln(k + 1.0) + gen.log_abs_dpsi_d(s, theta, k)
                )
        out[inside] = np.clip(np.nan_to_num(total, nan=1.0), 0.0, 1.0)
        return out

    if copula.dim != 2:
        raise NotImplementedError(
            f"the Kendall function for {copula.name} copulas is implemented for "
            "dim=2; Archimedean families support any dimension"
        )
    out[inside] = np.clip(_kendall_cdf_generic(copula, x), 0.0, 1.0)
    return out


def _kendall_cdf_generic(copula: Copula, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """``K`` by integrating the conditional distribution along the level curve."""
    from rcopula.transforms import conditional_cdf

    nodes, weights = np.polynomial.legendre.leggauss(_NODES)
    out = np.empty_like(t)
    for i, level in enumerate(t):
        # Map the rule onto (t, 1); below t the event C(u, v) <= t is certain.
        u = 0.5 * (1.0 - level) * nodes + 0.5 * (1.0 + level)
        w = 0.5 * (1.0 - level) * weights

        # v_t(u) solves C(u, v) = t, monotone in v, so bisection is safe.
        lo = np.full(u.shape, 1e-15)
        hi = np.full(u.shape, 1.0)
        for _ in range(_BISECTIONS):
            mid = 0.5 * (lo + hi)
            below = copula.cdf(np.column_stack([u, mid])) < level
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        v = 0.5 * (lo + hi)

        # dC/du at (u, v_t(u)) -- so the point is (u, v) and the conditioning
        # coordinate is the first. Writing it the other way round is invisible
        # for an exchangeable copula and silently wrong for an asymmetric one,
        # which is how a 0.013 error in Marshall-Olkin's K got in.
        out[i] = level + float(np.sum(w * conditional_cdf(copula, np.column_stack([u, v]), 0)))
    return out


def kendall_pdf(copula: Copula, t: ArrayLike) -> NDArray[np.float64]:
    r"""Density of the Kendall distribution function (R's ``dK``).

    Differentiating the Archimedean sum telescopes -- every term cancels against
    the next -- and only the last survives:

    .. math::
        K'(t) = \frac{s^{d-1}\,\bigl|\psi^{(d)}(s)\bigr|}
                     {(d-1)!\,\bigl|\psi'(s)\bigr|},
        \qquad s = \psi^{-1}(t),

    manifestly non-negative, as a density must be. Elsewhere it is a central
    difference of :func:`kendall_cdf`.

    Examples
    --------
    Integrates to one, and matches the derivative of ``K``:

    >>> import numpy as np
    >>> from scipy import integrate
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.kendall import kendall_cdf, kendall_pdf
    >>> cop = ClaytonCopula(3.0)
    >>> total = integrate.quad(lambda x: kendall_pdf(cop, x)[0], 0, 1)[0]
    >>> bool(abs(total - 1.0) < 1e-8)
    True
    >>> h = 1e-6
    >>> slope = (kendall_cdf(cop, 0.4 + h)[0] - kendall_cdf(cop, 0.4 - h)[0]) / (2 * h)
    >>> bool(abs(kendall_pdf(cop, 0.4)[0] - slope) < 1e-6)
    True
    """
    grid = np.atleast_1d(np.asarray(t, dtype=np.float64))
    out = np.zeros_like(grid)
    inside = (grid > 0.0) & (grid < 1.0)
    if not inside.any():
        return out

    x = grid[inside]
    if isinstance(copula, ArchimedeanCopula):
        s, theta = _archimedean_terms(copula, x)
        gen, d = copula.generator, copula.dim
        with np.errstate(divide="ignore", invalid="ignore"):
            log_density = (
                (d - 1) * np.log(s)
                - gammaln(float(d))
                + gen.log_abs_dpsi_d(s, theta, d)
                - gen.log_abs_dpsi(s, theta)
            )
        out[inside] = np.nan_to_num(np.exp(log_density), nan=0.0, posinf=0.0)
        return out

    step = 1e-6
    hi = np.minimum(x + step, 1.0 - 1e-15)
    lo = np.maximum(x - step, 1e-15)
    out[inside] = (kendall_cdf(copula, hi) - kendall_cdf(copula, lo)) / (hi - lo)
    return np.maximum(out, 0.0)


def kendall_ppf(copula: Copula, p: ArrayLike) -> NDArray[np.float64]:
    r"""Quantile function of :math:`K` (R's ``qK``), by bisection.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import GumbelCopula
    >>> from rcopula.kendall import kendall_cdf, kendall_ppf
    >>> cop = GumbelCopula(2.5)
    >>> p = np.array([0.05, 0.5, 0.95])
    >>> bool(np.allclose(kendall_cdf(cop, kendall_ppf(cop, p)), p, atol=1e-9))
    True
    """
    probs = np.atleast_1d(np.asarray(p, dtype=np.float64))
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("p must lie in [0, 1]")

    lo = np.full(probs.shape, 1e-15)
    hi = np.full(probs.shape, 1.0 - 1e-15)
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        below = kendall_cdf(copula, mid) < probs
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.where(probs <= 0.0, 0.0, np.where(probs >= 1.0, 1.0, 0.5 * (lo + hi)))


def kendall_rvs(
    copula: Copula,
    size: int = 1,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    r"""Draw from :math:`K` (R's ``rK``), by evaluating :math:`C` at its own draws.

    Exact by construction -- :math:`K` *is* the law of :math:`C(\mathbf U)` --
    and cheaper than inverting :math:`K`.

    Examples
    --------
    The draws reproduce the analytic ``K``:

    >>> import numpy as np
    >>> from scipy import stats
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.kendall import kendall_cdf, kendall_rvs
    >>> cop = ClaytonCopula(3.0)
    >>> w = kendall_rvs(cop, 20_000, random_state=0)
    >>> p = stats.kstest(w, lambda x: kendall_cdf(cop, x)).pvalue
    >>> bool(p > 0.01)
    True
    """
    return np.asarray(copula.cdf(copula.rvs(size, random_state=random_state)))


def kendall_empirical(data: ArrayLike, t: ArrayLike | None = None) -> NDArray[np.float64]:
    r"""The nonparametric Kendall function (R's ``Kn``).

    Genest & Rivest (1993) replace :math:`C(\mathbf X_i)` by the fraction of the
    sample it dominates,

    .. math::
        W_i = \frac{1}{n-1}\,\#\{j \ne i : \mathbf X_j < \mathbf X_i\},

    and take the empirical distribution of the :math:`W_i`. Being rank-based it
    needs no margins and no fitted copula, which makes it the natural check on
    whether a family's :math:`K` matches the data -- and the basis of the
    ``Sn^K`` goodness-of-fit statistics.

    Parameters
    ----------
    data : array_like
        ``(n, d)`` observations, on any scale.
    t : array_like, optional
        Where to evaluate. Defaults to the sorted ``W_i`` themselves.

    Examples
    --------
    It tracks the true ``K`` without knowing the family:

    >>> import numpy as np
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.kendall import kendall_cdf, kendall_empirical
    >>> cop = ClaytonCopula(3.0)
    >>> x = cop.rvs(4000, random_state=0)
    >>> grid = np.array([0.1, 0.3, 0.5, 0.8])
    >>> gap = np.abs(kendall_empirical(x, grid) - kendall_cdf(cop, grid))
    >>> bool(gap.max() < 0.03)
    True
    """
    x = np.atleast_2d(np.asarray(data, dtype=np.float64))
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"need at least two observations, got {n}")

    # W_i counts strict componentwise domination; the pairwise comparison is
    # (n, n, d), so do it one column at a time and accumulate the conjunction.
    dominated = np.ones((n, n), dtype=bool)
    for j in range(x.shape[1]):
        dominated &= x[:, j][None, :] < x[:, j][:, None]
    w = dominated.sum(axis=1) / (n - 1.0)

    if t is None:
        return np.sort(w)
    grid = np.atleast_1d(np.asarray(t, dtype=np.float64))
    return np.asarray(np.mean(w[None, :] <= grid[:, None], axis=1))


def kendall_return_period(
    copula: Copula, t: ArrayLike, interval: float = 1.0
) -> NDArray[np.float64]:
    r"""Kendall return period :math:`\text{interval}/(1 - K(t))`.

    The multivariate answer to "how often is an event this severe". A univariate
    return period asks how often *one* variable is exceeded; the joint event
    :math:`\{C(\mathbf U) > t\}` is a region, and every point on its boundary is
    equally extreme, so no single margin can express it.

    Because :math:`K(t) \ge t` for every copula, the Kendall return period is
    always **longer** than the univariate one at the same level. Reporting the
    univariate number for a compound event therefore overstates how often it
    happens.

    Parameters
    ----------
    copula : Copula
        The fitted dependence structure.
    t : array_like
        Critical level, in copula units.
    interval : float
        Mean time between observations, in whatever unit the answer should be
        in -- 1 for annual maxima.

    Examples
    --------
    Comonotone variables move together exactly, so the joint event is the
    univariate one and the two return periods coincide. Everything else waits
    longer, and independence longest of all:

    >>> from rcopula import FrechetUpperCopula, IndependenceCopula
    >>> from rcopula.kendall import kendall_return_period
    >>> float(round(kendall_return_period(FrechetUpperCopula(2), 0.99)[0], 1))
    100.0
    >>> float(round(kendall_return_period(IndependenceCopula(2), 0.99)[0], 1))
    19933.2

    The number that should give a practitioner pause: **two copulas with the
    same Kendall's tau of 0.5 give 199 years and 6689 years.** Gumbel has upper
    tail dependence and Clayton has none, and the critical layer is an
    upper-corner event -- so a rank correlation, however carefully estimated,
    does not determine the design life.

    >>> from rcopula import ClaytonCopula, GumbelCopula
    >>> float(round(kendall_return_period(GumbelCopula(2.0), 0.99)[0], 1))
    199.0
    >>> float(round(kendall_return_period(ClaytonCopula(2.0), 0.99)[0], 1))
    6689.0
    """
    if interval <= 0.0:
        raise ValueError(f"interval must be positive, got {interval}")
    k = kendall_cdf(copula, t)
    with np.errstate(divide="ignore"):
        return np.asarray(interval / (1.0 - k))


def return_period_level(
    copula: Copula, period: ArrayLike, interval: float = 1.0
) -> NDArray[np.float64]:
    r"""The critical level :math:`t` whose Kendall return period is ``period``.

    The inverse of :func:`kendall_return_period`, and the quantity a design
    standard actually specifies -- "the 100-year event" is a return period, and
    what an engineer needs is the level that goes with it.

    Examples
    --------
    >>> from rcopula import GumbelCopula
    >>> from rcopula.kendall import kendall_return_period, return_period_level
    >>> cop = GumbelCopula(2.0)
    >>> t = return_period_level(cop, 100.0)
    >>> bool(abs(kendall_return_period(cop, t)[0] - 100.0) < 1e-6)
    True
    """
    years = np.atleast_1d(np.asarray(period, dtype=np.float64))
    if np.any(years <= 0.0):
        raise ValueError("period must be positive")
    return kendall_ppf(copula, 1.0 - interval / years)
