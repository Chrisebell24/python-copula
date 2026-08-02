r"""Copulas for discrete and mixed margins.

Everything else in this package assumes continuous margins, where Sklar's
theorem gives a **unique** copula and rank-based inference is exact. Neither
holds for counts, ordinal scales, or a continuous variable paired with a binary
one -- and the literature that quietly applies continuous machinery to them is
large.

**What actually breaks.** Sklar's theorem still says a copula exists, but it is
unique only on :math:`\mathrm{Ran}\,F_1 \times \cdots \times \mathrm{Ran}\,F_d`.
For a Bernoulli margin that range is three points, so all a copula can be
identified from is what happens at three points: infinitely many copulas give
exactly the same joint distribution. This is not a small-sample problem and more
data does not fix it. It means a fitted parameter is interpretable *within a
chosen family* and comparisons across families are on much weaker ground than
they look (Genest and Nešlehová 2007, which is worth reading before using any of
this).

**What still works.** The joint distribution is perfectly well defined, and so
is its likelihood -- as a finite difference of the copula rather than a
derivative of it:

.. math::

    P(X = x) = \sum_{j \in \{0,1\}^d} (-1)^{|j|}
               C\bigl(u_1^{(j_1)}, \dots, u_d^{(j_d)}\bigr),
    \qquad u_k^{(0)} = F_k(x_k), \quad u_k^{(1)} = F_k(x_k^-).

That is exact, it is what :func:`discrete_pmf` computes, and maximising it is
what :func:`fit_discrete` does. For **mixed** margins the two operations combine:
differentiate along the continuous coordinates, difference along the discrete
ones -- :func:`mixed_pdf`.

**Ranks, and why they mislead.** Ties break the correspondence between a sample
rank correlation and the copula's own. Kendall's tau-b divides out the ties
within each margin, so it still reaches 1 for comonotone *identical* margins --
but when the margins differ, no coupling can align their atoms and the ceiling
drops: two Bernoullis at 0.1 and 0.9 cannot exceed 0.111 however strongly they
are coupled. :func:`tau_upper_bound` computes it. The practical consequence is
that inverting a sample tau to get a copula parameter, which is exact for
continuous margins, is simply wrong here -- which is why :func:`fit_discrete`
offers likelihood only.

**The distributional transform** (Ferguson 1967; Rüschendorf 2009) turns a
discrete variable into an exactly uniform one by randomising within each atom.
It is the honest version of jittering: :func:`distributional_transform` gives
pseudo-observations that any continuous-margin method can consume, at the cost
of the randomisation being part of the answer. Average over several draws.

============================  ================================================
:func:`discrete_pmf`          Exact probability mass, by inclusion-exclusion.
:func:`mixed_pdf`             Density for any mix of discrete and continuous.
:func:`fit_discrete`          Maximum likelihood on the exact mass function.
:func:`distributional_transform`  Randomised pseudo-observations.
:func:`tau_upper_bound`       The largest Kendall tau these margins allow.
:func:`checkerboard`          The canonical member of the identified class.
============================  ================================================

Examples
--------
>>> import numpy as np, rcopula as rc
>>> from scipy import stats
>>> from rcopula.discrete import discrete_pmf
>>> margins = [stats.poisson(3.0), stats.poisson(2.0)]
>>> x = np.array([[3, 2], [0, 0], [5, 4]])
>>> mass = discrete_pmf(rc.GaussianCopula(0.6), x, margins)
>>> bool(np.all(mass > 0))
True

References
----------
Genest, C. and Nešlehová, J. (2007). A primer on copulas for count data.
    *ASTIN Bulletin* 37(2), 475-515.
    The paper to read first; the source of the identifiability caveat.
Rüschendorf, L. (2009). On the distributional transform, Sklar's theorem, and
    the empirical copula process. *J. Statistical Planning and Inference*
    139(11), 3921-3927.
Ferguson, T. S. (1967). *Mathematical Statistics: A Decision Theoretic
    Approach*. Academic Press. The distributional transform.
Nikoloulopoulos, A. K. (2013). Copula-based models for multivariate discrete
    response data. In *Copulae in Mathematical and Quantitative Finance*,
    231-249. The inclusion-exclusion likelihood.
Song, P. X.-K., Li, M. and Yuan, Y. (2009). Joint regression analysis of
    correlated data using Gaussian copulas. *Biometrics* 65(1), 60-68.
Denuit, M. and Lambert, P. (2005). Constraints on concordance measures in
    bivariate discrete data. *J. Multivariate Analysis* 93(1), 40-57.
    Where the attainable range of Kendall's tau comes from.
Sun, T., Song, X. and Zhang, X. (2021). scDesign2: a transparent simulator for
    single-cell RNA sequencing data. *Genome Biology* 22, 163.
    A Gaussian copula with negative-binomial margins, at scale.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from rcopula.core.base import Copula

__all__ = [
    "DiscreteFitResult",
    "DiscreteMargin",
    "checkerboard",
    "discrete_loglik",
    "discrete_pmf",
    "distributional_transform",
    "fit_discrete",
    "mixed_pdf",
    "tau_upper_bound",
]

#: Probabilities below this are treated as zero when taking logs. Machine
#: epsilon would be tighter, but the inclusion-exclusion sum genuinely cancels
#: to that order for a nearly-comonotone copula and a rare cell, and a
#: log-likelihood of -700 is already an emphatic rejection.
_MASS_FLOOR = 1e-300


@runtime_checkable
class DiscreteMargin(Protocol):
    """What a discrete margin must provide.

    Satisfied by every discrete ``scipy.stats`` frozen distribution
    (``poisson``, ``nbinom``, ``binom``, ``geom``, ``randint``, ...).
    """

    def cdf(self, x: ArrayLike) -> Any: ...
    def pmf(self, x: ArrayLike) -> Any: ...
    def ppf(self, q: ArrayLike) -> Any: ...


def _left_limit(margin: Any, x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""``F(x-)``, the CDF just below each observed value.

    For a lattice margin this is ``F(x) - P(X = x)``, which is exact and does
    not depend on knowing the lattice spacing. Subtracting the mass is also
    numerically better than evaluating ``F(x - 1)``: the two agree in exact
    arithmetic, but the difference form keeps the atom's width accurate even
    where the CDF has run into 1.
    """
    return np.asarray(margin.cdf(x) - margin.pmf(x), dtype=float)


def _corner_values(
    x: NDArray[np.float64], margins: list[Any], discrete: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(F(x), F(x-))`` for every coordinate, with the two equal where the
    margin is continuous."""
    upper = np.column_stack(
        [np.asarray(m.cdf(x[:, j]), dtype=float) for j, m in enumerate(margins)]
    )
    lower = upper.copy()
    for j in np.flatnonzero(discrete):
        lower[:, j] = _left_limit(margins[j], x[:, j])
    return np.clip(upper, 0.0, 1.0), np.clip(lower, 0.0, 1.0)


def discrete_pmf(
    copula: Copula,
    x: ArrayLike,
    margins: list[Any],
) -> NDArray[np.float64]:
    r"""Exact probability mass of a copula model with discrete margins.

    Computes the :math:`2^d`-term inclusion-exclusion sum in the module
    docstring. This is the C-volume of the rectangle
    :math:`\prod_k (F_k(x_k^-), F_k(x_k)]`, which is what a copula's
    :math:`d`-increasing property guarantees is non-negative -- so a negative
    result here means the copula is not one.

    Parameters
    ----------
    copula : Copula
    x : array_like, shape (n, d)
        Observed values, on the margins' own scale.
    margins : list of frozen discrete distributions
        One per dimension.

    Returns
    -------
    ndarray, shape (n,)

    Notes
    -----
    Cost is :math:`2^d` copula CDF evaluations, each vectorised over
    observations. Past about :math:`d = 15` that is the binding constraint and a
    composite-likelihood approach (pairs, or a vine) is the usual answer.

    Examples
    --------
    The mass function must sum to one over the whole lattice:

    >>> import numpy as np, rcopula as rc
    >>> from scipy import stats
    >>> from rcopula.discrete import discrete_pmf
    >>> margins = [stats.poisson(2.0), stats.poisson(3.0)]
    >>> grid = np.array([[i, j] for i in range(40) for j in range(45)])
    >>> total = discrete_pmf(rc.ClaytonCopula(2.0), grid, margins).sum()
    >>> bool(abs(total - 1.0) < 1e-9)
    True

    And it must reproduce the margins when summed over the other coordinate:

    >>> marginal = discrete_pmf(rc.ClaytonCopula(2.0), grid, margins).reshape(40, 45).sum(axis=1)
    >>> bool(np.allclose(marginal, margins[0].pmf(np.arange(40)), atol=1e-9))
    True
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    dim = copula.dim
    if x.shape[1] != dim:
        raise ValueError(f"x has {x.shape[1]} columns but the copula has dim {dim}")
    if len(margins) != dim:
        raise ValueError(f"expected {dim} margins, got {len(margins)}")

    upper, lower = _corner_values(x, margins, np.ones(dim, dtype=bool))
    total = np.zeros(x.shape[0], dtype=float)
    for corner in itertools.product((0, 1), repeat=dim):
        chosen = np.where(np.array(corner, dtype=bool), lower, upper)
        sign = -1.0 if sum(corner) % 2 else 1.0
        total += sign * np.asarray(copula.cdf(chosen), dtype=float)
    # A C-volume cannot be negative; anything below zero is cancellation in the
    # sum, not a real quantity, so it is floored rather than propagated.
    return np.maximum(total, 0.0)


def mixed_pdf(
    copula: Copula,
    x: ArrayLike,
    margins: list[Any],
    discrete: ArrayLike,
    *,
    step: float = 1e-5,
) -> NDArray[np.float64]:
    r"""Density for a mix of discrete and continuous margins.

    The joint density of a mixed vector is a *partial* derivative: differentiate
    the copula along the continuous coordinates and difference it along the
    discrete ones, then multiply by the continuous marginal densities.

    With one discrete coordinate this reduces to a difference of h-functions,

    .. math::

        f(x_1, x_2) = f_1(x_1)\left[
            \frac{\partial C}{\partial u_1}(u_1, F_2(x_2))
            - \frac{\partial C}{\partial u_1}(u_1, F_2(x_2^-))
        \right],

    which is the form used in the transportation and biostatistics literature
    for joining a discrete choice to a continuous response.

    Parameters
    ----------
    copula : Copula
    x : array_like, shape (n, d)
    margins : list of frozen distributions
        Discrete ones must provide ``pmf``; continuous ones ``pdf``.
    discrete : array_like of bool, shape (d,)
        Which coordinates are discrete.
    step : float
        Finite-difference step for the continuous derivatives, on the copula
        scale. Only used when the copula has no analytic conditional CDF.

    Returns
    -------
    ndarray, shape (n,)

    Notes
    -----
    With no discrete coordinates this is the ordinary copula density times the
    marginal densities, and with all of them it is :func:`discrete_pmf`; both
    limits are checked in the test suite. The continuous derivatives go through
    :func:`~rcopula.conditional_cdf`, which is analytic for most families.

    Examples
    --------
    A continuous margin paired with a Bernoulli one. The density integrates and
    sums to one:

    >>> import numpy as np, rcopula as rc
    >>> from scipy import stats
    >>> from rcopula.discrete import mixed_pdf
    >>> margins = [stats.norm(), stats.bernoulli(0.3)]
    >>> grid = np.linspace(-8, 8, 2001)
    >>> rows = np.concatenate([
    ...     np.column_stack([grid, np.zeros_like(grid)]),
    ...     np.column_stack([grid, np.ones_like(grid)]),
    ... ])
    >>> values = mixed_pdf(rc.GaussianCopula(0.5), rows, margins, [False, True])
    >>> total = np.trapezoid(values[: grid.size], grid) + np.trapezoid(values[grid.size :], grid)
    >>> bool(abs(total - 1.0) < 1e-6)
    True
    """
    from rcopula.transforms import conditional_cdf

    x = np.atleast_2d(np.asarray(x, dtype=float))
    dim = copula.dim
    discrete = np.asarray(discrete, dtype=bool)
    if discrete.shape != (dim,):
        raise ValueError(f"discrete must have {dim} entries, got {discrete.shape}")
    if x.shape[1] != dim:
        raise ValueError(f"x has {x.shape[1]} columns but the copula has dim {dim}")
    if len(margins) != dim:
        raise ValueError(f"expected {dim} margins, got {len(margins)}")

    if not discrete.any():
        u = np.column_stack([np.asarray(m.cdf(x[:, j])) for j, m in enumerate(margins)])
        density = np.asarray(copula.pdf(np.clip(u, 1e-12, 1 - 1e-12)), dtype=float)
        for j, margin in enumerate(margins):
            density = density * np.asarray(margin.pdf(x[:, j]), dtype=float)
        return density
    if discrete.all():
        return discrete_pmf(copula, x, margins)

    upper, lower = _corner_values(x, margins, discrete)
    continuous = np.flatnonzero(~discrete)
    discrete_idx = np.flatnonzero(discrete)

    if continuous.size > 1:
        raise NotImplementedError(
            "mixed_pdf differentiates along at most one continuous coordinate; "
            f"got {continuous.size}. For several, either discretise the extras "
            "or use a vine, whose pair copulas each face this problem in two "
            "dimensions where it is solved."
        )

    axis = int(continuous[0])
    total = np.zeros(x.shape[0], dtype=float)
    for corner in itertools.product((0, 1), repeat=discrete_idx.size):
        point = upper.copy()
        for position, which in zip(discrete_idx, corner, strict=True):
            if which:
                point[:, position] = lower[:, position]
        sign = -1.0 if sum(corner) % 2 else 1.0
        total += sign * np.asarray(
            conditional_cdf(copula, np.clip(point, 1e-12, 1 - 1e-12), axis), dtype=float
        )
    del step  # analytic conditional CDFs throughout; kept for API stability
    return np.maximum(total, 0.0) * np.asarray(margins[axis].pdf(x[:, axis]), dtype=float)


def discrete_loglik(copula: Copula, x: ArrayLike, margins: list[Any]) -> float:
    """Log-likelihood of ``x`` under a copula with discrete margins.

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from scipy import stats
    >>> from rcopula.discrete import discrete_loglik
    >>> margins = [stats.poisson(2.0), stats.poisson(2.0)]
    >>> x = rc.CopulaDistribution(rc.GaussianCopula(0.7), margins).rvs(300, random_state=0)
    >>> strong = discrete_loglik(rc.GaussianCopula(0.7), x, margins)
    >>> weak = discrete_loglik(rc.GaussianCopula(0.0), x, margins)
    >>> bool(strong > weak)
    True
    """
    mass = discrete_pmf(copula, x, margins)
    return float(np.sum(np.log(np.maximum(mass, _MASS_FLOOR))))


@dataclass
class DiscreteFitResult:
    """A copula fitted to discrete data by exact maximum likelihood.

    Attributes
    ----------
    copula : Copula
        At the estimated parameters.
    params : ndarray
    loglik : float
    n_obs : int
    converged : bool
    independent_loglik : float
        The same margins under independence, for a likelihood ratio.
    """

    copula: Copula
    params: NDArray[np.float64]
    loglik: float
    n_obs: int
    converged: bool
    independent_loglik: float
    message: str = ""

    @property
    def n_params(self) -> int:
        """Free parameters estimated."""
        return int(np.sum(self.copula.free))

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return float(2 * self.n_params - 2 * self.loglik)

    @property
    def bic(self) -> float:
        """Bayesian information criterion."""
        return float(self.n_params * np.log(self.n_obs) - 2 * self.loglik)

    def independence_test(self) -> tuple[float, float]:
        """Likelihood ratio against independence.

        Unlike the constancy test in :mod:`rcopula.dynamic`, this null is
        interior for every family here, so the chi-squared reference is the
        usual asymptotic one.

        Returns
        -------
        statistic, pvalue
        """
        from scipy import stats as _stats

        statistic = max(2.0 * (self.loglik - self.independent_loglik), 0.0)
        return float(statistic), float(_stats.chi2(self.n_params).sf(statistic))

    def summary(self) -> str:
        """A printable report."""
        statistic, pvalue = self.independence_test()
        lines = [
            f"{self.copula.describe()} fitted to discrete margins",
            "=" * 68,
            f"  observations         {self.n_obs}",
            "  parameters           "
            + ", ".join(
                f"{name}={value:.6f}"
                for name, value in zip(self.copula.param_names, self.params, strict=True)
            ),
            f"  log-likelihood       {self.loglik: .4f}",
            f"  AIC / BIC            {self.aic: .4f} / {self.bic:.4f}",
            "",
            f"  under independence   {self.independent_loglik: .4f}",
            f"  LR vs independence   {statistic: .4f}  (p = {pvalue:.4g})",
            "",
            "  The copula is identified only on the margins' ranges, so this",
            "  parameter is interpretable within this family and not across",
            "  families (Genest and Neslehova 2007).",
        ]
        if not self.converged:
            lines += ["", f"  WARNING: optimiser did not converge -- {self.message}"]
        return "\n".join(lines)


def fit_discrete(
    x: ArrayLike,
    copula: Copula,
    margins: list[Any],
    *,
    start: ArrayLike | None = None,
) -> DiscreteFitResult:
    """Fit a copula to discrete data by maximising the exact mass function.

    The margins are taken as given -- fit them separately, which is the
    inference-functions-for-margins two-step and is what everyone does. The
    copula parameter is then the only unknown.

    Parameters
    ----------
    x : array_like, shape (n, d)
        Observed counts or codes.
    copula : Copula
        The family to fit. Its current parameters are the starting point.
    margins : list of frozen discrete distributions
    start : array_like, optional
        Override the starting parameters.

    Returns
    -------
    DiscreteFitResult

    Notes
    -----
    Rank-based estimation (``method="itau"``) is *not* offered here on purpose:
    with ties, the sample Kendall tau does not estimate the copula's tau, and
    inverting it produces a parameter with no defensible interpretation. See
    :func:`tau_upper_bound` for the size of the distortion.

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from scipy import stats
    >>> from rcopula.discrete import fit_discrete
    >>> margins = [stats.poisson(4.0), stats.poisson(4.0)]
    >>> x = rc.CopulaDistribution(rc.GaussianCopula(0.6), margins).rvs(2000, random_state=0)
    >>> result = fit_discrete(x, rc.GaussianCopula(0.0), margins)
    >>> bool(abs(result.params[0] - 0.6) < 0.06)
    True
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    free = np.asarray(copula.free, dtype=bool)
    if not free.any():
        loglik = discrete_loglik(copula, x, margins)
        return DiscreteFitResult(
            copula=copula,
            params=np.asarray(copula.params, dtype=float),
            loglik=loglik,
            n_obs=x.shape[0],
            converged=True,
            independent_loglik=loglik,
        )

    initial = (
        np.array(copula.params, dtype=float)[free]
        if start is None
        else np.asarray(start, dtype=float)
    )
    bounds = [b for b, is_free in zip(copula.param_bounds, free, strict=True) if is_free]
    # Pull infinite bounds in to something an optimiser can work with, and keep
    # off the endpoints, where several families are degenerate.
    finite = []
    for low, high in bounds:
        span = 25.0
        low = float(low) if np.isfinite(low) else -span
        high = float(high) if np.isfinite(high) else span
        pad = 1e-6 * max(1.0, high - low)
        finite.append((low + pad, high - pad))

    def objective(theta: NDArray[np.float64]) -> float:
        # `.params` hands back a read-only view so a fitted copula cannot be
        # mutated behind its own back; take a copy before writing into it.
        params = np.array(copula.params, dtype=float)
        params[free] = theta
        try:
            candidate = copula.with_params(params)
        except (ValueError, np.linalg.LinAlgError):
            return 1e12
        value = discrete_loglik(candidate, x, margins)
        return float(-value) if np.isfinite(value) else 1e12

    result = optimize.minimize(
        objective,
        np.clip(initial, [b[0] for b in finite], [b[1] for b in finite]),
        method="L-BFGS-B",
        bounds=finite,
    )
    params = np.array(copula.params, dtype=float)
    params[free] = result.x
    fitted = copula.with_params(params)

    from rcopula.core.other import IndependenceCopula

    return DiscreteFitResult(
        copula=fitted,
        params=params,
        loglik=float(-result.fun),
        n_obs=x.shape[0],
        converged=bool(result.success),
        independent_loglik=discrete_loglik(IndependenceCopula(copula.dim), x, margins),
        message=str(result.message),
    )


def distributional_transform(
    x: ArrayLike,
    margins: list[Any],
    *,
    random_state: Any = None,
    replicates: int = 1,
) -> NDArray[np.float64]:
    r"""Turn discrete observations into exactly uniform pseudo-observations.

    The distributional transform randomises within each atom,

    .. math:: U = F(X^-) + V\,\bigl(F(X) - F(X^-)\bigr), \qquad V \sim U(0,1),

    with :math:`V` independent of :math:`X`. The result is **exactly** uniform,
    not approximately -- which is what separates this from ad-hoc jittering --
    and its copula is a copula of :math:`X`, chosen at random from the
    identified class.

    Parameters
    ----------
    x : array_like, shape (n, d)
    margins : list of frozen distributions
        Continuous margins are passed through unchanged.
    random_state : None, int or Generator
    replicates : int
        Draw this many independent transforms and return their average on the
        copula scale. Averaging reduces the randomisation noise but biases the
        result towards the middle of each atom, so the default is 1 and anything
        else is a deliberate trade.

    Returns
    -------
    ndarray, shape (n, d)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from rcopula.discrete import distributional_transform
    >>> x = stats.poisson(3.0).rvs(20000, random_state=0)[:, None]
    >>> u = distributional_transform(x, [stats.poisson(3.0)], random_state=0)
    >>> bool(abs(u.mean() - 0.5) < 0.01)          # exactly uniform, so mean 1/2
    True
    >>> bool(abs(u.var() - 1 / 12) < 0.005)
    True
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if len(margins) != x.shape[1]:
        raise ValueError(f"expected {x.shape[1]} margins, got {len(margins)}")
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    rng = np.random.default_rng(random_state)

    upper = np.column_stack(
        [np.asarray(m.cdf(x[:, j]), dtype=float) for j, m in enumerate(margins)]
    )
    lower = upper.copy()
    for j, margin in enumerate(margins):
        if hasattr(margin, "pmf"):
            lower[:, j] = _left_limit(margin, x[:, j])

    total = np.zeros_like(upper)
    for _ in range(replicates):
        v = rng.uniform(size=upper.shape)
        total += lower + v * (upper - lower)
    return np.clip(total / replicates, 1e-12, 1 - 1e-12)


def tau_upper_bound(margins: list[Any], *, support: int = 200) -> float:
    r"""The largest Kendall's tau-b these discrete margins can attain.

    Evaluates :math:`\tau_b` at the comonotone coupling, which is where it is
    maximised. Since :math:`\tau_b` divides by
    :math:`\sqrt{(1-\sum_i p_i^2)(1-\sum_j q_j^2)}` it already corrects for the
    ties *within* each margin, and so reaches 1 when the two margins are
    identical -- comonotone Poisson(1) pairs really do give
    :math:`\tau_b = 1`.

    The ceiling bites when the margins **differ**, because then no coupling can
    align their atoms. Two Bernoullis with success probabilities 0.1 and 0.9
    cannot exceed 0.111 however they are coupled, so reading a fitted 0.1 there
    as "weak dependence" inverts the truth: it is as strong as the margins
    permit.

    Parameters
    ----------
    margins : list of two frozen discrete distributions
    support : int
        How far up the lattice to evaluate. The tail beyond this contributes
        less than its mass, so the default is generous for anything with a
        finite mean.

    Returns
    -------
    float

    Examples
    --------
    Identical margins reach 1; mismatched ones do not come close:

    >>> from scipy import stats
    >>> from rcopula.discrete import tau_upper_bound
    >>> round(tau_upper_bound([stats.poisson(1.0), stats.poisson(1.0)]), 4)
    1.0
    >>> round(tau_upper_bound([stats.bernoulli(0.1), stats.bernoulli(0.9)]), 4)
    0.1111
    >>> round(tau_upper_bound([stats.poisson(3.0), stats.nbinom(4, 0.5)]), 4)
    0.9438
    """
    if len(margins) != 2:
        raise ValueError(f"tau_upper_bound is bivariate; got {len(margins)} margins")
    grid = np.arange(support + 1)
    p = np.asarray(margins[0].pmf(grid), dtype=float)
    q = np.asarray(margins[1].pmf(grid), dtype=float)

    # The maximum is attained at the comonotone coupling, whose joint CDF is the
    # Frechet upper bound min(F, G). Its mass is the second difference.
    cdf_p, cdf_q = np.cumsum(p), np.cumsum(q)
    joint = np.diff(
        np.diff(np.minimum(cdf_p[:, None], cdf_q[None, :]), axis=1, prepend=0.0),
        axis=0,
        prepend=0.0,
    )
    # Population tau_b is [P(concordant) - P(discordant)] over the tie-corrected
    # normaliser. Comonotone means no discordant pairs at all, so only the first
    # term survives:  P(concordant) = 2 sum h(i,j) P(X > i, Y > j).
    cumulative = np.cumsum(np.cumsum(joint, axis=0), axis=1)
    survivor = 1.0 - cdf_p[:, None] - cdf_q[None, :] + cumulative
    concordant = 2.0 * float(np.sum(joint * survivor))

    denominator = np.sqrt((1.0 - float(np.sum(p**2))) * (1.0 - float(np.sum(q**2))))
    if denominator <= 0:
        return 0.0  # a degenerate margin: every pair is tied, so tau is undefined
    return float(min(concordant / denominator, 1.0))


def checkerboard(copula: Copula, margins: list[Any], *, support: int = 60) -> NDArray[np.float64]:
    r"""The checkerboard copula's mass on the lattice induced by the margins.

    Since the copula is identified only on :math:`\mathrm{Ran}\,F_1 \times
    \mathrm{Ran}\,F_2`, an infinite family of copulas fits any discrete data
    equally well. The **checkerboard** member spreads each cell's mass uniformly
    over its rectangle: it is the canonical representative, it is the one the
    distributional transform targets on average, and it is the one to plot when
    a picture of "the" fitted copula is wanted.

    Parameters
    ----------
    copula : Copula
        Bivariate.
    margins : list of two frozen discrete distributions
    support : int
        Lattice extent.

    Returns
    -------
    ndarray, shape (support + 1, support + 1)
        Cell probabilities, summing to one up to the tail truncation.

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from scipy import stats
    >>> from rcopula.discrete import checkerboard
    >>> mass = checkerboard(rc.GaussianCopula(0.5), [stats.poisson(3.0)] * 2)
    >>> bool(abs(mass.sum() - 1.0) < 1e-8)
    True
    >>> bool(np.all(mass >= 0))
    True
    """
    if copula.dim != 2:
        raise ValueError(f"checkerboard is bivariate; got dim {copula.dim}")
    grid = np.arange(support + 1)
    pairs = np.array([[i, j] for i in grid for j in grid], dtype=float)
    return discrete_pmf(copula, pairs, margins).reshape(support + 1, support + 1)
