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
    """The methods a discrete marginal distribution needs for the functions in this module.

    This is a typing protocol, not a class to instantiate: any object with
    ``cdf``, ``pmf`` and ``ppf`` methods qualifies. It is satisfied by every
    discrete ``scipy.stats`` frozen distribution (``poisson``, ``nbinom``,
    ``binom``, ``geom``, ``randint``, ...), e.g. ``scipy.stats.poisson(3.0)``.

    The methods are, for an array of values ``x`` or probabilities ``q``:

    - ``cdf(x)`` -- ``P(X <= x)``, array of float in ``[0, 1]``;
    - ``pmf(x)`` -- ``P(X = x)``, array of float in ``[0, 1]``;
    - ``ppf(q)`` -- the quantile function, the smallest ``x`` with
      ``cdf(x) >= q``.

    Because this is a ``runtime_checkable`` protocol,
    ``isinstance(margin, DiscreteMargin)`` checks only that the three methods
    exist.
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
    r"""Compute the exact probability of each observed row of counts under a copula model.

    Use this when every variable is discrete (counts, categories coded as
    integers, ...). It returns :math:`P(X_1 = x_1, \dots, X_d = x_d)` for each
    row, combining the copula (the dependence) with the given marginal
    distributions.

    Computes the :math:`2^d`-term inclusion-exclusion sum in the module
    docstring. This is the C-volume of the rectangle
    :math:`\prod_k (F_k(x_k^-), F_k(x_k)]`, which is what a copula's
    :math:`d`-increasing property guarantees is non-negative -- so a negative
    result here means the copula is not one.

    Parameters
    ----------
    copula : Copula
        Any ``d``-dimensional copula from this package, e.g.
        ``GaussianCopula(0.6)``.
    x : array_like of float or int, shape (n, d) or (d,)
        Observed values, on the margins' own scale (e.g. the counts
        themselves, not ranks). A single row may be given as shape ``(d,)``.
    margins : list of DiscreteMargin, length d
        One frozen discrete distribution per column, e.g.
        ``[scipy.stats.poisson(3.0), scipy.stats.poisson(2.0)]``.

    Returns
    -------
    numpy.ndarray of float, shape (n,)
        Probability of each row, in ``[0, 1]``. Tiny negative values from
        floating-point cancellation are set to zero.

    Raises
    ------
    ValueError
        If ``x`` does not have ``copula.dim`` columns, or ``margins`` does not
        have ``copula.dim`` entries.

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
    r"""Compute the likelihood of each row when some variables are discrete and one is continuous.

    For example, a continuous measurement paired with a yes/no outcome. The
    result is a probability for the discrete coordinates and a density for
    the continuous one, which is what a likelihood for such data needs.

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
        Any ``d``-dimensional copula from this package.
    x : array_like of float, shape (n, d) or (d,)
        Observed values on the margins' own scale.
    margins : list of frozen scipy.stats distributions, length d
        Discrete ones must provide ``cdf`` and ``pmf``; continuous ones
        ``cdf`` and ``pdf``.
    discrete : array_like of bool, shape (d,)
        Which coordinates are discrete (``True``) and which continuous
        (``False``). At most one may be ``False`` unless all are.
    step : float, default 1e-5
        Finite-difference step, on the copula scale, for the derivative along
        the continuous coordinate. Used whenever that derivative has no closed
        form: for every copula with ``dim > 2``, and for bivariate families
        other than the Archimedean, Gaussian and Student t ones (whose
        conditional CDFs are analytic, so ``step`` does not affect them). The
        step is shrunk near the edges of the unit interval so the difference
        stays inside it. Must be in ``(0, 0.5)``.

    Returns
    -------
    numpy.ndarray of float, shape (n,)
        The joint mass-density of each row, non-negative.

    Raises
    ------
    ValueError
        If ``discrete``, ``x`` or ``margins`` does not match ``copula.dim``,
        or ``step`` is not in ``(0, 0.5)``.
    NotImplementedError
        If more than one coordinate is continuous while at least one is
        discrete.

    Notes
    -----
    With no discrete coordinates this is the ordinary copula density times the
    marginal densities, and with all of them it is :func:`discrete_pmf`; both
    limits are checked in the test suite. For a bivariate Archimedean,
    Gaussian or t copula the continuous derivative goes through the analytic
    :func:`~rcopula.conditional_cdf`; otherwise it is a central difference of
    the copula CDF with step ``step``, which is only as accurate as that CDF
    (in ``d > 2`` the elliptical CDFs are themselves numerical integrals, so a
    larger step than the default can be the more accurate choice there).

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
    from rcopula.core.archimedean import ArchimedeanCopula
    from rcopula.core.elliptical import GaussianCopula, StudentCopula
    from rcopula.transforms import conditional_cdf

    if not 0.0 < float(step) < 0.5:
        raise ValueError(f"step must be in (0, 0.5), got {step}")
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
    analytic = dim == 2 and isinstance(copula, ArchimedeanCopula | GaussianCopula | StudentCopula)
    total = np.zeros(x.shape[0], dtype=float)
    for corner in itertools.product((0, 1), repeat=discrete_idx.size):
        point = upper.copy()
        for position, which in zip(discrete_idx, corner, strict=True):
            if which:
                point[:, position] = lower[:, position]
        sign = -1.0 if sum(corner) % 2 else 1.0
        point = np.clip(point, 1e-12, 1 - 1e-12)
        if analytic:
            partial = np.asarray(conditional_cdf(copula, point, axis), dtype=float)
        else:
            partial = _partial_derivative(copula, point, axis, float(step))
        total += sign * partial
    return np.maximum(total, 0.0) * np.asarray(margins[axis].pdf(x[:, axis]), dtype=float)


def _partial_derivative(
    copula: Copula, u: NDArray[np.float64], axis: int, step: float
) -> NDArray[np.float64]:
    """``dC/du_axis`` by central differences of the CDF, in any dimension.

    The step is shrunk per row near 0 and 1 so the two evaluation points stay
    inside the unit interval; the result is clipped to ``[0, 1]``, the range of
    a conditional probability.
    """
    centre = u[:, axis]
    h = np.minimum(step, 0.5 * np.minimum(centre, 1.0 - centre))
    h = np.maximum(h, 1e-12)
    hi, lo = u.copy(), u.copy()
    hi[:, axis] = np.minimum(centre + h, 1.0)
    lo[:, axis] = np.maximum(centre - h, 0.0)
    difference = np.asarray(copula.cdf(hi), dtype=float) - np.asarray(copula.cdf(lo), dtype=float)
    return np.clip(difference / (hi[:, axis] - lo[:, axis]), 0.0, 1.0)


def discrete_loglik(copula: Copula, x: ArrayLike, margins: list[Any]) -> float:
    """Score how well a copula with discrete margins explains the data (higher is better).

    Returns the log-likelihood: the sum over rows of the log of
    :func:`discrete_pmf`. Comparing it across parameter values (or, with
    care, families) shows which explains the data best; :func:`fit_discrete`
    maximises it.

    Parameters
    ----------
    copula : Copula
        Any ``d``-dimensional copula from this package.
    x : array_like of float or int, shape (n, d)
        Observed values on the margins' own scale.
    margins : list of DiscreteMargin, length d
        One frozen discrete distribution per column.

    Returns
    -------
    float
        The log-likelihood. Rows with probability below ``1e-300`` contribute
        ``log(1e-300)`` (about -691) rather than minus infinity.

    Raises
    ------
    ValueError
        As for :func:`discrete_pmf`.

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
    """The result of fitting a copula to discrete data: the fitted copula plus fit statistics.

    Returned by :func:`fit_discrete`; you normally do not construct it
    yourself. Call :meth:`summary` for a readable report.

    Attributes
    ----------
    copula : Copula
        The copula at the estimated parameters.
    params : numpy.ndarray of float, shape (p,)
        All of the copula's parameters (free and fixed), in the order of
        ``copula.param_names``.
    loglik : float
        Log-likelihood at the estimate.
    n_obs : int
        Number of observations (rows) used.
    converged : bool
        Whether the optimiser reported success.
    independent_loglik : float
        The same margins under independence, for a likelihood ratio.
    message : str, default ""
        The optimiser's status message; useful when ``converged`` is False.
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
        """Number of copula parameters that were estimated (not held fixed).

        Returns
        -------
        int
        """
        return int(np.sum(self.copula.free))

    @property
    def aic(self) -> float:
        """Akaike information criterion, ``2 * n_params - 2 * loglik``; lower is better.

        Returns
        -------
        float
        """
        return float(2 * self.n_params - 2 * self.loglik)

    @property
    def bic(self) -> float:
        """Bayesian information criterion, ``n_params * log(n_obs) - 2 * loglik``; lower is better.

        Returns
        -------
        float
        """
        return float(self.n_params * np.log(self.n_obs) - 2 * self.loglik)

    def independence_test(self) -> tuple[float, float]:
        """Test whether the variables are dependent at all, against the independence model.

        This is a likelihood-ratio test. A small p-value is evidence of
        dependence.

        Unlike the constancy test in :mod:`rcopula.dynamic`, this null is
        interior for every family here, so the chi-squared reference is the
        usual asymptotic one.

        Returns
        -------
        statistic : float
            ``2 * (loglik - independent_loglik)``, floored at zero.
        pvalue : float
            Upper-tail probability of a chi-squared distribution with
            ``n_params`` degrees of freedom.
        """
        from scipy import stats as _stats

        statistic = max(2.0 * (self.loglik - self.independent_loglik), 0.0)
        return float(statistic), float(_stats.chi2(self.n_params).sf(statistic))

    def summary(self) -> str:
        """Return a printable text report of the fit.

        Includes the parameter estimates, log-likelihood, AIC/BIC, the
        likelihood-ratio test against independence, and a warning if the
        optimiser did not converge.

        Returns
        -------
        str
        """
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
    """Estimate a copula's parameters from discrete data (counts, codes), by maximum likelihood.

    The margins are taken as given -- fit them separately, which is the
    inference-functions-for-margins two-step and is what everyone does. The
    copula parameter is then the only unknown.

    Parameters
    ----------
    x : array_like of float or int, shape (n, d)
        Observed counts or codes, on the margins' own scale.
    copula : Copula
        The family to fit, with ``dim == d``. Its current parameters are the
        starting point, and any parameters it marks as fixed stay fixed.
    margins : list of DiscreteMargin, length d
        One already-fitted frozen discrete distribution per column.
    start : array_like of float or None, default None
        Starting values for the *free* parameters only, overriding the
        copula's current ones: exactly one finite value per free parameter.
        ``None`` uses the copula's current values.

    Returns
    -------
    DiscreteFitResult
        The fitted copula, its parameters, log-likelihood, convergence flag
        and the log-likelihood of the same margins under independence. If the
        copula has no free parameters nothing is optimised: the result holds
        the copula as given and its log-likelihood, and ``independent_loglik``
        is still the independence copula's, so the likelihood ratio compares
        the given copula with independence.

    Raises
    ------
    ValueError
        If ``x`` or ``margins`` does not match ``copula.dim``, or ``start``
        does not have one finite value per free parameter.

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
    from rcopula.core.other import IndependenceCopula

    x = np.atleast_2d(np.asarray(x, dtype=float))
    free = np.asarray(copula.free, dtype=bool)
    if start is not None:
        start_values = np.atleast_1d(np.asarray(start, dtype=float))
        if start_values.ndim != 1 or start_values.size != int(free.sum()):
            raise ValueError(
                f"start must hold one value per free parameter ({int(free.sum())}), "
                f"got shape {np.shape(start)}"
            )
        if not np.all(np.isfinite(start_values)):
            raise ValueError(f"start must be finite, got {start_values.tolist()}")
    independent_loglik = discrete_loglik(IndependenceCopula(copula.dim), x, margins)
    if not free.any():
        loglik = discrete_loglik(copula, x, margins)
        return DiscreteFitResult(
            copula=copula,
            params=np.asarray(copula.params, dtype=float),
            loglik=loglik,
            n_obs=x.shape[0],
            converged=True,
            independent_loglik=independent_loglik,
            message="no free parameters; the copula was evaluated, not fitted",
        )

    initial = np.array(copula.params, dtype=float)[free] if start is None else start_values
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

    return DiscreteFitResult(
        copula=fitted,
        params=params,
        loglik=float(-result.fun),
        n_obs=x.shape[0],
        converged=bool(result.success),
        independent_loglik=independent_loglik,
        message=str(result.message),
    )


def distributional_transform(
    x: ArrayLike,
    margins: list[Any],
    *,
    random_state: Any = None,
    replicates: int = 1,
) -> NDArray[np.float64]:
    r"""Turn discrete observations into values between 0 and 1 by adding controlled randomness.

    Each count is replaced by a random point inside the probability interval
    it occupies, so the result is exactly uniform and can be fed to any
    method that assumes continuous data (fitting, plots, tests). The
    randomness is part of the answer, so repeat with different seeds and
    compare.

    The distributional transform randomises within each atom,

    .. math:: U = F(X^-) + V\,\bigl(F(X) - F(X^-)\bigr), \qquad V \sim U(0,1),

    with :math:`V` independent of :math:`X`. The result is **exactly** uniform,
    not approximately -- which is what separates this from ad-hoc jittering --
    and its copula is a copula of :math:`X`, chosen at random from the
    identified class.

    Parameters
    ----------
    x : array_like of float or int, shape (n, d) or (d,)
        Observed values on the margins' own scale.
    margins : list of frozen scipy.stats distributions, length d
        One per column. A margin with a ``pmf`` method is treated as discrete;
        any other (continuous) margin is simply passed through its ``cdf``.
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator for the randomisation.
    replicates : int, default 1
        Draw this many independent transforms (at least 1) and return their
        average on the copula scale. Averaging reduces the randomisation noise
        but biases the result towards the middle of each atom, so the default
        is 1 and anything else is a deliberate trade.

    Returns
    -------
    numpy.ndarray of float, shape (n, d)
        Pseudo-observations, clipped to ``[1e-12, 1 - 1e-12]``.

    Raises
    ------
    ValueError
        If ``margins`` does not have one entry per column of ``x``, or
        ``replicates < 1``.

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


def _lattice_masses(
    margins: list[Any], support: Any, caller: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The two margins' masses on ``0, ..., support``, after checking that is
    where they live.

    Both lattice functions index their output by the count itself, so a margin
    with mass below zero or off the integers would be silently truncated or
    missed entirely. Detect that from the margin's own CDF and refuse.
    """
    if isinstance(support, bool) or not isinstance(support, int | np.integer) or int(support) < 0:
        raise ValueError(f"{caller}: support must be a non-negative integer, got {support!r}")
    grid = np.arange(int(support) + 1)
    masses = []
    for k, margin in enumerate(margins):
        below = float(np.asarray(margin.cdf(-0.5)))
        if below > 1e-12:
            raise ValueError(
                f"{caller}: margin {k} puts probability {below:.3g} on negative values; "
                "it must be supported on the non-negative integers 0, 1, 2, ... "
                "(shift it, e.g. with scipy's loc, so its smallest value is 0)"
            )
        pmf = np.asarray(margin.pmf(grid), dtype=float)
        within = float(np.asarray(margin.cdf(int(support))))
        if abs(within - float(pmf.sum())) > 1e-9 * max(1.0, within):
            raise ValueError(
                f"{caller}: margin {k} puts probability on values that are not "
                "integers; it must be supported on the non-negative integers"
            )
        masses.append(pmf)
    return masses[0], masses[1]


def tau_upper_bound(margins: list[Any], *, support: int = 200) -> float:
    r"""Compute the largest rank correlation (Kendall's tau-b) two discrete margins allow.

    Use it to judge a measured correlation between discrete variables: a tau
    of 0.1 may already be the strongest dependence the margins permit.

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
    margins : list of DiscreteMargin, length 2
        Two frozen discrete distributions supported on the non-negative
        integers ``0, 1, 2, ...``.
    support : int, default 200
        How far up the lattice to evaluate: values ``0, ..., support`` are
        used. The tail beyond this contributes less than its mass, so the
        default is generous for anything with a finite mean.

    Returns
    -------
    float
        The attainable maximum of Kendall's tau-b, in ``[0, 1]``. Returns 0.0
        if either margin puts all its mass on one value.

    Raises
    ------
    ValueError
        If ``margins`` does not have exactly two entries, ``support`` is not a
        non-negative integer, or a margin puts mass on negative or non-integer
        values (which the lattice ``0, ..., support`` would silently miss).

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
    p, q = _lattice_masses(margins, support, "tau_upper_bound")

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
    r"""Tabulate the probability of every pair of count values under a bivariate copula model.

    The table is the checkerboard copula's mass on the lattice induced by the
    margins, and is the natural thing to plot as a picture of a copula fitted
    to discrete data.

    Since the copula is identified only on :math:`\mathrm{Ran}\,F_1 \times
    \mathrm{Ran}\,F_2`, an infinite family of copulas fits any discrete data
    equally well. The **checkerboard** member spreads each cell's mass uniformly
    over its rectangle: it is the canonical representative, it is the one the
    distributional transform targets on average, and it is the one to plot when
    a picture of "the" fitted copula is wanted.

    Parameters
    ----------
    copula : Copula
        A bivariate copula (``dim == 2``).
    margins : list of DiscreteMargin, length 2
        Two frozen discrete distributions supported on the non-negative
        integers.
    support : int, default 60
        Lattice extent: values ``0, ..., support`` are tabulated in each
        coordinate.

    Returns
    -------
    numpy.ndarray of float, shape (support + 1, support + 1)
        Cell probabilities; entry ``[i, j]`` is :math:`P(X_1 = i, X_2 = j)`.
        They sum to one up to the tail truncation.

    Raises
    ------
    ValueError
        If the copula is not bivariate, ``margins`` does not have two entries,
        ``support`` is not a non-negative integer, or a margin puts mass on
        negative or non-integer values.

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
    if len(margins) != 2:
        raise ValueError(f"checkerboard is bivariate; got {len(margins)} margins")
    _lattice_masses(margins, support, "checkerboard")
    grid = np.arange(support + 1)
    pairs = np.array([[i, j] for i in grid for j in grid], dtype=float)
    return discrete_pmf(copula, pairs, margins).reshape(support + 1, support + 1)
