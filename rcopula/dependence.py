r"""Rank transforms and sample dependence measures.

The entry point for nearly every copula analysis is :func:`pseudo_obs`: copula
methods work on the unit cube, but data does not arrive there. Ranks scaled by
:math:`n+1` are the standard bridge, and using :math:`n+1` rather than :math:`n`
is not cosmetic — dividing by :math:`n` would place a point at exactly 1, where
most copula densities are infinite.

References
----------
Genest, C. and Favre, A.-C. (2007). Everything you always wanted to know about
    copula modeling but were afraid to ask. *Journal of Hydrologic Engineering*
    12(4), 347-368.
    The standard reference for rank-based copula inference.
Kojadinovic, I. (2017). Some copula inference procedures adapted to the presence
    of ties. *Computational Statistics & Data Analysis* 112, 24-41.
    For the tie-handling options.
Blomqvist, N. (1950). On a measure of dependence between two random variables.
    *Annals of Mathematical Statistics* 21(4), 593-600.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "TailEstimate",
    "beta_n",
    "cor_kendall",
    "cor_spearman",
    "fit_lambda",
    "pseudo_obs",
    "to_emp_margins",
]

TIES_METHODS = ("average", "min", "max", "dense", "ordinal", "random")


def pseudo_obs(
    x: ArrayLike,
    ties_method: str = "average",
    lower_tail: bool = True,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64] | pd.DataFrame:
    r"""Turn each column of data into ranks scaled to lie between 0 and 1.

    These are the "pseudo-observations" (R's ``pobs``). Copula methods work on
    values between 0 and 1, but data does not arrive that way; replacing each
    value by its rank within its column puts it there while keeping only the
    information about *how the columns move together*. Call this first on raw
    data before fitting a copula.

    Each column is replaced by :math:`r_{ij} / (n+1)`, where :math:`r_{ij}` is
    the rank of observation :math:`i` within column :math:`j`.

    Parameters
    ----------
    x : array_like or pandas.DataFrame of float, shape (n, d) or (n,)
        Observations, one row per observation and one column per variable. A
        1-D input is treated as a single column, shape ``(n, 1)``. A
        ``pandas`` frame is returned as a frame with its columns and index
        preserved. Must have at least one row.
    ties_method : str, default "average"
        How to rank tied values: one of ``"average"``, ``"min"``, ``"max"``,
        ``"dense"``, ``"ordinal"``, ``"random"``. ``"average"`` gives tied
        values the mean of the ranks they span, as in R. ``"random"`` puts each
        group of tied values in a uniformly random order (distinct values keep
        their order), whatever the magnitude of the data.
    lower_tail : bool, default True
        If ``False``, return ``1 - pseudo_obs(x)``, which is the transform of
        the survival copula (the copula of the data flipped upside down).
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator for breaking ties. Only used when
        ``ties_method="random"``.

    Returns
    -------
    numpy.ndarray of float, shape (n, d), or pandas.DataFrame
        Values strictly inside ``(0, 1)``. A DataFrame (same index and
        columns) when ``x`` was a DataFrame, otherwise an ndarray.

    Raises
    ------
    ValueError
        If ``ties_method`` is not one of the listed names, or ``x`` has no
        rows.

    Notes
    -----
    The scaling by ``n + 1`` keeps every value strictly below 1. Dividing by
    ``n`` would put the largest observation at exactly 1, where Archimedean and
    elliptical densities diverge.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import pseudo_obs
    >>> pseudo_obs([[3.0, 10.0], [1.0, 30.0], [2.0, 20.0]])
    array([[0.75, 0.25],
           [0.25, 0.75],
           [0.5 , 0.5 ]])

    The transform is invariant to any increasing transformation of a margin —
    which is precisely what makes copulas separate dependence from margins:

    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(50, 2))
    >>> bool(np.array_equal(pseudo_obs(x), pseudo_obs(np.exp(x) * 3.0)))
    True

    Column names survive:

    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1.0, 3.0, 2.0], "b": [7.0, 5.0, 9.0]})
    >>> list(pseudo_obs(df).columns)
    ['a', 'b']
    """
    if ties_method not in TIES_METHODS:
        raise ValueError(f"ties_method must be one of {TIES_METHODS}, got {ties_method!r}")

    frame = x if isinstance(x, pd.DataFrame) else None
    arr = np.asarray(frame.to_numpy() if frame is not None else x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    if n == 0:
        raise ValueError("cannot compute pseudo-observations from an empty sample")

    if ties_method == "random":
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        ranks = np.column_stack([_random_tie_ranks(col, rng) for col in arr.T])
    else:
        ranks = np.column_stack([stats.rankdata(col, method=ties_method) for col in arr.T])

    out = ranks / (n + 1.0)
    if not lower_tail:
        out = 1.0 - out

    if frame is not None:
        return pd.DataFrame(out, index=frame.index, columns=frame.columns)
    return out


def _random_tie_ranks(column: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
    """Ranks ``1..n`` with every tie group put in a uniformly random order.

    Sorting on ``(value, random key)`` leaves distinct values in their true
    order and permutes each group of equal values uniformly at random, whatever
    the magnitude of the values. (Adding a tiny jitter instead fails as soon as
    the jitter is below the values' floating-point resolution.) A column with a
    missing value is all-NaN, as :func:`scipy.stats.rankdata` gives.
    """
    n = column.size
    if np.isnan(column).any():
        return np.full(n, np.nan)
    order = np.lexsort((rng.random(n), column))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return ranks


def cor_kendall(x: ArrayLike) -> NDArray[np.float64]:
    """Measure how strongly each pair of columns rises and falls together, by rank.

    Returns the matrix of pairwise Kendall's tau (R's ``corKendall``). Kendall's
    tau is the probability that two observations are ordered the same way in
    both columns minus the probability they are ordered oppositely: 1 means
    the columns always move together, -1 always opposite, 0 no tendency. It
    depends only on ranks, so it is unaffected by the marginal distributions,
    and many copula families have a closed-form link between tau and their
    parameter (see ``from_tau`` on the copula classes).

    Parameters
    ----------
    x : array_like of float, shape (n, d)
        Data or pseudo-observations, one column per variable. Must be 2-D.

    Returns
    -------
    numpy.ndarray of float, shape (d, d)
        Symmetric matrix with ones on the diagonal; entry ``[i, j]`` is
        Kendall's tau between columns ``i`` and ``j``, in ``[-1, 1]``. Ties
        are handled as tau-b, exactly as :func:`scipy.stats.kendalltau`; a
        pair involving a constant column or a missing value is NaN.

    Raises
    ------
    ValueError
        If ``x`` is not 2-D.

    Notes
    -----
    The result is identical, to the last bit, to calling
    :func:`scipy.stats.kendalltau` on every pair of columns. For many columns
    and a moderate number of rows the whole matrix is computed at once from
    blocked products of pairwise-sign matrices (``O(n^2 d^2)`` work, but in
    BLAS), which is hundreds of times faster than ``d(d-1)/2`` separate scipy
    calls -- 800 columns of 1000 rows take seconds rather than minutes. For long
    series with few columns the per-pair ``O(n log n)`` scipy route is used.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, cor_kendall
    >>> u = ClaytonCopula(2.0, dim=3).rvs(2000, random_state=0)
    >>> m = cor_kendall(u)
    >>> m.shape
    (3, 3)
    >>> bool(np.allclose(np.diag(m), 1.0))
    True
    >>> bool(abs(m[0, 1] - 0.5) < 0.05)          # population tau is 0.5
    True
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"cor_kendall needs a 2-D array (n, d); got shape {arr.shape}")
    n, d = arr.shape
    if d < 2:
        return np.eye(d)
    if _kendall_vectorised_is_faster(n, d):
        return _kendall_by_sign_products(arr)
    out = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            out[i, j] = out[j, i] = stats.kendalltau(arr[:, i], arr[:, j]).statistic
    return out


#: Elements per block of the sign tensor in :func:`_kendall_by_sign_products`
#: (float32, so 2**23 elements is 32 MB). Also keeps every per-block sum of
#: +-1 products below 2**24, where float32 stops representing integers exactly.
_KENDALL_BLOCK = 2**23


def _kendall_vectorised_is_faster(n: int, d: int) -> bool:
    """Rough cost model: ``n^2 d (1 + d/600)`` sign products against ``d^2/2``
    scipy calls of ``O(n log n)`` plus a fixed per-call overhead. Both routes
    give identical answers; this only picks the quicker one."""
    pairs = d * (d - 1) / 2.0
    scipy_cost = pairs * (1e-4 + 3e-8 * n * max(np.log2(max(n, 2)), 1.0))
    sign_cost = 0.5 * n * n * d * 2e-9 * (1.0 + d / 600.0)
    return sign_cost < scipy_cost


def _kendall_by_sign_products(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Kendall's tau-b for every pair of columns at once, exactly as scipy computes it.

    For each pair of rows ``i < j`` the concordance of columns ``k`` and ``l``
    is ``sign(x_ik - x_jk) * sign(x_il - x_jl)``, so the matrix of
    ``concordant - discordant`` counts is ``A.T @ A`` with ``A`` the
    ``(n(n-1)/2, d)`` matrix of pairwise signs. Ties give a sign of zero and
    drop out of the numerator automatically, exactly as in tau-b. ``A`` is never
    held whole: it is built and multiplied in blocks of at most
    ``_KENDALL_BLOCK`` entries, in float32 (exact for these small integers, and
    each block's sums stay below 2**24), and the integer counts accumulated in
    int64. The
    final division mirrors :func:`scipy.stats.kendalltau` operation for
    operation, so the result matches it to the last bit.
    """
    n, d = arr.shape
    total = n * (n - 1) // 2
    has_nan = np.isnan(arr).any(axis=0)
    # Dense ranks: exact small integers (so float32 differences are exact signs)
    # and immune to inf - inf.
    ranks = np.zeros((n, d), dtype=np.float32)
    ties = np.zeros(d, dtype=np.int64)
    for j in range(d):
        if has_nan[j]:
            continue
        _, inverse, counts = np.unique(arr[:, j], return_inverse=True, return_counts=True)
        ranks[:, j] = inverse.reshape(-1)
        counts = counts.astype(np.int64)
        ties[j] = int((counts * (counts - 1) // 2).sum())

    concordance = np.zeros((d, d), dtype=np.int64)
    capacity = max(n, _KENDALL_BLOCK // d)
    block = np.empty((capacity, d), dtype=np.float32)

    def accumulate(filled: int) -> None:
        signs = block[:filled]
        # sign() of an integer difference is a clip to [-1, 1]; np.sign is
        # several times slower than these two passes.
        np.minimum(signs, 1.0, out=signs)
        np.maximum(signs, -1.0, out=signs)
        concordance[...] += np.rint(signs.T @ signs).astype(np.int64)

    filled = 0
    for i in range(n - 1):
        count = n - i - 1
        if filled + count > capacity:
            accumulate(filled)
            filled = 0
        np.subtract(ranks[i], ranks[i + 1 :], out=block[filled : filled + count])
        filled += count
    if filled:
        accumulate(filled)

    out = np.eye(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(d):
            for j in range(i + 1, d):
                if has_nan[i] or has_nan[j] or ties[i] == total or ties[j] == total:
                    value = np.nan
                else:
                    value = (
                        int(concordance[i, j])
                        / np.sqrt(total - int(ties[i]))
                        / np.sqrt(total - int(ties[j]))
                    )
                    value = min(1.0, max(-1.0, value))
                out[i, j] = out[j, i] = value
    return out


def cor_spearman(x: ArrayLike) -> NDArray[np.float64]:
    """Measure how strongly each pair of columns is correlated, after ranking them.

    Returns the matrix of pairwise Spearman's rho: the ordinary (Pearson)
    correlation of the ranks. Like Kendall's tau it ranges from -1 to 1 and is
    unaffected by the marginal distributions.

    Parameters
    ----------
    x : array_like of float, shape (n, d)
        Data or pseudo-observations, one column per variable. Must be 2-D.

    Returns
    -------
    numpy.ndarray of float, shape (d, d)
        Symmetric matrix with ones on the diagonal; entry ``[i, j]`` is
        Spearman's rho between columns ``i`` and ``j``, in ``[-1, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import GaussianCopula, cor_spearman
    >>> u = GaussianCopula(0.7, dim=2).rvs(4000, random_state=1)
    >>> bool(abs(cor_spearman(u)[0, 1] - GaussianCopula(0.7).rho()) < 0.05)
    True
    """
    arr = np.asarray(x, dtype=np.float64)
    d = arr.shape[1]
    # scipy.stats.spearmanr collapses to a scalar for exactly two columns, so
    # build the matrix explicitly rather than special-casing the shape.
    out = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            out[i, j] = out[j, i] = stats.spearmanr(arr[:, i], arr[:, j]).statistic
    return out


def beta_n(u: ArrayLike) -> float:
    r"""Measure dependence by how often all columns are on the same side of their medians.

    This is the sample Blomqvist's beta (R's ``betan``).
    :math:`\beta` measures dependence at the centre only: the proportion of
    observations in the two concordant quadrants around the median, rescaled to
    :math:`[-1, 1]`. Cheap to compute and robust, but blind to the tails.

    Parameters
    ----------
    u : array_like of float, shape (n, d)
        Data or pseudo-observations, ``d >= 2`` columns. Only the position of
        each value relative to its column median matters, so raw data works
        as well as ranks.

    Returns
    -------
    float
        The estimate, in ``[-1, 1]`` for ``d = 2``: 1 when every observation is
        in the all-below or all-above corner, about 0 under independence.

    Notes
    -----
    For ``d`` columns the statistic is
    :math:`(2^{d-1}(p_{\le} + p_{>}) - 1) / (2^{d-1} - 1)`, where
    :math:`p_{\le}` and :math:`p_{>}` are the fractions of rows with every
    coordinate at or below, or every coordinate above, its column median.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, beta_n
    >>> u = ClaytonCopula(2.0).rvs(20_000, random_state=3)
    >>> bool(abs(beta_n(u) - ClaytonCopula(2.0).beta()) < 0.03)
    True

    Independence gives approximately zero:

    >>> rng = np.random.default_rng(0)
    >>> bool(abs(beta_n(rng.uniform(size=(20_000, 2)))) < 0.03)
    True
    """
    arr = np.asarray(u, dtype=np.float64)
    d = arr.shape[1]
    centre = np.median(arr, axis=0)
    below = np.all(arr <= centre, axis=1).mean()
    above = np.all(arr > centre, axis=1).mean()
    return float((2.0 ** (d - 1) * (below + above) - 1.0) / (2.0 ** (d - 1) - 1.0))


# ======================================================================
# Nonparametric tail dependence
# ======================================================================
#
# References for this section:
#
# Schmidt, R. and Stadtmuller, U. (2006). Non-parametric estimation of tail
#     dependence. *Scandinavian Journal of Statistics* 33(2), 307-335.
#     The empirical-copula estimator and its asymptotic normality.
# Frahm, G., Junker, M. and Schmidt, R. (2005). Estimating the tail-dependence
#     coefficient: properties and pitfalls.
#     *Insurance: Mathematics and Economics* 37(1), 80-100.
#     Why every estimator here needs a threshold, and what going wrong looks
#     like -- the source of the plateau advice below.
# Caperaa, P., Fougeres, A.-L. and Genest, C. (1997). A nonparametric
#     estimation procedure for bivariate extreme value copulas.
#     *Biometrika* 84(3), 567-577.  The log-ratio estimator.


@dataclass(frozen=True)
class TailEstimate:
    """How likely two variables are to be extreme together, estimated from data alone.

    Returned by :func:`fit_lambda`. ``lower`` estimates the chance that one
    variable is in its extreme low tail given that the other is (joint
    crashes); ``upper`` the same for the high tail. Both are between 0 (no
    tendency to be extreme together) and 1.

    Attributes
    ----------
    lower, upper : float
        The lower- and upper-tail dependence estimates at the chosen
        threshold, each in ``[0, 1]``.
    lower_se, upper_se : float
        Asymptotic standard errors. The counts behind them are binomial, so
        these are only meaningful when ``k`` is not tiny -- below about 20
        exceedances, use a bootstrap instead.
    k : int
        Number of order statistics used (how many of the most extreme
        observations define "the tail").
    n : int
        Sample size.
    method : str
        The estimator used: ``"schmidt-stadtmuller"`` or ``"log"``.
    path : numpy.ndarray of float, shape (m, 3)
        ``(k, lower, upper)`` over a range of thresholds. **Look at this**: a
        threshold-dependent estimator is only believable where the path is flat,
        and the plateau is the estimate. See :func:`~rcopula.plots.tail_plot`.
    """

    lower: float
    upper: float
    lower_se: float
    upper_se: float
    k: int
    n: int
    method: str
    path: NDArray[np.float64]

    def summary(self) -> str:
        """Return a printable text report, with the 95% interval each estimate implies.

        The intervals are estimate +/- 1.96 standard errors, clipped to
        ``[0, 1]``.

        Returns
        -------
        str
            A multi-line table of estimate, standard error and interval for
            each tail.
        """
        return "\n".join(
            [
                f"Tail dependence ({self.method}), n = {self.n}, k = {self.k}",
                "=" * 68,
                f"  {'':<8}{'estimate':>12}{'SE':>10}{'95% lower':>13}{'95% upper':>13}",
                f"  {'lower':<8}{self.lower:>12.4f}{self.lower_se:>10.4f}"
                f"{max(0.0, self.lower - 1.96 * self.lower_se):>13.4f}"
                f"{min(1.0, self.lower + 1.96 * self.lower_se):>13.4f}",
                f"  {'upper':<8}{self.upper:>12.4f}{self.upper_se:>10.4f}"
                f"{max(0.0, self.upper - 1.96 * self.upper_se):>13.4f}"
                f"{min(1.0, self.upper + 1.96 * self.upper_se):>13.4f}",
                "",
                "  These are threshold estimates. Check `path` is flat around k",
                "  before believing either number.",
            ]
        )


def _tail_counts(u: NDArray[np.float64], k: int) -> tuple[float, float]:
    """Empirical copula in each corner at radius ``k/n``, scaled to a ratio."""
    n = u.shape[0]
    threshold = k / n
    lower = float(np.mean(np.all(u <= threshold, axis=1))) / threshold
    upper = float(np.mean(np.all(u > 1.0 - threshold, axis=1))) / threshold
    return lower, upper


def fit_lambda(
    x: ArrayLike,
    k: int | None = None,
    *,
    method: str = "schmidt-stadtmuller",
    ties_method: str = "average",
) -> TailEstimate:
    r"""Estimate how likely two variables are to be extreme together, assuming no family.

    Use this to decide *which* family to fit: whether the data shows joint
    crashes (lower tail), joint booms (upper tail), both, or neither.

    Every parametric estimate of :math:`\lambda` is really an estimate of the
    *family*: fit a Gaussian copula and you will get zero whatever the data
    says, fit a t copula and you will get something positive. This asks the data
    directly, which is the right way round when the question is *which family*.

    Two estimators, both built on the empirical copula near a corner:

    ``"schmidt-stadtmuller"``
        :math:`\hat\lambda_U = \frac{n}{k}\,\hat C\bigl(\text{corner of size }
        k/n\bigr)`, the proportion of points in the corner divided by what
        independence would put there. Asymptotically normal, and the standard
        choice.
    ``"log"``
        :math:`\hat\lambda_U = 2 - \log \hat C(u,u) / \log u` at
        :math:`u = 1 - k/n` (Caperaa, Fougeres and Genest). Less variable in the
        far tail, more biased when the copula is not extreme-value.

    Parameters
    ----------
    x : array_like of float, shape (n, 2)
        Data or pseudo-observations for exactly two variables; ranks are taken
        either way.
    k : int or None, default None
        Number of most-extreme observations to use, with ``1 <= k < n``.
        ``None`` means :math:`\lfloor \sqrt n \rfloor`, which is a
        convention rather than a result -- **look at the path**.
    method : {"schmidt-stadtmuller", "log"}, default "schmidt-stadtmuller"
        Which estimator to use; see above.
    ties_method : str, default "average"
        How to rank tied values; one of the names accepted by
        :func:`pseudo_obs`, which it is passed to.

    Returns
    -------
    TailEstimate
        Lower and upper estimates with standard errors, ``k``, ``n``,
        ``method`` and the threshold ``path``.

    Raises
    ------
    ValueError
        If ``x`` does not have exactly two columns, ``method`` is unknown, or
        ``k`` is outside ``1 <= k < n``.

    Notes
    -----
    There is no threshold-free estimator of a tail dependence coefficient, and
    no automatic choice of ``k`` that is right in general: small ``k`` is
    unbiased and noisy, large ``k`` is stable and measures the middle of the
    distribution rather than the tail. Frahm, Junker and Schmidt recommend
    reading the estimate off a *plateau* in ``k``, and the returned ``path``
    exists so that can be done rather than assumed.

    Examples
    --------
    It separates the two tails of an asymmetric family, which no single number
    for "dependence" can:

    >>> import numpy as np, rcopula as rc
    >>> from rcopula.dependence import fit_lambda
    >>> clayton = rc.ClaytonCopula.from_tau(0.5).rvs(20000, random_state=0)
    >>> estimate = fit_lambda(clayton)
    >>> bool(estimate.lower > 0.5 and estimate.upper < 0.2)
    True

    The Gaussian case is the one to understand before trusting any single
    number. Its true tail dependence is **zero**, but it vanishes only
    logarithmically, so at any feasible threshold the estimate is visibly
    positive -- here 0.22 against a t copula's 0.43 at the same Kendall's tau:

    >>> heavy = rc.StudentCopula.from_tau(0.5, df=3.0).rvs(20000, random_state=0)
    >>> light = rc.GaussianCopula.from_tau(0.5).rvs(20000, random_state=0)
    >>> round(fit_lambda(heavy).upper, 2), round(fit_lambda(light).upper, 2)
    (0.43, 0.22)

    What tells them apart is the *path*, not the point. Real tail dependence
    gives a flat plateau; the Gaussian's estimate slides steadily downwards as
    the threshold is pushed out, because it is converging to zero:

    >>> def slope(u):
    ...     path = fit_lambda(u).path
    ...     keep = path[:, 0] >= 50
    ...     return float(np.polyfit(np.log(path[keep, 0]), path[keep, 2], 1)[0])
    >>> bool(slope(heavy) < 0.5 * slope(light))
    True
    """
    u = np.asarray(pseudo_obs(np.asarray(x, dtype=np.float64), ties_method=ties_method))
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(f"tail dependence is bivariate; got shape {u.shape}")
    n = u.shape[0]
    if method not in ("schmidt-stadtmuller", "log"):
        raise ValueError(f"method must be 'schmidt-stadtmuller' or 'log', got {method!r}")
    chosen = int(np.floor(np.sqrt(n))) if k is None else int(k)
    if not 1 <= chosen < n:
        raise ValueError(f"k must satisfy 1 <= k < n = {n}, got {chosen}")

    def estimate(size: int) -> tuple[float, float]:
        lower, upper = _tail_counts(u, size)
        if method == "log":
            # lambda = 2 - log C(v,v) / log v, evaluated at v = 1 - k/n. *Both*
            # tails use v near one: the lower tail goes through the survival
            # copula, C-hat(v,v) = 2v - 1 + C(1-v, 1-v), not through C at a
            # small argument. Using log(k/n) for the lower tail instead looks
            # symmetric and gives the two tails back swapped.
            radius = size / n
            v = 1.0 - radius
            upper_c = 1.0 - 2.0 * radius + upper * radius
            lower_c = 1.0 - 2.0 * radius + lower * radius
            with np.errstate(divide="ignore", invalid="ignore"):
                lower = 2.0 - float(np.log(max(lower_c, 1e-300)) / np.log(v))
                upper = 2.0 - float(np.log(max(upper_c, 1e-300)) / np.log(v))
        return float(np.clip(lower, 0.0, 1.0)), float(np.clip(upper, 0.0, 1.0))

    lower, upper = estimate(chosen)

    # The corner count is binomial(n, lambda k / n), so the ratio has variance
    # lambda (1 - lambda k/n) / k -- which is why the standard error grows as
    # the threshold is pushed out, and why k cannot simply be made small.
    lower_se = float(np.sqrt(max(lower * (1.0 - lower * chosen / n), 0.0) / chosen))
    upper_se = float(np.sqrt(max(upper * (1.0 - upper * chosen / n), 0.0) / chosen))

    grid = np.unique(np.linspace(max(5, n // 200), max(6, n // 4), 40).astype(int))
    path = np.array([(size, *estimate(int(size))) for size in grid], dtype=float)

    return TailEstimate(
        lower=lower,
        upper=upper,
        lower_se=lower_se,
        upper_se=upper_se,
        k=chosen,
        n=n,
        method=method,
        path=path,
    )


def to_emp_margins(u: ArrayLike, data: ArrayLike) -> NDArray[np.float64]:
    r"""Turn values between 0 and 1 back into values that look like a reference dataset.

    Each column of ``u`` is mapped onto the observed values of the matching
    column of ``data`` (the empirical margins).

    The inverse direction of :func:`pseudo_obs`, and the last step of a
    simulation that wants to keep the data's own marginal shapes rather than
    impose parametric ones. Draw from a fitted copula, push the draws through
    here, and the result has the copula's dependence with the sample's margins
    -- the copula analogue of filtered historical simulation.

    Formally each coordinate is passed through the empirical quantile function
    :math:`\hat F_j^{-1}`, so the output can only take values the reference
    sample actually contains.

    Parameters
    ----------
    u : array_like of float, shape (n, d)
        Values in :math:`[0, 1]`, typically from ``copula.rvs``. A 1-D input
        is treated as a single row.
    data : array_like of float, shape (m, d)
        The reference sample. Needs the same number of columns, not the same
        number of rows.

    Returns
    -------
    numpy.ndarray of float, shape (n, d)
        Column ``j`` holds empirical quantiles of ``data[:, j]`` at levels
        ``u[:, j]``; every value is one that occurs in ``data``.

    Raises
    ------
    ValueError
        If ``u`` and ``data`` have different numbers of columns, or ``u`` has
        values outside ``[0, 1]``.

    Notes
    -----
    The output is confined to the reference sample's range **by construction**:
    an empirical quantile function cannot extrapolate. That is a feature for a
    historical-simulation study and a serious limitation for a tail one -- a
    99.9% capital number computed this way can never exceed the worst loss
    already observed. Use parametric margins when the question is about events
    larger than anything in the sample.

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from rcopula.dependence import to_emp_margins
    >>> rng = np.random.default_rng(0)
    >>> history = rng.lognormal(size=(2000, 2)) * [1.0, 5.0]
    >>> drawn = to_emp_margins(rc.ClaytonCopula(2.0).rvs(5000, random_state=0), history)

    The margins are the sample's, so the medians match -- compared relatively,
    since the second column is scaled by five and an absolute bound would be
    measuring the scaling rather than the agreement:

    >>> ratio = np.median(drawn, axis=0) / np.median(history, axis=0)
    >>> bool(np.all(np.abs(ratio - 1.0) < 0.1))
    True

    ...and the dependence is the copula's, which the history did not have:

    >>> bool(abs(rc.cor_kendall(drawn)[0, 1] - 0.5) < 0.03)
    True

    Nothing outside the reference range can come out:

    >>> bool(drawn.max() <= history.max() and drawn.min() >= history.min())
    True
    """
    values = np.atleast_2d(np.asarray(u, dtype=float))
    reference = np.atleast_2d(np.asarray(data, dtype=float))
    if values.shape[1] != reference.shape[1]:
        raise ValueError(
            f"u has {values.shape[1]} columns and the reference has "
            f"{reference.shape[1]}; they must match"
        )
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("u must lie in [0, 1]; it is meant to be uniform")

    out = np.empty_like(values)
    for j in range(values.shape[1]):
        column = np.sort(reference[:, j])
        # The empirical quantile at level p is the ceil(p*m)-th order statistic,
        # which is the definition R's `quantile(type = 1)` uses and the only one
        # that returns a value the sample actually contains.
        index = np.clip(np.ceil(values[:, j] * column.size).astype(int) - 1, 0, column.size - 1)
        out[:, j] = column[index]
    return out
