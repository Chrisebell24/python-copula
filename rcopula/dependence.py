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

__all__ = ["TailEstimate", "beta_n", "cor_kendall", "cor_spearman", "fit_lambda", "pseudo_obs"]

TIES_METHODS = ("average", "min", "max", "dense", "ordinal", "random")


def pseudo_obs(
    x: ArrayLike,
    ties_method: str = "average",
    lower_tail: bool = True,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64] | pd.DataFrame:
    r"""Rank-transform data onto the unit cube (R's ``pobs``).

    Each column is replaced by :math:`r_{ij} / (n+1)`, where :math:`r_{ij}` is
    the rank of observation :math:`i` within column :math:`j`.

    Parameters
    ----------
    x : array_like or DataFrame
        ``(n, d)`` observations. A ``pandas`` frame is returned as a frame with
        its columns and index preserved.
    ties_method : str
        How to rank tied values: one of ``average``, ``min``, ``max``,
        ``dense``, ``ordinal``, ``random``. ``average`` is the default, as in R.
    lower_tail : bool
        If ``False``, return ``1 - pseudo_obs(x)``, which is the transform of
        the survival copula.
    random_state : Generator, int or None
        Only used when ``ties_method="random"``, which breaks ties at random.

    Returns
    -------
    ndarray or DataFrame
        Values strictly inside ``(0, 1)``.

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
        ranks = np.column_stack(
            [stats.rankdata(col + rng.uniform(0, 1e-12, n), method="ordinal") for col in arr.T]
        )
    else:
        ranks = np.column_stack([stats.rankdata(col, method=ties_method) for col in arr.T])

    out = ranks / (n + 1.0)
    if not lower_tail:
        out = 1.0 - out

    if frame is not None:
        return pd.DataFrame(out, index=frame.index, columns=frame.columns)
    return out


def cor_kendall(x: ArrayLike) -> NDArray[np.float64]:
    """Pairwise Kendall's tau matrix (R's ``corKendall``).

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
    d = arr.shape[1]
    out = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            out[i, j] = out[j, i] = stats.kendalltau(arr[:, i], arr[:, j]).statistic
    return out


def cor_spearman(x: ArrayLike) -> NDArray[np.float64]:
    """Pairwise Spearman's rho matrix.

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
    r"""Sample Blomqvist's beta (R's ``betan``).

    :math:`\beta` measures dependence at the centre only: the proportion of
    observations in the two concordant quadrants around the median, rescaled to
    :math:`[-1, 1]`. Cheap to compute and robust, but blind to the tails.

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
    """A nonparametric tail dependence estimate.

    Attributes
    ----------
    lower, upper : float
        The estimates at the chosen threshold.
    lower_se, upper_se : float
        Asymptotic standard errors. The counts behind them are binomial, so
        these are only meaningful when ``k`` is not tiny -- below about 20
        exceedances, use a bootstrap instead.
    k : int
        Number of order statistics used.
    n : int
    method : str
    path : ndarray, shape (m, 3)
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
        """A printable report, with the interval each estimate implies."""
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
    r"""Estimate tail dependence without assuming a family.

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
    x : array_like, shape (n, 2)
        Data or pseudo-observations; ranks are taken either way.
    k : int, optional
        Order statistics to use. Defaults to :math:`\lfloor \sqrt n \rfloor`,
        which is a convention rather than a result -- **look at the path**.
    method : {"schmidt-stadtmuller", "log"}
    ties_method : str
        Passed to :func:`pseudo_obs`.

    Returns
    -------
    TailEstimate

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
