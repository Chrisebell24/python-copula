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

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = ["beta_n", "cor_kendall", "cor_spearman", "pseudo_obs"]

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
