r"""Goodness-of-fit test statistics for copulas.

All of these compare a fitted parametric copula against the data's own
empirical copula. They differ in *what* they compare and hence in what kind of
departure they are sensitive to.

``Sn``
    Cramer-von Mises distance between the empirical and fitted copulas,
    :math:`S_n = \sum_i \{C_n(\hat U_i) - C_{\hat\theta}(\hat U_i)\}^2`.
    The blanket test: no tuning, good all-round power, and the one to reach for
    by default.

``Tn``
    The Kolmogorov-Smirnov analogue, :math:`\max_i |C_n - C_{\hat\theta}|`.
    Less powerful than ``Sn`` in nearly every published comparison, but
    occasionally more sensitive to a single localised discrepancy.

``AnChisq`` / ``AnGamma``
    Anderson-Darling statistics applied after collapsing the Rosenblatt-
    transformed data to one dimension -- by a chi-squared or a gamma
    aggregation respectively. Cheap, and sensitive in the tails where ``Sn``
    is weakest, but they only see the aggregated variable.

References
----------
Genest, C., Remillard, B. and Beaudoin, D. (2009). Goodness-of-fit tests for
    copulas: a review and a power study.
    *Insurance: Mathematics and Economics* 44(2), 199-213.
    The definitive comparison; ``Sn`` is its recommendation.
Genest, C. and Remillard, B. (2008). Validity of the parametric bootstrap for
    goodness-of-fit testing in semiparametric models.
    *Annales de l'IHP Probabilites et Statistiques* 44(6), 1096-1127.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from rcopula.core.base import Copula

__all__ = ["STATISTICS", "empirical_copula_at", "gof_statistic"]

STATISTICS = ("Sn", "Tn", "AnChisq", "AnGamma")


def empirical_copula_at(u: ArrayLike, at: ArrayLike | None = None) -> NDArray[np.float64]:
    r"""Empirical copula :math:`C_n` evaluated at each row of ``at``.

    Uses the ``1/n`` scaling of Genest-Remillard-Beaudoin, matching R's
    ``C.n`` with ``offset = 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.gof.statistics import empirical_copula_at
    >>> u = np.array([[0.2, 0.3], [0.5, 0.6], [0.8, 0.9]])
    >>> empirical_copula_at(u)
    array([0.33333333, 0.66666667, 1.        ])
    """
    data = np.atleast_2d(np.asarray(u, dtype=np.float64))
    points = data if at is None else np.atleast_2d(np.asarray(at, dtype=np.float64))
    n = data.shape[0]
    below = np.all(data[None, :, :] <= points[:, None, :], axis=2)
    return below.sum(axis=1) / n


def _anderson_darling(w: NDArray[np.float64]) -> float:
    r"""Anderson-Darling statistic for uniformity of ``w`` on ``(0, 1)``.

    :math:`A^2 = -n - \frac{1}{n}\sum_i (2i-1)\{\log w_{(i)} + \log(1 - w_{(n+1-i)})\}`.
    """
    n = w.size
    x = np.sort(np.clip(w, 1e-12, 1.0 - 1e-12))
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (np.log(x) + np.log1p(-x[::-1]))) / n)


def gof_statistic(
    u: ArrayLike,
    copula: Copula | None = None,
    method: str = "Sn",
) -> float:
    r"""Evaluate a goodness-of-fit statistic (R's ``gofTstat``).

    Parameters
    ----------
    u : array_like
        ``(n, d)`` pseudo-observations. For ``AnChisq``/``AnGamma`` these should
        already be Rosenblatt-transformed, as in R.
    copula : Copula, optional
        The fitted copula, required for ``Sn`` and ``Tn``.
    method : {"Sn", "Tn", "AnChisq", "AnGamma"}

    Returns
    -------
    float
        The statistic. Larger values indicate worse fit.

    Examples
    --------
    A well-specified model gives a small ``Sn``; a misspecified one does not:

    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, GumbelCopula, fit, pseudo_obs
    >>> from rcopula.gof import gof_statistic
    >>> u = pseudo_obs(ClaytonCopula(4.0).rvs(500, random_state=0))
    >>> right = gof_statistic(u, fit(ClaytonCopula(), u).copula)
    >>> wrong = gof_statistic(u, fit(GumbelCopula(), u).copula)
    >>> bool(wrong > 5 * right)
    True
    """
    if method not in STATISTICS:
        raise ValueError(f"method must be one of {STATISTICS}, got {method!r}")

    data = np.atleast_2d(np.asarray(u, dtype=np.float64))
    n, d = data.shape

    if method in ("Sn", "Tn"):
        if copula is None:
            raise ValueError(f"method={method!r} needs a fitted copula")
        diff = empirical_copula_at(data) - copula.cdf(data)
        return float(np.sum(diff**2)) if method == "Sn" else float(n * np.max(np.abs(diff)))

    clipped = np.clip(data, 1e-12, 1.0 - 1e-12)
    if method == "AnChisq":
        # Sum of squared normal quantiles is chi-squared with d degrees of
        # freedom when the transformed data really are independent uniforms.
        w = stats.chi2.cdf(np.sum(stats.norm.ppf(clipped) ** 2, axis=1), df=d)
    else:
        # Sum of -log u is Erlang(d, 1) under the same null.
        w = stats.gamma.cdf(np.sum(-np.log(clipped), axis=1), a=d, scale=1.0)
    return _anderson_darling(w)
