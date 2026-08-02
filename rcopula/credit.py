r"""Credit portfolio and structured-product utilities.

Copulas entered mainstream finance through credit: Li (2000) proposed joining
individual default times with a Gaussian copula, and the whole CDO market was
built on that idea. Its failure in 2007-08 is the most consequential
demonstration of the point this package keeps returning to -- **the Gaussian
copula has no tail dependence**, so it prices simultaneous defaults as
essentially impossible, and senior tranches were sold on that assumption.

Everything here is deliberately model-agnostic: pass any copula and see what
changes. Swapping :class:`~rcopula.GaussianCopula` for
:class:`~rcopula.StudentCopula` or :class:`~rcopula.ClaytonCopula` at matched
Kendall's tau reprices senior tranches by multiples, and reproducing that is
the most useful thing this module does.

==============================  ==============================================
:func:`default_indicators`      Which names default by the horizon.
:func:`default_times`           Li (2000) copula-linked default times.
:func:`portfolio_loss`          Simulated fractional portfolio loss.
:func:`tranche_loss`            Loss allocated to one tranche.
:func:`tranche_expected_loss`   Expected tranche loss, as a fraction.
:func:`tranche_spread`          Approximate fair spread.
:func:`nth_to_default_probability`  Basket default swap leg.
:func:`vasicek_loss_cdf`        Large-pool closed form -- the validation anchor.
:func:`implied_correlation`     Correlation implying a given tranche loss.
==============================  ==============================================

.. warning::

   These are **reference implementations for analysis and teaching**, not a
   production pricing library. Spreads use a flat-hazard, single-period
   approximation with no discounting curve, no accrual on default and no
   counterparty adjustment.

References
----------
Li, D. X. (2000). On default correlation: a copula function approach.
    *Journal of Fixed Income* 9(4), 43-54.
Vasicek, O. (2002). The distribution of loan portfolio value.
    *Risk* 15(12), 160-162. The large-homogeneous-pool limit.
Gordy, M. B. (2003). A risk-factor model foundation for ratings-based bank
    capital rules. *Journal of Financial Intermediation* 12(3), 199-232.
Hull, J. and White, A. (2004). Valuation of a CDO and an nth-to-default CDS
    without Monte Carlo simulation. *Journal of Derivatives* 12(2), 8-23.
MacKenzie, D. and Spears, T. (2014). "The formula that killed Wall Street":
    the Gaussian copula and modelling practices in investment banking.
    *Social Studies of Science* 44(3), 393-417.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize
from scipy.special import ndtr, ndtri

from rcopula.core.base import Copula

__all__ = [
    "default_indicators",
    "default_times",
    "implied_correlation",
    "nth_to_default_probability",
    "portfolio_loss",
    "tranche_expected_loss",
    "tranche_loss",
    "tranche_spread",
    "vasicek_loss_cdf",
]


def _as_vector(value: ArrayLike, dim: int, name: str) -> NDArray[np.float64]:
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if arr.size == 1:
        return np.full(dim, float(arr[0]))
    if arr.size != dim:
        raise ValueError(f"{name} has length {arr.size}, expected 1 or {dim}")
    return arr


def default_indicators(
    copula: Copula,
    default_prob: ArrayLike,
    n: int = 100_000,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.bool_]:
    r"""Simulate which names default before the horizon.

    A name defaults when its copula coordinate falls below its default
    probability: :math:`U_i \le p_i`. That is exactly the one-factor threshold
    model when ``copula`` is Gaussian with exchangeable correlation, but works
    for any family.

    Returns
    -------
    ndarray of bool
        ``(n, d)``, ``True`` where the name defaulted.

    Examples
    --------
    The marginal default rate is whatever you asked for, regardless of copula:

    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, GaussianCopula
    >>> from rcopula.credit import default_indicators
    >>> d = default_indicators(GaussianCopula(0.3, dim=50), 0.02, 40_000, 0)
    >>> bool(abs(d.mean() - 0.02) < 0.002)
    True

    but the *joint* behaviour is not. Clayton produces far more all-or-nothing
    outcomes at the same marginal probability:

    >>> gauss = default_indicators(GaussianCopula(0.3, dim=50), 0.02, 40_000, 0)
    >>> clayton = default_indicators(ClaytonCopula(2.0, dim=50), 0.02, 40_000, 0)
    >>> bool(clayton.sum(axis=1).max() > gauss.sum(axis=1).max())
    True
    """
    p = _as_vector(default_prob, copula.dim, "default_prob")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("default probabilities must lie in [0, 1]")
    u = copula.rvs(n, random_state=random_state)
    return u <= p


def default_times(
    copula: Copula,
    hazard_rate: ArrayLike,
    n: int = 100_000,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    r"""Copula-linked default times under constant hazard rates (Li 2000).

    With a flat hazard :math:`\lambda_i`, the survival function is
    :math:`e^{-\lambda_i t}`, so :math:`\tau_i = -\log(1 - U_i)/\lambda_i`
    turns a copula draw into a default time while preserving each name's own
    credit curve.

    Examples
    --------
    Marginal default times are exponential whatever the copula:

    >>> import numpy as np
    >>> from scipy import stats
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.credit import default_times
    >>> t = default_times(ClaytonCopula(3.0, dim=4), 0.02, 20_000, random_state=0)
    >>> bool(stats.kstest(t[:, 0], stats.expon(scale=1 / 0.02).cdf).pvalue > 0.01)
    True
    """
    lam = _as_vector(hazard_rate, copula.dim, "hazard_rate")
    if np.any(lam <= 0):
        raise ValueError("hazard rates must be strictly positive")
    u = copula.rvs(n, random_state=random_state)
    return -np.log1p(-u) / lam


def portfolio_loss(
    copula: Copula,
    default_prob: ArrayLike,
    lgd: ArrayLike = 0.6,
    exposure: ArrayLike | None = None,
    n: int = 100_000,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    r"""Simulate the fractional loss on a credit portfolio.

    Parameters
    ----------
    default_prob : array_like
        Per-name default probability to the horizon.
    lgd : array_like
        Loss given default, as a fraction of exposure.
    exposure : array_like, optional
        Per-name exposure. Defaults to equal. Losses are returned as a fraction
        of total exposure, so the result always lies in ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import GaussianCopula
    >>> from rcopula.credit import portfolio_loss
    >>> loss = portfolio_loss(GaussianCopula(0.2, dim=100), 0.03, 0.6, n=20_000,
    ...                       random_state=0)
    >>> bool(0 <= loss.min() and loss.max() <= 1)
    True
    >>> bool(abs(loss.mean() - 0.03 * 0.6) < 0.003)      # mean loss = PD x LGD
    True
    """
    d = copula.dim
    p = _as_vector(default_prob, d, "default_prob")
    severity = _as_vector(lgd, d, "lgd")
    weight = np.full(d, 1.0) if exposure is None else _as_vector(exposure, d, "exposure")
    weight = weight / weight.sum()

    defaulted = default_indicators(copula, p, n, random_state)
    return defaulted @ (weight * severity)


# ======================================================================
# Tranches
# ======================================================================


def tranche_loss(loss: ArrayLike, attachment: float, detachment: float) -> NDArray[np.float64]:
    r"""Loss allocated to a tranche, as a fraction of tranche notional.

    A tranche spanning :math:`[a, d]` absorbs
    :math:`\min(\max(L - a, 0),\, d - a)`, rescaled by its width. Equity
    tranches take the first losses; senior tranches are untouched until the
    subordination below them is exhausted, which is precisely why their pricing
    is so sensitive to the probability of *many* simultaneous defaults.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.credit import tranche_loss
    >>> losses = np.array([0.0, 0.02, 0.05, 0.10, 0.30])
    >>> tranche_loss(losses, 0.03, 0.07)     # a 3-7% mezzanine tranche
    array([0. , 0. , 0.5, 1. , 1. ])
    """
    if not 0.0 <= attachment < detachment <= 1.0:
        raise ValueError(
            f"need 0 <= attachment < detachment <= 1, got [{attachment}, {detachment}]"
        )
    x = np.asarray(loss, dtype=np.float64)
    width = detachment - attachment
    return np.clip(x - attachment, 0.0, width) / width


def tranche_expected_loss(loss: ArrayLike, attachment: float, detachment: float) -> float:
    """Expected tranche loss, as a fraction of tranche notional.

    Examples
    --------
    >>> from rcopula import GaussianCopula
    >>> from rcopula.credit import portfolio_loss, tranche_expected_loss
    >>> loss = portfolio_loss(GaussianCopula(0.2, dim=100), 0.05, n=40_000,
    ...                       random_state=0)
    >>> equity = tranche_expected_loss(loss, 0.0, 0.03)
    >>> senior = tranche_expected_loss(loss, 0.15, 0.30)
    >>> bool(equity > senior)          # equity absorbs losses first
    True
    """
    return float(np.mean(tranche_loss(loss, attachment, detachment)))


def tranche_spread(
    loss: ArrayLike,
    attachment: float,
    detachment: float,
    maturity: float = 5.0,
    discount_rate: float = 0.0,
) -> float:
    r"""Approximate fair running spread on a tranche, in basis points.

    Equates the protection leg to the premium leg under a **single-period,
    flat-curve** approximation:

    .. math::  s \approx \frac{\mathrm{EL}}{T \cdot (1 - \mathrm{EL}/2)}

    with the second factor a crude adjustment for notional amortising as losses
    accrue. Real pricing integrates over a default-time distribution with a
    discount curve and premium accruals; this is for comparing *models*, not
    for quoting.

    Examples
    --------
    Equity trades far wider than senior:

    >>> from rcopula import GaussianCopula
    >>> from rcopula.credit import portfolio_loss, tranche_spread
    >>> loss = portfolio_loss(GaussianCopula(0.2, dim=100), 0.05, n=40_000,
    ...                       random_state=0)
    >>> bool(tranche_spread(loss, 0.0, 0.03) > 10 * tranche_spread(loss, 0.15, 0.30))
    True
    """
    el = tranche_expected_loss(loss, attachment, detachment)
    if maturity <= 0:
        raise ValueError(f"maturity must be positive, got {maturity}")
    discount = np.exp(-discount_rate * maturity / 2.0) if discount_rate else 1.0
    denominator = maturity * max(1.0 - el / 2.0, 1e-12) * discount
    return float(1e4 * el / denominator)


def nth_to_default_probability(
    copula: Copula,
    default_prob: ArrayLike,
    n_th: int = 1,
    n: int = 100_000,
    random_state: np.random.Generator | int | None = None,
) -> float:
    r"""Probability that at least ``n_th`` names default -- a basket default swap.

    First-to-default is worth most when defaults are *independent* (many chances
    for a first one); ``n``-th-to-default is worth most when they are
    *dependent* (defaults arrive together). That inversion is the whole
    economics of correlation trading, and it is visible directly here.

    Examples
    --------
    >>> from rcopula import ClaytonCopula, IndependenceCopula
    >>> from rcopula.credit import nth_to_default_probability as ntd
    >>> free = IndependenceCopula(10)
    >>> tied = ClaytonCopula(5.0, dim=10)
    >>> bool(ntd(free, 0.05, 1, 40_000, 0) > ntd(tied, 0.05, 1, 40_000, 0))
    True
    >>> bool(ntd(tied, 0.05, 5, 40_000, 0) > ntd(free, 0.05, 5, 40_000, 0))
    True
    """
    if not 1 <= n_th <= copula.dim:
        raise ValueError(f"n_th must lie in 1..{copula.dim}, got {n_th}")
    defaulted = default_indicators(copula, default_prob, n, random_state)
    return float(np.mean(defaulted.sum(axis=1) >= n_th))


# ======================================================================
# The large-pool limit, and correlation implied from it
# ======================================================================


def vasicek_loss_cdf(x: ArrayLike, default_prob: float, correlation: float) -> NDArray[np.float64]:
    r"""Vasicek large-homogeneous-pool loss distribution.

    .. math::

        P(L \le x) = \Phi\!\left(
            \frac{\sqrt{1-\rho}\,\Phi^{-1}(x) - \Phi^{-1}(p)}{\sqrt{\rho}}\right).

    The closed-form limit of the one-factor Gaussian model as the number of
    names grows, and the foundation of the Basel IRB capital formula. It is the
    **validation anchor** for this module: a simulated Gaussian-copula portfolio
    with many names must converge to it.

    Examples
    --------
    Simulation converges to the closed form:

    >>> import numpy as np
    >>> from rcopula import GaussianCopula
    >>> from rcopula.credit import portfolio_loss, vasicek_loss_cdf
    >>> pd_, rho = 0.05, 0.2
    >>> loss = portfolio_loss(GaussianCopula(rho, dim=800), pd_, lgd=1.0,
    ...                       n=40_000, random_state=0)
    >>> empirical = np.mean(loss <= 0.10)
    >>> closed_form = float(vasicek_loss_cdf(0.10, pd_, rho))
    >>> bool(abs(empirical - closed_form) < 0.02)
    True
    """
    if not 0.0 < default_prob < 1.0:
        raise ValueError(f"default_prob must lie in (0, 1), got {default_prob}")
    if not 0.0 < correlation < 1.0:
        raise ValueError(f"correlation must lie in (0, 1), got {correlation}")
    q = np.clip(np.asarray(x, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return ndtr(
        (np.sqrt(1.0 - correlation) * ndtri(q) - ndtri(default_prob)) / np.sqrt(correlation)
    )


def implied_correlation(
    target_expected_loss: float,
    default_prob: float,
    attachment: float,
    detachment: float,
    lgd: float = 1.0,
    n_names: int = 200,
    n: int = 40_000,
    random_state: np.random.Generator | int | None = None,
) -> float:
    r"""Gaussian-copula correlation reproducing a given tranche expected loss.

    The market's "implied correlation": invert the one-factor Gaussian model
    until it reproduces an observed tranche price. Doing this tranche by tranche
    on the same pool produces the **correlation skew** -- different tranches
    implying different correlations, which is a contradiction if the model were
    right, and is the standard evidence that it is not.

    Raises
    ------
    ValueError
        If the target is unattainable over ``correlation`` in ``(0, 1)``.

    Examples
    --------
    Round-trip: price a tranche at a known correlation, then recover it.

    >>> from rcopula import GaussianCopula
    >>> from rcopula.credit import (
    ...     implied_correlation, portfolio_loss, tranche_expected_loss)
    >>> loss = portfolio_loss(GaussianCopula(0.25, dim=200), 0.05, lgd=1.0,
    ...                       n=40_000, random_state=1)
    >>> el = tranche_expected_loss(loss, 0.03, 0.07)
    >>> rho = implied_correlation(el, 0.05, 0.03, 0.07, n_names=200, random_state=1)
    >>> bool(abs(rho - 0.25) < 0.06)
    True
    """
    from rcopula.core.elliptical import GaussianCopula

    def mismatch(rho: float) -> float:
        loss = portfolio_loss(
            GaussianCopula(rho, dim=n_names), default_prob, lgd, None, n, random_state
        )
        return tranche_expected_loss(loss, attachment, detachment) - target_expected_loss

    lo, hi = 1e-4, 0.95
    f_lo, f_hi = mismatch(lo), mismatch(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"expected loss {target_expected_loss:.4g} is not attainable for the "
            f"[{attachment}, {detachment}] tranche at any correlation in (0, 1); "
            f"reachable range is roughly "
            f"[{target_expected_loss + min(f_lo, f_hi):.4g}, "
            f"{target_expected_loss + max(f_lo, f_hi):.4g}]"
        )
    return float(optimize.brentq(mismatch, lo, hi, xtol=1e-4))
