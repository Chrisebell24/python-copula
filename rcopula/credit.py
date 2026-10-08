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
   approximation with a flat discount rate, no accrual on default and no
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
    r"""Simulate which borrowers ("names") default before the horizon.

    Each simulated scenario is one possible future: a row of ``True``/``False``
    flags saying which names in the portfolio defaulted. Each name defaults
    with the probability you give it; the copula decides how often names
    default *together*. Use it as the raw building block for loss
    distributions, basket default swaps or your own payoff logic.

    Parameters
    ----------
    copula : Copula
        Any ``rcopula`` copula whose dimension ``d`` equals the number of names.
        It controls the dependence between defaults only, not their marginal
        probabilities.
    default_prob : float or array_like of float, shape (d,)
        Probability that each name defaults by the horizon, each in ``[0, 1]``
        (e.g. ``0.02`` for 2%). A single number applies to every name.
    n : int, default 100_000
        Number of simulated scenarios.
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator for reproducible draws.

    Returns
    -------
    numpy.ndarray of bool, shape (n, d)
        ``True`` where the name defaulted in that scenario.

    Raises
    ------
    ValueError
        If any default probability lies outside ``[0, 1]``, or
        ``default_prob`` has a length other than 1 or ``d``.

    Notes
    -----
    A name defaults when its copula coordinate falls below its default
    probability: :math:`U_i \le p_i`. That is exactly the one-factor threshold
    model when ``copula`` is Gaussian with exchangeable correlation, but works
    for any family.

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
    r"""Simulate *when* each name defaults, in years, with defaults linked by a copula.

    This is the Li (2000) default-time model behind classic CDO pricing. Each
    name keeps its own credit curve (a constant hazard rate, i.e. a constant
    annual default intensity); the copula decides whether early defaults tend
    to cluster. Use it when timing matters, e.g. first-to-default within five
    years: ``(times <= 5.0)``.

    Parameters
    ----------
    copula : Copula
        Any ``rcopula`` copula whose dimension ``d`` equals the number of names.
    hazard_rate : float or array_like of float, shape (d,)
        Constant annual hazard rate of each name, strictly positive (``0.02``
        means roughly a 2% chance of default per year; a common rule of thumb
        is ``spread / (1 - recovery)``). A single number applies to every name.
    n : int, default 100_000
        Number of simulated scenarios.
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator for reproducible draws.

    Returns
    -------
    numpy.ndarray of float, shape (n, d)
        Default time of each name in each scenario, in years (the same time
        unit as ``1 / hazard_rate``). Values are positive and unbounded.

    Raises
    ------
    ValueError
        If any hazard rate is zero or negative, or ``hazard_rate`` has a
        length other than 1 or ``d``.

    Notes
    -----
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
    r"""Simulate the credit portfolio's loss, as a fraction of total exposure.

    Each scenario's loss is the sum, over the names that defaulted, of
    exposure times loss-given-default, divided by total exposure. A result of
    ``0.04`` means the portfolio lost 4% of its notional. Feed the output to
    :func:`tranche_loss`, :func:`tranche_spread` or a VaR/ES function.

    Parameters
    ----------
    copula : Copula
        Any ``rcopula`` copula whose dimension ``d`` equals the number of names.
    default_prob : float or array_like of float, shape (d,)
        Per-name default probability to the horizon, each in ``[0, 1]``. A
        single number applies to every name.
    lgd : float or array_like of float, shape (d,), default 0.6
        Loss given default, as a fraction of exposure (``0.6`` means 40%
        recovery), each in ``[0, 1]``. A single number applies to every name.
    exposure : float or array_like of float, shape (d,), optional
        Per-name exposure (notional), in any currency unit, each finite and
        ``>= 0`` with a positive total. Defaults to equal. Only relative sizes
        matter: losses are returned as a fraction of total exposure, so the
        result always lies in ``[0, 1]``.
    n : int, default 100_000
        Number of simulated scenarios.
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator for reproducible draws.

    Returns
    -------
    numpy.ndarray of float, shape (n,)
        Fractional portfolio loss in each scenario.

    Raises
    ------
    ValueError
        If any default probability or LGD lies outside ``[0, 1]``, any
        exposure is negative or not finite, the exposures sum to zero, or a
        per-name input has a length other than 1 or ``d``.

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
    if not np.all((severity >= 0) & (severity <= 1)):
        raise ValueError("lgd must lie in [0, 1] for every name")
    weight = np.full(d, 1.0) if exposure is None else _as_vector(exposure, d, "exposure")
    if not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("exposure must be finite and >= 0 for every name")
    total = weight.sum()
    if total <= 0:
        raise ValueError("exposures sum to zero; at least one name needs a positive exposure")
    weight = weight / total

    defaulted = default_indicators(copula, p, n, random_state)
    return defaulted @ (weight * severity)


# ======================================================================
# Tranches
# ======================================================================


def tranche_loss(loss: ArrayLike, attachment: float, detachment: float) -> NDArray[np.float64]:
    r"""Share of a tranche's notional wiped out by each portfolio loss.

    A tranche is a slice of the portfolio's losses between an attachment and a
    detachment point. For a 3-7% tranche, a 2% portfolio loss leaves it
    untouched, a 5% loss wipes out half of it, and anything above 7% wipes it
    out completely.

    Parameters
    ----------
    loss : float or array_like of float, shape (n,)
        Portfolio loss(es) as a fraction of total notional, typically the
        output of :func:`portfolio_loss`.
    attachment : float
        Lower edge of the tranche, as a fraction of portfolio notional (e.g.
        ``0.03`` for 3%).
    detachment : float
        Upper edge of the tranche, as a fraction of portfolio notional. Must
        satisfy ``0 <= attachment < detachment <= 1``.

    Returns
    -------
    numpy.ndarray of float, same shape as ``loss``
        Tranche loss as a fraction of tranche notional, each in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``0 <= attachment < detachment <= 1`` does not hold.

    Notes
    -----
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
    """Average share of a tranche's notional lost, across simulated scenarios.

    This is the number that drives a tranche's price: the mean of
    :func:`tranche_loss` over all scenarios. ``0.25`` means the tranche
    expects to lose a quarter of its notional by the horizon.

    Parameters
    ----------
    loss : array_like of float, shape (n,)
        Simulated portfolio losses as fractions of total notional, typically
        the output of :func:`portfolio_loss`.
    attachment : float
        Lower edge of the tranche, as a fraction of portfolio notional.
    detachment : float
        Upper edge of the tranche, as a fraction of portfolio notional. Must
        satisfy ``0 <= attachment < detachment <= 1``.

    Returns
    -------
    float
        Expected tranche loss as a fraction of tranche notional, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``0 <= attachment < detachment <= 1`` does not hold.

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
    r"""Rough fair running spread on a tranche, in basis points per year.

    Turns a tranche's expected loss into the annual premium a protection
    buyer would pay. Use it to compare how much different copulas reprice
    the same tranche, not to quote a trade.

    Parameters
    ----------
    loss : array_like of float, shape (n,)
        Simulated portfolio losses as fractions of total notional, typically
        the output of :func:`portfolio_loss`. They should be losses to
        ``maturity``.
    attachment : float
        Lower edge of the tranche, as a fraction of portfolio notional.
    detachment : float
        Upper edge of the tranche, as a fraction of portfolio notional. Must
        satisfy ``0 <= attachment < detachment <= 1``.
    maturity : float, default 5.0
        Tranche maturity in years. Must be positive.
    discount_rate : float, default 0.0
        Continuously compounded annual rate (``0.03`` for 3%), applied to
        **both** legs. Zero means no discounting. Because both legs are
        discounted on the same schedule the rate only has a second-order
        effect: a higher rate slightly *tightens* the spread (it down-weights
        the late, amortised premium payments).

    Returns
    -------
    float
        Approximate fair spread in basis points per year (``250.0`` means
        2.5% of tranche notional per year).

    Raises
    ------
    ValueError
        If ``maturity`` is not positive, or the attachment/detachment points
        are invalid.

    Notes
    -----
    Equates the protection leg to the premium leg assuming the tranche
    loss accrues **linearly** over ``[0, T]`` on a flat discount curve
    :math:`e^{-rt}`. The protection leg pays :math:`\mathrm{EL}/T` per year;
    the premium leg pays :math:`s` on the outstanding notional
    :math:`1 - \mathrm{EL}\,t/T`. Both are discounted with the same curve:

    .. math::

        s = \frac{(\mathrm{EL}/T)\,A_0}{A_0 - (\mathrm{EL}/T)\,A_1},\qquad
        A_0 = \int_0^T e^{-rt}\,dt,\quad A_1 = \int_0^T t\,e^{-rt}\,dt,

    which reduces to :math:`s = \mathrm{EL} / (T (1 - \mathrm{EL}/2))` at
    :math:`r = 0` (the second factor is the adjustment for notional
    amortising as losses accrue). Real pricing integrates over a default-time distribution with a
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
    # Both legs discounted on the same curve, loss accruing linearly in time.
    # a1_over_a0 is the discount-weighted mean payment time; T/2 at r = 0.
    rt = discount_rate * maturity
    if abs(rt) < 1e-8:
        a1_over_a0 = maturity / 2.0
    else:
        a0 = -np.expm1(-rt) / discount_rate
        a1 = (-np.expm1(-rt) - rt * np.exp(-rt)) / discount_rate**2
        a1_over_a0 = a1 / a0
    rate = el / maturity
    denominator = max(1.0 - rate * a1_over_a0, 1e-12)
    return float(1e4 * rate / denominator)


def nth_to_default_probability(
    copula: Copula,
    default_prob: ArrayLike,
    n_th: int = 1,
    n: int = 100_000,
    random_state: np.random.Generator | int | None = None,
) -> float:
    r"""Probability that at least ``n_th`` names in a basket default by the horizon.

    This is the trigger probability of an ``n_th``-to-default basket swap:
    protection pays out once the ``n_th`` default occurs. ``n_th=1`` is
    first-to-default.

    Parameters
    ----------
    copula : Copula
        Any ``rcopula`` copula whose dimension ``d`` equals the basket size.
    default_prob : float or array_like of float, shape (d,)
        Per-name default probability to the horizon, each in ``[0, 1]``. A
        single number applies to every name.
    n_th : int, default 1
        How many defaults trigger the payout, between 1 and ``d``.
    n : int, default 100_000
        Number of simulated scenarios.
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator for reproducible draws.

    Returns
    -------
    float
        Simulated probability, in ``[0, 1]``, that ``n_th`` or more names
        default.

    Raises
    ------
    ValueError
        If ``n_th`` is outside ``1..d``, or any default probability lies
        outside ``[0, 1]``.

    Notes
    -----
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
    r"""Probability that a very large, uniform loan pool loses at most ``x`` (Vasicek).

    A closed-form answer, with no simulation, to "how likely is my portfolio
    loss to stay below ``x``?" for a pool of many small, identical loans
    whose defaults are linked by a one-factor Gaussian copula. It is the
    formula behind Basel IRB capital, and a check on simulated results.

    Parameters
    ----------
    x : float or array_like of float
        Loss level(s) as a fraction of the pool (losses here assume 100% loss
        given default; to apply an LGD, evaluate at ``x / lgd``).
    default_prob : float
        Default probability of each loan to the horizon, strictly in
        ``(0, 1)``.
    correlation : float
        Asset correlation of the one-factor Gaussian model, strictly in
        ``(0, 1)`` (Basel uses roughly ``0.12`` to ``0.24`` for corporates).

    Returns
    -------
    numpy.ndarray of float, same shape as ``x``
        :math:`P(L \le x)`, each in ``[0, 1]``. A scalar ``x`` gives a 0-d
        array; wrap it in ``float()`` if needed.

    Raises
    ------
    ValueError
        If ``default_prob`` or ``correlation`` is not strictly inside
        ``(0, 1)``.

    Notes
    -----
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
    r"""Find the Gaussian-copula correlation that reproduces a tranche's expected loss.

    The credit market's "implied correlation", analogous to implied
    volatility: the single correlation number that makes the Gaussian copula
    model match an observed tranche price, here expressed as an expected
    loss.

    Parameters
    ----------
    target_expected_loss : float
        Expected tranche loss to match, as a fraction of tranche notional (as
        returned by :func:`tranche_expected_loss`), in ``[0, 1]``.
    default_prob : float
        Default probability of every name to the horizon, in ``[0, 1]``.
    attachment : float
        Lower edge of the tranche, as a fraction of portfolio notional.
    detachment : float
        Upper edge of the tranche, as a fraction of portfolio notional. Must
        satisfy ``0 <= attachment < detachment <= 1``.
    lgd : float, default 1.0
        Loss given default, as a fraction of exposure, for every name.
    n_names : int, default 200
        Number of equally weighted names in the simulated pool.
    n : int, default 40_000
        Number of simulated scenarios per trial correlation.
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator for reproducible results. Whatever is passed, every
        trial correlation reuses the **same** random numbers (common random
        numbers): a ``Generator`` or ``None`` is used once to draw a seed,
        which is then reused for every trial, so the Monte Carlo mismatch is
        a smooth function of the correlation and the root search is stable.

    Returns
    -------
    float
        Implied correlation, found to within about ``1e-4``.

    Raises
    ------
    ValueError
        If the target is unattainable for any correlation in the search
        range ``[0, 0.999]``, or the tranche bounds are invalid.

    Notes
    -----
    The correlation is found by Brent root search over ``[0, 0.999]`` on a
    Monte Carlo estimate, so it carries simulation error, and each call runs
    dozens of full simulations. (Exactly 1 is excluded because the
    equicorrelation matrix is singular there.)

    Inverting the one-factor Gaussian model until it reproduces an observed
    tranche price, tranche by tranche on the same pool, produces the
    **correlation skew** -- different tranches implying different
    correlations, which is a contradiction if the model were right, and is
    the standard evidence that it is not.

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

    # Common random numbers: every trial correlation must see the same draws,
    # otherwise the mismatch is noisy in rho and brentq can wander. An int seed
    # already guarantees that; a Generator / None is used once to fix a seed.
    if isinstance(random_state, (int, np.integer)):
        seed = int(random_state)
    else:
        seed = int(np.random.default_rng(random_state).integers(2**63 - 1))

    def mismatch(rho: float) -> float:
        loss = portfolio_loss(GaussianCopula(rho, dim=n_names), default_prob, lgd, None, n, seed)
        return tranche_expected_loss(loss, attachment, detachment) - target_expected_loss

    lo, hi = 0.0, 0.999
    f_lo, f_hi = mismatch(lo), mismatch(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"expected loss {target_expected_loss:.4g} is not attainable for the "
            f"[{attachment}, {detachment}] tranche at any correlation in [0, 0.999]; "
            f"reachable range is roughly "
            f"[{target_expected_loss + min(f_lo, f_hi):.4g}, "
            f"{target_expected_loss + max(f_lo, f_hi):.4g}]"
        )
    return float(optimize.brentq(mismatch, lo, hi, xtol=1e-4))
