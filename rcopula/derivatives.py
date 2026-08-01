r"""Multi-asset option pricing under copula dependence.

A single correlation number is enough for a multi-asset payoff only when the
joint law is multivariate lognormal. It is not: each underlying has its own
volatility smile, and the smiles say the marginals are not lognormal. Copulas
resolve the tension exactly -- **take each marginal risk-neutral distribution
from its own smile, and supply the dependence separately** -- which is what
:class:`SmileMargin` and :func:`basket_implied_vol` are for.

============================  ==================================================
:func:`lognormal_terminal`    Black-76 terminal distribution, as a margin.
:class:`SmileMargin`          Risk-neutral marginal implied by a volatility smile.
:func:`basket_option`         Option on a weighted basket.
:func:`rainbow_option`        Best-of / worst-of on several assets.
:func:`spread_option`         Option on the difference of two assets.
:func:`black76`               Single-asset closed form.
:func:`margrabe`              Exchange option -- exact, the validation anchor.
:func:`kirk_spread`           Kirk's spread-option approximation.
:func:`implied_volatility`    Invert Black-76 for a quoted price.
:func:`basket_implied_vol`    The basket's own smile, implied by the copula.
============================  ==================================================

Prices are Monte-Carlo unless stated otherwise, and a ``standard_error`` is
returned alongside so the noise is visible rather than assumed away.

.. warning::

   Reference implementations for analysis, not a production pricing library.
   Single flat rate, no dividends beyond what the forward already embeds, no
   early exercise, no calibration to a term structure.

References
----------
Margrabe, W. (1978). The value of an option to exchange one asset for another.
    *Journal of Finance* 33(1), 177-186.
Kirk, E. (1995). Correlation in the energy markets. In *Managing Energy Price
    Risk*. Risk Publications.
Breeden, D. T. and Litzenberger, R. H. (1978). Prices of state-contingent
    claims implicit in option prices. *Journal of Business* 51(4), 621-651.
Cherubini, U., Luciano, E. and Vecchiato, W. (2004). *Copula Methods in
    Finance*. Wiley.
Black, F. (1976). The pricing of commodity contracts.
    *Journal of Financial Economics* 3(1-2), 167-179.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats
from scipy.special import ndtr

from rcopula.core.base import Copula
from rcopula.distribution import CopulaDistribution, Margin

__all__ = [
    "MonteCarloPrice",
    "SmileMargin",
    "basket_implied_vol",
    "basket_option",
    "black76",
    "implied_volatility",
    "kirk_spread",
    "lognormal_terminal",
    "margrabe",
    "rainbow_option",
    "spread_option",
]


class MonteCarloPrice(NamedTuple):
    """A simulated price with its Monte-Carlo standard error."""

    price: float
    standard_error: float
    n: int

    def __repr__(self) -> str:
        return f"MonteCarloPrice(price={self.price:.6g} +/- {self.standard_error:.2g})"


def _discount(rate: float, maturity: float) -> float:
    return float(np.exp(-rate * maturity))


def _mc(payoff: NDArray[np.float64], rate: float, maturity: float) -> MonteCarloPrice:
    df = _discount(rate, maturity)
    n = payoff.size
    return MonteCarloPrice(
        price=float(df * payoff.mean()),
        standard_error=float(df * payoff.std(ddof=1) / np.sqrt(n)),
        n=n,
    )


# ======================================================================
# Marginals
# ======================================================================


def lognormal_terminal(forward: float, vol: float, maturity: float) -> Margin:
    r"""Terminal distribution of an asset under Black-76, as a frozen margin.

    :math:`S_T = F \exp(\sigma\sqrt{T}\,Z - \sigma^2 T/2)`, so
    :math:`\mathbb{E}[S_T] = F` -- the martingale property that makes the
    resulting prices arbitrage-free.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.derivatives import lognormal_terminal
    >>> m = lognormal_terminal(100.0, 0.25, 1.0)
    >>> bool(abs(m.mean() - 100.0) < 1e-9)
    True
    """
    if vol <= 0 or maturity <= 0:
        raise ValueError(f"vol and maturity must be positive, got {vol} and {maturity}")
    sigma = vol * np.sqrt(maturity)
    return stats.lognorm(s=sigma, scale=forward * np.exp(-0.5 * sigma**2))


class SmileMargin:
    r"""Risk-neutral marginal distribution implied by a volatility smile.

    Breeden & Litzenberger (1978): the undiscounted call price determines the
    whole risk-neutral law, with

    .. math::
        F(K) = 1 + e^{rT}\,\frac{\partial C}{\partial K}.

    This is what lets a basket be priced **without assuming its components are
    lognormal**. Each underlying keeps the distribution its own smile implies,
    and the copula supplies the dependence -- which a single correlation number
    cannot do once the marginals are non-lognormal.

    Parameters
    ----------
    strikes : array_like
        Strikes at which the smile is quoted, increasing.
    vols : array_like
        Implied volatilities at those strikes.
    forward, maturity, rate : float
        Forward, time to expiry, and the discount rate.

    Examples
    --------
    A flat smile must reproduce the lognormal margin it came from:

    >>> import numpy as np
    >>> from rcopula.derivatives import SmileMargin, lognormal_terminal
    >>> strikes = np.linspace(50, 200, 60)
    >>> flat = SmileMargin(strikes, np.full(60, 0.25), forward=100.0, maturity=1.0)
    >>> exact = lognormal_terminal(100.0, 0.25, 1.0)
    >>> bool(abs(flat.cdf(110.0) - exact.cdf(110.0)) < 0.01)
    True

    A downward-sloping smile puts more mass in the left tail, as it should:

    >>> skewed = SmileMargin(strikes, 0.25 + 0.0007 * (100 - strikes),
    ...                      forward=100.0, maturity=1.0)
    >>> bool(skewed.cdf(70.0) > flat.cdf(70.0))
    True
    """

    def __init__(
        self,
        strikes: ArrayLike,
        vols: ArrayLike,
        forward: float,
        maturity: float,
        rate: float = 0.0,
    ) -> None:
        k = np.asarray(strikes, dtype=np.float64).ravel()
        v = np.asarray(vols, dtype=np.float64).ravel()
        if k.size != v.size:
            raise ValueError(f"got {k.size} strikes and {v.size} vols")
        if k.size < 4:
            raise ValueError("need at least four quoted strikes to differentiate the smile")
        if np.any(np.diff(k) <= 0):
            raise ValueError("strikes must be strictly increasing")

        self.strikes, self.vols = k, v
        self.forward, self.maturity, self.rate = float(forward), float(maturity), float(rate)

        calls = np.array(
            [black76(forward, kk, vv, maturity, rate) for kk, vv in zip(k, v, strict=True)]
        )
        # F(K) = 1 + e^{rT} dC/dK, by Breeden-Litzenberger.
        slope = np.gradient(calls, k)
        cdf = np.clip(1.0 + np.exp(rate * maturity) * slope, 0.0, 1.0)
        # Enforce monotonicity: numerical differentiation of a quoted smile is
        # not guaranteed to give a valid distribution function.
        self._cdf_grid = np.maximum.accumulate(cdf)
        self._k_grid = k

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Risk-neutral distribution function."""
        return np.interp(np.asarray(x, dtype=np.float64), self._k_grid, self._cdf_grid)

    def ppf(self, q: ArrayLike) -> NDArray[np.float64]:
        """Risk-neutral quantile function, by inverting the interpolated CDF."""
        qq = np.asarray(q, dtype=np.float64)
        # np.interp needs an increasing x; ties from the monotone fix are fine.
        return np.interp(qq, self._cdf_grid, self._k_grid)

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Risk-neutral density, the second derivative of the call price."""
        density = np.gradient(self._cdf_grid, self._k_grid)
        return np.interp(np.asarray(x, dtype=np.float64), self._k_grid, np.maximum(density, 0.0))

    def __repr__(self) -> str:
        return (
            f"SmileMargin(forward={self.forward:g}, maturity={self.maturity:g}, "
            f"{self.strikes.size} strikes)"
        )


# ======================================================================
# Closed forms
# ======================================================================


def black76(
    forward: float,
    strike: float,
    vol: float,
    maturity: float,
    rate: float = 0.0,
    kind: str = "call",
) -> float:
    r"""Black-76 price of a European option on a forward.

    Examples
    --------
    >>> from rcopula.derivatives import black76
    >>> float(round(black76(100.0, 100.0, 0.2, 1.0), 6))
    7.965567

    Put-call parity holds:

    >>> c = black76(100.0, 90.0, 0.2, 1.0)
    >>> p = black76(100.0, 90.0, 0.2, 1.0, kind="put")
    >>> bool(abs((c - p) - (100.0 - 90.0)) < 1e-10)
    True
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    df = _discount(rate, maturity)
    if vol <= 0 or maturity <= 0:
        intrinsic = max(forward - strike, 0.0) if kind == "call" else max(strike - forward, 0.0)
        return float(df * intrinsic)

    sigma = vol * np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * sigma**2) / sigma
    d2 = d1 - sigma
    if kind == "call":
        return float(df * (forward * ndtr(d1) - strike * ndtr(d2)))
    return float(df * (strike * ndtr(-d2) - forward * ndtr(-d1)))


def margrabe(
    forward1: float,
    forward2: float,
    vol1: float,
    vol2: float,
    correlation: float,
    maturity: float,
    rate: float = 0.0,
) -> float:
    r"""Exact price of an option to exchange asset 2 for asset 1.

    Payoff :math:`\max(S_1 - S_2, 0)`. Margrabe (1978) showed this is
    Black-Scholes with the *spread* volatility

    .. math::  \sigma^2 = \sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2,

    and no strike. Because it is exact under jointly lognormal dynamics -- which
    is precisely a Gaussian copula with lognormal margins -- it is the
    validation anchor for :func:`spread_option`.

    Examples
    --------
    >>> from rcopula.derivatives import margrabe
    >>> float(round(margrabe(100.0, 95.0, 0.2, 0.3, 0.5, 1.0), 6))
    12.952273

    Perfect correlation with equal vols leaves only the intrinsic difference:

    >>> float(round(margrabe(100.0, 95.0, 0.25, 0.25, 1.0, 1.0), 10))
    5.0
    """
    sigma_sq = vol1**2 + vol2**2 - 2.0 * correlation * vol1 * vol2
    if sigma_sq <= 0:
        return float(_discount(rate, maturity) * max(forward1 - forward2, 0.0))
    sigma = np.sqrt(sigma_sq * maturity)
    d1 = (np.log(forward1 / forward2) + 0.5 * sigma**2) / sigma
    return float(_discount(rate, maturity) * (forward1 * ndtr(d1) - forward2 * ndtr(d1 - sigma)))


def kirk_spread(
    forward1: float,
    forward2: float,
    strike: float,
    vol1: float,
    vol2: float,
    correlation: float,
    maturity: float,
    rate: float = 0.0,
) -> float:
    r"""Kirk's (1995) approximation for a spread option, :math:`\max(S_1-S_2-K, 0)`.

    Treats :math:`S_2 + K` as a single lognormal asset, which is exact at
    :math:`K = 0` (where it reduces to :func:`margrabe`) and stays accurate
    for moderate strikes. Widely used in energy markets, where spread options
    are the standard product.

    Examples
    --------
    At zero strike it coincides with Margrabe:

    >>> from rcopula.derivatives import kirk_spread, margrabe
    >>> a = kirk_spread(100.0, 95.0, 0.0, 0.2, 0.3, 0.5, 1.0)
    >>> b = margrabe(100.0, 95.0, 0.2, 0.3, 0.5, 1.0)
    >>> bool(abs(a - b) < 1e-10)
    True
    """
    adjusted = forward2 + strike
    weight = forward2 / adjusted if adjusted != 0 else 0.0
    vol2_eff = vol2 * weight
    sigma_sq = vol1**2 + vol2_eff**2 - 2.0 * correlation * vol1 * vol2_eff
    if sigma_sq <= 0:
        return float(_discount(rate, maturity) * max(forward1 - adjusted, 0.0))
    sigma = np.sqrt(sigma_sq * maturity)
    d1 = (np.log(forward1 / adjusted) + 0.5 * sigma**2) / sigma
    return float(_discount(rate, maturity) * (forward1 * ndtr(d1) - adjusted * ndtr(d1 - sigma)))


def implied_volatility(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    rate: float = 0.0,
    kind: str = "call",
) -> float:
    """Invert Black-76 for the volatility matching a quoted price.

    Examples
    --------
    >>> from rcopula.derivatives import black76, implied_volatility
    >>> p = black76(100.0, 110.0, 0.27, 1.5)
    >>> float(round(implied_volatility(p, 100.0, 110.0, 1.5), 10))
    0.27
    """
    df = _discount(rate, maturity)
    intrinsic = df * (max(forward - strike, 0.0) if kind == "call" else max(strike - forward, 0.0))
    if price <= intrinsic + 1e-14:
        return 0.0

    def gap(vol: float) -> float:
        return black76(forward, strike, vol, maturity, rate, kind) - price

    try:
        return float(optimize.brentq(gap, 1e-8, 10.0, xtol=1e-12))
    except ValueError as exc:
        raise ValueError(
            f"price {price:.6g} is not attainable for any volatility "
            f"(intrinsic value is {intrinsic:.6g})"
        ) from exc


# ======================================================================
# Multi-asset payoffs
# ======================================================================


def _terminal_prices(
    copula: Copula,
    margins: Margin | list[Margin],
    n: int,
    random_state: np.random.Generator | int | None,
) -> NDArray[np.float64]:
    joint = CopulaDistribution(copula, margins)
    return np.asarray(joint.rvs(n, random_state=random_state), dtype=np.float64)


def basket_option(
    copula: Copula,
    margins: Margin | list[Margin],
    strike: float,
    maturity: float,
    weights: ArrayLike | None = None,
    rate: float = 0.0,
    kind: str = "call",
    n: int = 200_000,
    random_state: np.random.Generator | int | None = None,
) -> MonteCarloPrice:
    r"""Option on a weighted basket, :math:`\max(\sum_i w_i S_i - K, 0)`.

    The basket is where the dependence model earns its keep. A basket is *less*
    volatile than its components, by an amount that depends entirely on how they
    co-move -- and on how they co-move **in the tail**, which correlation alone
    does not capture.

    Examples
    --------
    >>> from rcopula import GaussianCopula
    >>> from rcopula.derivatives import basket_option, lognormal_terminal
    >>> margins = [lognormal_terminal(100.0, 0.25, 1.0)] * 3
    >>> price = basket_option(GaussianCopula(0.4, dim=3), margins, 100.0, 1.0,
    ...                       n=80_000, random_state=0)
    >>> bool(price.price > 0 and price.standard_error < 0.2)
    True

    Higher dependence means less diversification and a more valuable option:

    >>> low = basket_option(GaussianCopula(0.0, dim=3), margins, 100.0, 1.0,
    ...                     n=80_000, random_state=0).price
    >>> high = basket_option(GaussianCopula(0.9, dim=3), margins, 100.0, 1.0,
    ...                      n=80_000, random_state=0).price
    >>> bool(high > low)
    True
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    d = copula.dim
    w = np.full(d, 1.0 / d) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.size != d:
        raise ValueError(f"got {w.size} weight(s) for a copula of dimension {d}")

    basket = _terminal_prices(copula, margins, n, random_state) @ w
    payoff = (
        np.maximum(basket - strike, 0.0) if kind == "call" else np.maximum(strike - basket, 0.0)
    )
    return _mc(payoff, rate, maturity)


def rainbow_option(
    copula: Copula,
    margins: Margin | list[Margin],
    strike: float,
    maturity: float,
    rate: float = 0.0,
    kind: str = "call",
    on: str = "best",
    n: int = 200_000,
    random_state: np.random.Generator | int | None = None,
) -> MonteCarloPrice:
    r"""Best-of or worst-of option on several assets.

    Payoff :math:`\max(\max_i S_i - K, 0)` for ``on="best"``, or with
    :math:`\min_i` for ``on="worst"``.

    These are the payoffs most sensitive to dependence, and in opposite
    directions: a best-of is worth most when the assets are *independent*
    (many chances for one to finish high), a worst-of when they move
    *together* (less chance that any one drags the minimum down).

    Examples
    --------
    >>> from rcopula import GaussianCopula
    >>> from rcopula.derivatives import lognormal_terminal, rainbow_option
    >>> margins = [lognormal_terminal(100.0, 0.3, 1.0)] * 3
    >>> free, tied = GaussianCopula(0.0, dim=3), GaussianCopula(0.9, dim=3)
    >>> best_free = rainbow_option(free, margins, 100.0, 1.0, n=80_000,
    ...                            random_state=0, on="best").price
    >>> best_tied = rainbow_option(tied, margins, 100.0, 1.0, n=80_000,
    ...                            random_state=0, on="best").price
    >>> bool(best_free > best_tied)
    True

    and the worst-of ordering is reversed:

    >>> worst_free = rainbow_option(free, margins, 100.0, 1.0, n=80_000,
    ...                             random_state=0, on="worst").price
    >>> worst_tied = rainbow_option(tied, margins, 100.0, 1.0, n=80_000,
    ...                             random_state=0, on="worst").price
    >>> bool(worst_tied > worst_free)
    True
    """
    if on not in ("best", "worst"):
        raise ValueError(f"on must be 'best' or 'worst', got {on!r}")
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    prices = _terminal_prices(copula, margins, n, random_state)
    chosen = prices.max(axis=1) if on == "best" else prices.min(axis=1)
    payoff = (
        np.maximum(chosen - strike, 0.0) if kind == "call" else np.maximum(strike - chosen, 0.0)
    )
    return _mc(payoff, rate, maturity)


def spread_option(
    copula: Copula,
    margins: list[Margin],
    strike: float,
    maturity: float,
    rate: float = 0.0,
    n: int = 200_000,
    random_state: np.random.Generator | int | None = None,
) -> MonteCarloPrice:
    r"""Option on the spread between two assets, :math:`\max(S_1 - S_2 - K, 0)`.

    At :math:`K = 0` with a Gaussian copula and lognormal margins this is the
    Margrabe exchange option, which has an exact price -- so
    :func:`margrabe` is the check that this simulation is right.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import GaussianCopula
    >>> from rcopula.derivatives import lognormal_terminal, margrabe, spread_option
    >>> margins = [lognormal_terminal(100.0, 0.2, 1.0), lognormal_terminal(95.0, 0.3, 1.0)]
    >>> mc = spread_option(GaussianCopula(0.5), margins, 0.0, 1.0,
    ...                    n=400_000, random_state=0)
    >>> exact = margrabe(100.0, 95.0, 0.2, 0.3, 0.5, 1.0)
    >>> bool(abs(mc.price - exact) < 4 * mc.standard_error)
    True
    """
    if copula.dim != 2:
        raise ValueError(f"a spread option is bivariate; got dim={copula.dim}")
    prices = _terminal_prices(copula, margins, n, random_state)
    payoff = np.maximum(prices[:, 0] - prices[:, 1] - strike, 0.0)
    return _mc(payoff, rate, maturity)


def basket_implied_vol(
    copula: Copula,
    margins: Margin | list[Margin],
    strikes: ArrayLike,
    maturity: float,
    weights: ArrayLike | None = None,
    rate: float = 0.0,
    n: int = 200_000,
    random_state: np.random.Generator | int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""The basket's own implied-volatility smile, implied by the copula.

    Prices the basket at each strike, then inverts Black-76 to express the
    result as an implied volatility. This is the calculation a single
    correlation number cannot do: give each component the marginal its **own
    smile** implies (see :class:`SmileMargin`), choose a dependence structure,
    and the basket smile falls out.

    Returns
    -------
    strikes, vols : ndarray
        The input strikes and the implied volatility at each.

    Examples
    --------
    Lognormal components under a Gaussian copula give a nearly flat basket
    smile -- the basket is then close to lognormal itself:

    >>> import numpy as np
    >>> from rcopula import GaussianCopula
    >>> from rcopula.derivatives import basket_implied_vol, lognormal_terminal
    >>> margins = [lognormal_terminal(100.0, 0.25, 1.0)] * 3
    >>> k, v = basket_implied_vol(GaussianCopula(0.5, dim=3), margins,
    ...                           [90, 100, 110], 1.0, n=200_000, random_state=0)
    >>> bool(v.std() < 0.02)
    True

    A tail-dependent copula bends it, because the basket is no longer lognormal:

    >>> from rcopula import GumbelCopula
    >>> _, vg = basket_implied_vol(GumbelCopula.from_tau(1 / 3, dim=3), margins,
    ...                            [90, 100, 110], 1.0, n=200_000, random_state=0)
    >>> bool(vg.std() > v.std())
    True
    """
    d = copula.dim
    w = np.full(d, 1.0 / d) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))

    basket = _terminal_prices(copula, margins, n, random_state) @ w
    forward = float(basket.mean())

    vols = np.empty(k.size)
    for i, strike in enumerate(k):
        price = _mc(np.maximum(basket - strike, 0.0), rate, maturity).price
        vols[i] = implied_volatility(price, forward, float(strike), maturity, rate)
    return k, vols
