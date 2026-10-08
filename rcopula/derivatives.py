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
:func:`cms_spread_option`     Option on a spread of two CMS rates.
:func:`cms_convexity_adjustment`  The CMS convexity correction.
:func:`cms_margin`            A convexity-adjusted CMS rate, as a margin.
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
Hull, J. C. (2018). *Options, Futures, and Other Derivatives*, 10th ed.,
    ch. 34. The convexity adjustment used by
    :func:`cms_convexity_adjustment`.
Hagan, P. S. (2003). Convexity conundrums: pricing CMS swaps, caps, and floors.
    *Wilmott Magazine*, March, 38-44.
    The replication view of the same adjustment.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats
from scipy.special import log_ndtr, ndtr, ndtri_exp

from rcopula.core.base import Copula
from rcopula.distribution import CopulaDistribution, Margin

__all__ = [
    "CmsLeg",
    "MonteCarloPrice",
    "SmileMargin",
    "basket_implied_vol",
    "basket_option",
    "black76",
    "cms_convexity_adjustment",
    "cms_margin",
    "cms_spread_option",
    "implied_volatility",
    "kirk_spread",
    "lognormal_terminal",
    "margrabe",
    "rainbow_option",
    "spread_option",
]


class MonteCarloPrice(NamedTuple):
    """A simulated option price together with how noisy that estimate is.

    Every Monte-Carlo pricer in this module returns one of these. Treat
    ``price +/- 2 * standard_error`` as a rough 95% band: if two prices differ
    by less than a couple of standard errors, the simulation cannot tell them
    apart. Increase ``n`` in the pricer to shrink the error (it falls like
    ``1 / sqrt(n)``).

    Being a ``NamedTuple``, it unpacks as ``price, se, n = result``.

    Attributes
    ----------
    price : float
        Discounted present value of the option, in the same currency units as
        the underlying prices (times the notional, where there is one).
    standard_error : float
        Monte-Carlo standard error of ``price``, same units. Measures
        simulation noise only, not model error.
    n : int
        Number of simulated scenarios the estimate is based on.
    """

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
    r"""The distribution of an asset's price at expiry, assuming a flat Black-76 vol.

    Use this as a margin (one per underlying) for the multi-asset pricers in
    this module when you have a single at-the-money volatility per asset and
    no smile. If you do have a smile, use :class:`SmileMargin` instead.

    Parameters
    ----------
    forward : float
        Forward price of the asset for delivery at ``maturity`` (currency
        units). The distribution's mean equals this.
    vol : float
        Annualised Black-76 (lognormal) volatility as a decimal, e.g. ``0.25``
        for 25%. Must be positive.
    maturity : float
        Time to expiry in years. Must be positive.

    Returns
    -------
    scipy.stats frozen distribution
        A frozen ``scipy.stats.lognorm`` for the terminal price :math:`S_T`,
        with ``.cdf``, ``.ppf``, ``.rvs``, ``.mean`` etc.

    Raises
    ------
    ValueError
        If ``vol`` or ``maturity`` is not positive.

    Notes
    -----
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
    r"""The distribution of an asset's price at expiry, read off its volatility smile.

    Option prices across strikes tell you the market's (risk-neutral)
    probability of the asset finishing at each level. This class turns a quoted
    smile into that distribution, so a basket or rainbow option can be priced
    with each underlying keeping the skew its own options show, instead of
    assuming it is lognormal. Pass instances as ``margins`` to
    :func:`basket_option`, :func:`rainbow_option`, :func:`basket_implied_vol`
    and friends.

    Parameters
    ----------
    strikes : array_like of float, shape (k,)
        Strikes at which the smile is quoted, strictly increasing, in the same
        currency units as ``forward``. At least four are needed. Outside the
        quoted range the tails are extrapolated (see ``tails`` and Notes).
    vols : array_like of float, shape (k,)
        Black-76 implied volatilities at those strikes, as decimals (``0.25``
        for 25%).
    forward : float
        Forward price of the asset for delivery at ``maturity``.
    maturity : float
        Time to expiry in years.
    rate : float, default 0.0
        Continuously compounded discount rate, as a decimal.
    tails : {"lognormal", "flat"}, default "lognormal"
        How to extend the distribution beyond the quoted strikes.
        ``"lognormal"`` attaches lognormal tails using the implied vol at the
        nearest end strike, so quantiles can fall outside the quoted range.
        ``"flat"`` holds the CDF constant outside the range (the behaviour
        of earlier versions): all tail mass sits on the end strikes and
        :meth:`ppf` never leaves ``[strikes[0], strikes[-1]]``.

    Attributes
    ----------
    strikes : numpy.ndarray of float, shape (k,)
        The quoted strikes.
    vols : numpy.ndarray of float, shape (k,)
        The quoted implied vols.
    forward : float
        Forward price.
    maturity : float
        Time to expiry in years.
    rate : float
        Discount rate.

    Raises
    ------
    ValueError
        If ``strikes`` and ``vols`` differ in length, fewer than four strikes
        are given, the strikes are not strictly increasing, ``forward`` or
        ``maturity`` is not positive, or ``tails`` is unknown.

    Notes
    -----
    Breeden & Litzenberger (1978): the undiscounted call price determines the
    whole risk-neutral law, with

    .. math::
        F(K) = 1 + e^{rT}\,\frac{\partial C}{\partial K}.

    This is what lets a basket be priced **without assuming its components are
    lognormal**. Each underlying keeps the distribution its own smile implies,
    and the copula supplies the dependence -- which a single correlation number
    cannot do once the marginals are non-lognormal.

    The derivative is taken numerically on the quoted grid and the result is
    clipped to [0, 1] and forced to be non-decreasing. Between strikes the CDF
    is linearly interpolated.

    Beyond the quoted range (``tails="lognormal"``), the left tail is a
    lognormal with the forward and the implied vol of the lowest strike,
    rescaled so the CDF is continuous there:
    :math:`F(x) = F(K_1)\,G_L(x)/G_L(K_1)` for :math:`x < K_1`. The right tail
    is the mirror image using the highest strike's vol:
    :math:`1 - F(x) = (1 - F(K_k))\,\bar G_R(x)/\bar G_R(K_k)`. Holding the
    end-strike vol flat in the wings is the usual simple extrapolation; it
    keeps the CDF continuous and monotone and puts no mass on the end strikes
    themselves. With ``tails="flat"`` the CDF is held constant outside the
    range instead.

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
        tails: str = "lognormal",
    ) -> None:
        if tails not in ("lognormal", "flat"):
            raise ValueError(f"tails must be 'lognormal' or 'flat', got {tails!r}")
        if not (forward > 0 and maturity > 0):
            raise ValueError(f"forward and maturity must be positive, got {forward} and {maturity}")
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
        self.tails = tails
        # Lognormal wing parameters: total vol at each end strike, and the
        # log-CDF / log-survival of that wing at its anchor strike.
        self._s_lo = float(v[0] * np.sqrt(maturity))
        self._s_hi = float(v[-1] * np.sqrt(maturity))
        self._lo_anchor = float(log_ndtr(self._d(k[0], self._s_lo)))
        self._hi_anchor = float(log_ndtr(-self._d(k[-1], self._s_hi)))

    def _d(self, x: ArrayLike, s: float) -> NDArray[np.float64]:
        """Standardised log-moneyness of ``x`` under a lognormal with total vol ``s``."""
        return np.asarray((np.log(np.asarray(x) / self.forward) + 0.5 * s * s) / s)

    def _left_mass(self) -> float:
        return float(self._cdf_grid[0])

    def _right_mass(self) -> float:
        return float(1.0 - self._cdf_grid[-1])

    def cdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """Probability that the asset finishes at or below ``x`` (risk-neutral CDF).

        Parameters
        ----------
        x : array_like of float
            Price level(s) at expiry, any shape.

        Returns
        -------
        numpy.ndarray of float, same shape as ``x``
            Probabilities in [0, 1].
        """
        xx = np.asarray(x, dtype=np.float64)
        out = np.asarray(np.interp(xx, self._k_grid, self._cdf_grid), dtype=np.float64)
        if self.tails == "flat":
            return out[()]
        lo, hi = self._k_grid[0], self._k_grid[-1]
        left = xx < lo
        if np.any(left) and self._left_mass() > 0:
            xl = xx[left]
            with np.errstate(divide="ignore"):
                ratio = np.exp(log_ndtr(self._d(np.maximum(xl, 0.0), self._s_lo)) - self._lo_anchor)
            out[left] = self._left_mass() * np.where(xl > 0, ratio, 0.0)
        elif np.any(left):
            out[left] = 0.0
        right = xx > hi
        if np.any(right):
            ratio = np.exp(log_ndtr(-self._d(xx[right], self._s_hi)) - self._hi_anchor)
            out[right] = 1.0 - self._right_mass() * ratio
        return out[()]

    def ppf(self, q: ArrayLike) -> NDArray[np.float64]:
        """The price level the asset finishes below with probability ``q`` (quantile).

        This is the inverse of :meth:`cdf`, computed by inverting the
        interpolated CDF. It is what the copula machinery calls to turn
        uniform draws into prices.

        Parameters
        ----------
        q : array_like of float
            Probabilities in [0, 1], any shape.

        Returns
        -------
        numpy.ndarray of float, same shape as ``q``
            Price levels. With ``tails="lognormal"`` these extend below
            ``strikes[0]`` (towards 0 as ``q -> 0``) and above ``strikes[-1]``
            (to ``inf`` at ``q = 1``); with ``tails="flat"`` they stay within
            ``[strikes[0], strikes[-1]]``.
        """
        qq = np.asarray(q, dtype=np.float64)
        # np.interp needs an increasing x; ties from the monotone fix are fine.
        out = np.asarray(np.interp(qq, self._cdf_grid, self._k_grid), dtype=np.float64)
        if self.tails == "flat":
            return out[()]
        f_lo, f_hi = self._cdf_grid[0], self._cdf_grid[-1]
        left = qq < f_lo
        if np.any(left):
            with np.errstate(divide="ignore"):
                logp = np.log(np.maximum(qq[left], 0.0)) - np.log(f_lo) + self._lo_anchor
            d = ndtri_exp(np.minimum(logp, 0.0))
            out[left] = self.forward * np.exp(self._s_lo * d - 0.5 * self._s_lo**2)
        right = qq > f_hi
        if np.any(right):
            with np.errstate(divide="ignore"):
                logp = (
                    np.log(np.maximum(1.0 - qq[right], 0.0)) - np.log(1.0 - f_hi) + self._hi_anchor
                )
            d = -ndtri_exp(np.minimum(logp, 0.0))
            with np.errstate(over="ignore"):
                out[right] = self.forward * np.exp(self._s_hi * d - 0.5 * self._s_hi**2)
        return out[()]

    def pdf(self, x: ArrayLike) -> NDArray[np.float64]:
        """How likely each price level at expiry is (risk-neutral density).

        Equivalently, the (undiscounted) second derivative of the call price
        with respect to strike. Computed numerically from the CDF grid and
        floored at zero.

        Parameters
        ----------
        x : array_like of float
            Price level(s) at expiry, any shape.

        Returns
        -------
        numpy.ndarray of float, same shape as ``x``
            Non-negative density values (probability per unit of price).
        """
        xx = np.asarray(x, dtype=np.float64)
        density = np.gradient(self._cdf_grid, self._k_grid)
        out = np.asarray(np.interp(xx, self._k_grid, np.maximum(density, 0.0)), dtype=np.float64)
        lo, hi = self._k_grid[0], self._k_grid[-1]
        outside = (xx < lo) | (xx > hi)
        if self.tails == "flat":
            out[outside] = 0.0
            return out[()]
        left = (xx < lo) & (xx > 0)
        if np.any(left):
            xl = xx[left]
            d = self._d(xl, self._s_lo)
            log_g = -0.5 * d * d - 0.5 * np.log(2 * np.pi) - np.log(xl * self._s_lo)
            out[left] = self._left_mass() * np.exp(log_g - self._lo_anchor)
        out[xx <= 0] = 0.0
        right = xx > hi
        if np.any(right):
            xr = xx[right]
            d = self._d(xr, self._s_hi)
            log_g = -0.5 * d * d - 0.5 * np.log(2 * np.pi) - np.log(xr * self._s_hi)
            out[right] = self._right_mass() * np.exp(log_g - self._hi_anchor)
        return out[()]

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
    r"""Price of a plain European call or put on one asset, using the Black-76 formula.

    The single-asset building block: quote a forward, a strike and a vol, get
    a price. Used here to check the multi-asset pricers and to convert prices
    to implied vols.

    Parameters
    ----------
    forward : float
        Forward price of the underlying for delivery at ``maturity``.
    strike : float
        Option strike, same units as ``forward``.
    vol : float
        Annualised Black-76 (lognormal) volatility as a decimal. If zero or
        negative, the discounted intrinsic value is returned.
    maturity : float
        Time to expiry in years. If zero or negative, the discounted intrinsic
        value is returned.
    rate : float, default 0.0
        Continuously compounded discount rate, as a decimal.
    kind : {"call", "put"}, default "call"
        Option type.

    Returns
    -------
    float
        Discounted option price, in the units of ``forward``.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"call"`` or ``"put"``.

    Notes
    -----
    :math:`C = e^{-rT}[F N(d_1) - K N(d_2)]` with
    :math:`d_{1,2} = [\ln(F/K) \pm \sigma^2 T/2] / (\sigma\sqrt{T})`
    (Black, 1976).

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
    r"""Exact price of the right to swap asset 2 for asset 1 at expiry (Margrabe formula).

    The payoff is whatever asset 1 is worth above asset 2, or nothing:
    :math:`\max(S_1 - S_2, 0)`, a spread option with zero strike. It is exact
    when both prices are lognormal with a constant correlation, which makes it
    the benchmark for checking :func:`spread_option`.

    Parameters
    ----------
    forward1 : float
        Forward price of asset 1 (the one you receive).
    forward2 : float
        Forward price of asset 2 (the one you give up), same units.
    vol1 : float
        Annualised lognormal volatility of asset 1, as a decimal.
    vol2 : float
        Annualised lognormal volatility of asset 2, as a decimal.
    correlation : float
        Correlation of the two assets' log-returns, in [-1, 1].
    maturity : float
        Time to expiry in years.
    rate : float, default 0.0
        Continuously compounded discount rate, as a decimal.

    Returns
    -------
    float
        Discounted option price, in the units of the forwards. If the spread
        volatility is zero, the discounted intrinsic value.

    Notes
    -----
    Margrabe (1978) showed this is
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
    r"""Fast approximate price of a spread option with a strike (Kirk's formula).

    Payoff :math:`\max(S_1 - S_2 - K, 0)`: pays when asset 1 beats asset 2 by
    more than ``strike``. The standard closed-form quick price for crack,
    spark and calendar spreads; compare it with :func:`spread_option` to see
    what a non-Gaussian dependence changes.

    Parameters
    ----------
    forward1 : float
        Forward price of the long asset.
    forward2 : float
        Forward price of the short asset, same units.
    strike : float
        Spread strike, same units. ``0`` reproduces :func:`margrabe`.
    vol1 : float
        Annualised lognormal volatility of asset 1, as a decimal.
    vol2 : float
        Annualised lognormal volatility of asset 2, as a decimal.
    correlation : float
        Correlation of the two assets' log-returns, in [-1, 1].
    maturity : float
        Time to expiry in years.
    rate : float, default 0.0
        Continuously compounded discount rate, as a decimal.

    Returns
    -------
    float
        Approximate discounted option price, in the units of the forwards.

    Notes
    -----
    Kirk (1995) treats :math:`S_2 + K` as a single lognormal asset, which is exact at
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
    """The volatility that makes Black-76 reproduce a given option price.

    The usual way to quote an option price as a vol. Here it is also how
    :func:`basket_implied_vol` turns simulated basket prices into a smile.

    Parameters
    ----------
    price : float
        Discounted option price to match, in the units of ``forward``.
    forward : float
        Forward price of the underlying.
    strike : float
        Option strike, same units.
    maturity : float
        Time to expiry in years.
    rate : float, default 0.0
        Continuously compounded discount rate, as a decimal.
    kind : {"call", "put"}, default "call"
        Option type of the quoted price.

    Returns
    -------
    float
        Annualised implied volatility as a decimal. Returns ``0.0`` when the
        price is at or below discounted intrinsic value.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"call"`` or ``"put"``, or no volatility in
        (1e-8, 10] reproduces ``price`` -- typically because the price exceeds
        the no-arbitrage upper bound.

    Notes
    -----
    Solved with Brent's method on ``black76(vol) - price``.

    Examples
    --------
    >>> from rcopula.derivatives import black76, implied_volatility
    >>> p = black76(100.0, 110.0, 0.27, 1.5)
    >>> float(round(implied_volatility(p, 100.0, 110.0, 1.5), 10))
    0.27
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
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
    r"""Monte-Carlo price of a call or put on a weighted basket of assets.

    Payoff :math:`\max(\sum_i w_i S_i - K, 0)` for a call. Each asset's price
    at expiry comes from its own margin; the copula decides how they move
    together. Swap the copula (e.g. Gaussian for Student-t or Gumbel) to see
    how much of the price is the dependence assumption.

    Parameters
    ----------
    copula : Copula
        Dependence between the ``d`` underlyings, e.g.
        ``GaussianCopula(0.4, dim=3)``.
    margins : scipy.stats frozen distribution or list of them, length d
        Terminal price distribution of each underlying, e.g. from
        :func:`lognormal_terminal` or :class:`SmileMargin`. A single margin is
        reused for every asset.
    strike : float
        Basket strike, in the units of the weighted basket value.
    maturity : float
        Time to expiry in years (used only for discounting).
    weights : array_like of float, shape (d,), optional
        Basket weights (number of units of each asset). Default is equal
        weights ``1 / d``.
    rate : float, default 0.0
        Continuously compounded discount rate, as a decimal.
    kind : {"call", "put"}, default "call"
        Option type.
    n : int, default 200_000
        Number of simulated scenarios.
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator, for reproducible prices.

    Returns
    -------
    MonteCarloPrice
        ``(price, standard_error, n)``: discounted price, its simulation
        standard error, and the scenario count.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"call"`` or ``"put"``, or ``weights`` does not
        have one entry per copula dimension.

    Notes
    -----
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
    r"""Monte-Carlo price of an option on the best (or worst) performer of several assets.

    Payoff :math:`\max(\max_i S_i - K, 0)` for ``on="best"``, or with
    :math:`\min_i` for ``on="worst"`` (calls; puts flip the sign). Worst-of
    structures are common in structured notes, so this is where an
    over-optimistic dependence assumption is most expensive.

    Parameters
    ----------
    copula : Copula
        Dependence between the ``d`` underlyings.
    margins : scipy.stats frozen distribution or list of them, length d
        Terminal price distribution of each underlying. A single margin is
        reused for every asset. For best/worst comparisons to make sense the
        assets should be on a common scale (e.g. all with the same forward, or
        normalised to performance).
    strike : float
        Strike applied to the best or worst terminal price.
    maturity : float
        Time to expiry in years (used only for discounting).
    rate : float, default 0.0
        Continuously compounded discount rate, as a decimal.
    kind : {"call", "put"}, default "call"
        Option type.
    on : {"best", "worst"}, default "best"
        Whether the payoff references the highest or lowest terminal price.
    n : int, default 200_000
        Number of simulated scenarios.
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator, for reproducible prices.

    Returns
    -------
    MonteCarloPrice
        ``(price, standard_error, n)``.

    Raises
    ------
    ValueError
        If ``on`` is not ``"best"``/``"worst"`` or ``kind`` is not
        ``"call"``/``"put"``.

    Notes
    -----
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
    kind: str = "call",
) -> MonteCarloPrice:
    r"""Monte-Carlo price of a call or put on the spread between two assets.

    Call payoff :math:`\max(S_1 - S_2 - K, 0)`: pays when asset 1 beats asset
    2 by more than the strike (e.g. a crack or spark spread). Put payoff
    :math:`\max(K - (S_1 - S_2), 0)`. Unlike :func:`kirk_spread`, any margins
    and any copula can be used.

    Parameters
    ----------
    copula : Copula
        Bivariate (``dim=2``) dependence between the two assets.
    margins : list of scipy.stats frozen distributions, length 2
        Terminal price distributions of asset 1 (long) and asset 2 (short).
    strike : float
        Spread strike, in price units.
    maturity : float
        Time to expiry in years (used only for discounting).
    rate : float, default 0.0
        Continuously compounded discount rate, as a decimal.
    n : int, default 200_000
        Number of simulated scenarios.
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator, for reproducible prices.
    kind : {"call", "put"}, default "call"
        Option type.

    Returns
    -------
    MonteCarloPrice
        ``(price, standard_error, n)``.

    Raises
    ------
    ValueError
        If the copula is not bivariate, or ``kind`` is not ``"call"`` or
        ``"put"``.

    Notes
    -----
    Put-call parity holds scenario by scenario:
    :math:`C - P = e^{-rT}(\mathbb{E}[S_1 - S_2] - K)`.

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
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    prices = _terminal_prices(copula, margins, n, random_state)
    spread = prices[:, 0] - prices[:, 1]
    payoff = (
        np.maximum(spread - strike, 0.0) if kind == "call" else np.maximum(strike - spread, 0.0)
    )
    return _mc(payoff, rate, maturity)


# ======================================================================
# Constant-maturity swap (CMS) rates
# ======================================================================


def _par_bond(y: float, coupon: float, tenor: float, frequency: int) -> tuple[float, float, float]:
    r"""Price of a fixed-coupon bond as a function of its yield, with derivatives.

    :math:`G(y) = \sum_{i=1}^{N} \frac{c/m}{(1+y/m)^i} + \frac{1}{(1+y/m)^N}`,
    the standard yield-to-price map for a bond paying ``coupon`` ``frequency``
    times a year for ``tenor`` years. Returns :math:`(G, G', G'')`.
    """
    m = float(frequency)
    n_periods = round(tenor * m)
    if n_periods < 1:
        raise ValueError(f"tenor {tenor} is shorter than one coupon period")
    if 1.0 + y / m <= 0.0:
        raise ValueError(f"yield {y} is below -{m:g}, where the bond price diverges")

    i = np.arange(1, n_periods + 1, dtype=np.float64)
    disc = (1.0 + y / m) ** (-i)
    c = coupon / m

    price = float(c * disc.sum() + disc[-1])
    d1 = float(
        (c * (-i / m) * disc / (1.0 + y / m)).sum() + (-n_periods / m) * disc[-1] / (1.0 + y / m)
    )
    d2 = float(
        (c * (i * (i + 1.0) / m**2) * disc / (1.0 + y / m) ** 2).sum()
        + (n_periods * (n_periods + 1.0) / m**2) * disc[-1] / (1.0 + y / m) ** 2
    )
    return price, d1, d2


def cms_convexity_adjustment(
    forward: float,
    vol: float,
    maturity: float,
    tenor: float,
    frequency: int = 2,
    model: str = "lognormal",
) -> float:
    r"""How much to add to a forward swap rate to get the expected CMS rate actually paid.

    A constant-maturity swap (CMS) pays a swap rate on a single date, not over
    the swap's own schedule, and that timing mismatch makes the expected paid
    rate **higher** than the forward swap rate. This returns that gap (the
    convexity adjustment), which grows with vol, time to fixing and swap
    tenor.

    Parameters
    ----------
    forward : float
        Forward swap rate, as a decimal (``0.05`` for 5%).
    vol : float
        Volatility of that rate: lognormal (a decimal, e.g. ``0.20``) by
        default, or normal/absolute (in rate units, e.g. ``0.01`` for 100 bp)
        if ``model="normal"``. Must be non-negative.
    maturity : float
        Time to the fixing, in years. Must be non-negative.
    tenor : float
        Tenor of the underlying swap, in years.
    frequency : int, default 2
        Fixed-leg payments per year.
    model : {"lognormal", "normal"}, default "lognormal"
        Dynamics assumed for the swap rate. Normal (Bachelier) is the market
        standard for rates near or below zero, where a lognormal vol is
        meaningless.

    Returns
    -------
    float
        The additive adjustment, in the same units as ``forward`` (multiply by
        1e4 for basis points). ``0.0`` if ``vol`` or ``maturity`` is zero.

    Raises
    ------
    ValueError
        If ``model`` is unknown, ``vol`` or ``maturity`` is negative,
        ``tenor`` is shorter than one coupon period, or the rate is so
        negative that the bond price diverges.

    Notes
    -----
    The forward swap rate is a martingale under the *annuity* measure, not the
    payment measure, so the expected rate that actually gets paid is **higher**
    than the forward. The correction follows
    from the fact that the bond's forward *price* is the martingale: expanding
    :math:`G(y_T)` to second order in
    :math:`\mathbb{E}[G(y_T)] = G(y_0)` gives

    .. math::
        \mathbb{E}^T[y_T] - y_0 \;\approx\;
            -\tfrac{1}{2}\,\mathrm{Var}[y_T]\,\frac{G''(y_0)}{G'(y_0)},

    with :math:`\mathrm{Var}[y_T] = y_0^2\sigma^2 T` for a lognormal rate and
    :math:`\sigma^2 T` for a normal one. Since :math:`G' < 0` and
    :math:`G'' > 0`, the adjustment is positive.

    Ignoring it is not a rounding error: at a 5% rate, 20% vol and 5 years to
    expiry on a 10-year swap it is worth several basis points, and a CMS spread
    option is a difference of two such rates, so the two adjustments do not
    cancel -- the longer tenor carries the larger one, which is the whole
    directional content of the trade.

    This is the second-order approximation. Hagan's replication approach prices
    the same quantity as a strip of swaptions and so captures the smile; that is
    more accurate but needs a full swaption surface. What this function gives is
    the standard first-cut number, and :func:`cms_margin` builds on it.

    Examples
    --------
    Around 24 basis points on a 10-year rate fixing in 5 years -- far too large
    to leave out of a spread that is itself only ~150 bp wide:

    >>> from rcopula.derivatives import cms_convexity_adjustment
    >>> ca = cms_convexity_adjustment(0.05, 0.20, 5.0, 10.0)
    >>> float(round(ca * 1e4, 2))
    23.62

    It is positive, and grows with volatility, expiry and tenor:

    >>> longer = cms_convexity_adjustment(0.05, 0.20, 5.0, 30.0)
    >>> bool(longer > ca > 0)
    True

    With no volatility there is nothing to adjust:

    >>> cms_convexity_adjustment(0.05, 0.0, 5.0, 10.0)
    0.0
    """
    if model not in ("lognormal", "normal"):
        raise ValueError(f"model must be 'lognormal' or 'normal', got {model!r}")
    if vol < 0.0:
        raise ValueError(f"vol must be non-negative, got {vol}")
    if maturity < 0.0:
        raise ValueError(f"maturity must be non-negative, got {maturity}")
    if vol == 0.0 or maturity == 0.0:
        return 0.0

    variance = (forward**2 if model == "lognormal" else 1.0) * vol**2 * maturity
    _, d1, d2 = _par_bond(forward, forward, tenor, frequency)
    return float(-0.5 * variance * d2 / d1)


def cms_margin(
    forward: float,
    vol: float,
    maturity: float,
    tenor: float,
    frequency: int = 2,
    model: str = "lognormal",
) -> Margin:
    """The distribution of a CMS rate at its fixing, centred on the convexity-adjusted rate.

    Use it as one margin of a copula model of two or more swap rates; this is
    what :func:`cms_spread_option` does for each leg.

    Parameters
    ----------
    forward : float
        Forward swap rate, as a decimal.
    vol : float
        Volatility of the rate: lognormal (decimal) for ``model="lognormal"``,
        absolute (rate units) for ``model="normal"``. Must be positive for
        either model.
    maturity : float
        Time to the fixing, in years. Must be positive.
    tenor : float
        Tenor of the underlying swap, in years.
    frequency : int, default 2
        Fixed-leg payments per year.
    model : {"lognormal", "normal"}, default "lognormal"
        Shape of the distribution.

    Returns
    -------
    scipy.stats frozen distribution
        ``scipy.stats.lognorm`` (via :func:`lognormal_terminal`) or
        ``scipy.stats.norm``, with mean ``forward +
        cms_convexity_adjustment(...)``.

    Raises
    ------
    ValueError
        As :func:`cms_convexity_adjustment`; and, for either model, if
        ``vol`` or ``maturity`` is not positive. A zero vol or zero time to
        fixing would make the rate a known constant (a point mass), which
        is not a continuous margin a copula can use.

    Notes
    -----
    The mean is the forward swap rate plus
    :func:`cms_convexity_adjustment`; the shape is lognormal or normal
    according to ``model``. Plugging these into a copula is how a CMS spread
    option gets priced with each leg keeping its own dynamics.

    Examples
    --------
    The mean is the adjusted rate, not the forward:

    >>> from rcopula.derivatives import cms_convexity_adjustment, cms_margin
    >>> m = cms_margin(0.05, 0.20, 5.0, 10.0)
    >>> ca = cms_convexity_adjustment(0.05, 0.20, 5.0, 10.0)
    >>> bool(abs(m.mean() - (0.05 + ca)) < 1e-12)
    True
    """
    adjusted = forward + cms_convexity_adjustment(forward, vol, maturity, tenor, frequency, model)
    if vol <= 0 or maturity <= 0:
        raise ValueError(
            f"vol and maturity must be positive for a CMS margin, got {vol} and {maturity}"
        )
    if model == "normal":
        return stats.norm(loc=adjusted, scale=vol * np.sqrt(maturity))
    return lognormal_terminal(adjusted, vol, maturity)


class CmsLeg(NamedTuple):
    """One leg of a CMS spread option: which swap rate, and how volatile it is.

    A small record passed in a list of two to :func:`cms_spread_option`, e.g.
    ``CmsLeg(0.045, 0.22, 10.0)`` for a 10-year rate at 4.5% with 22%
    lognormal vol.

    Attributes
    ----------
    forward : float
        Forward swap rate, as a decimal.
    vol : float
        Volatility of that rate, on the scale implied by ``model``
        (lognormal decimal, or absolute rate units for ``"normal"``).
    tenor : float
        Swap tenor in years -- the "constant maturity".
    frequency : int, default 2
        Fixed-leg payments per year.
    model : {"lognormal", "normal"}, default "lognormal"
        Marginal dynamics.
    """

    forward: float
    vol: float
    tenor: float
    frequency: int = 2
    model: str = "lognormal"


def cms_spread_option(
    copula: Copula,
    legs: list[CmsLeg],
    strike: float,
    maturity: float,
    rate: float = 0.0,
    notional: float = 1.0,
    kind: str = "call",
    n: int = 200_000,
    random_state: np.random.Generator | int | None = None,
) -> MonteCarloPrice:
    r"""Monte-Carlo price of an option on the gap between two swap rates (a curve trade).

    Payoff :math:`N \max(y_1 - y_2 - K, 0)` on the two convexity-adjusted swap
    rates. The classic trade is 10-year minus 2-year: a bet on the slope of the
    curve rather than on its level.

    Parameters
    ----------
    copula : Copula
        Bivariate (``dim=2``) dependence between the two rates.
    legs : list of CmsLeg, length 2
        The two legs, in the order they appear in the spread
        (``legs[0] - legs[1]``).
    strike : float
        Spread strike, as a decimal (``0.015`` for 150 bp).
    maturity : float
        Fixing date, in years; also used for discounting.
    rate : float, default 0.0
        Continuously compounded discount rate, as a decimal.
    notional : float, default 1.0
        Notional the rate spread is paid on.
    kind : {"call", "put"}, default "call"
        A call pays when the curve steepens beyond the strike; a put when it
        flattens.
    n : int, default 200_000
        Number of simulated scenarios.
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator, for reproducible prices.

    Returns
    -------
    MonteCarloPrice
        ``(price, standard_error, n)``, in units of ``notional``.

    Raises
    ------
    ValueError
        If the copula is not bivariate, ``legs`` does not have exactly two
        entries, or ``kind`` is not ``"call"``/``"put"``.

    Notes
    -----
    This is a case where the copula is doing real work. The two rates are
    strongly dependent but not lognormally so; the spread is a small difference
    of two large numbers, so its distribution is extremely sensitive to how the
    joint tails behave; and the market quotes each leg's smile separately. A
    single correlation applied to two lognormals gets all three wrong at once.
    Choosing the dependence structure explicitly at least makes the assumption
    visible -- and, in the Gaussian-copula/lognormal case, reduces to the
    Margrabe/Kirk answer that desks already use.

    Examples
    --------
    A 10y-2y steepener struck at the current spread:

    >>> import rcopula as rc
    >>> from rcopula.derivatives import CmsLeg, cms_spread_option
    >>> legs = [CmsLeg(0.045, 0.22, 10.0), CmsLeg(0.030, 0.28, 2.0)]
    >>> mc = cms_spread_option(rc.GaussianCopula(0.85), legs, 0.015, 5.0,
    ...                        n=200_000, random_state=0)
    >>> bool(mc.price > 0.0)
    True

    Higher correlation squeezes the spread's distribution and so cheapens the
    option -- the correlation exposure that makes these trades interesting:

    >>> tight = cms_spread_option(rc.GaussianCopula(0.97), legs, 0.015, 5.0,
    ...                           n=200_000, random_state=0)
    >>> bool(tight.price < mc.price)
    True

    Lower-tail dependence, which pins the two rates together when both fall,
    prices differently again even at the same rank correlation:

    >>> clayton = cms_spread_option(rc.ClaytonCopula.from_tau(rc.GaussianCopula(0.85).tau()),
    ...                             legs, 0.015, 5.0, n=200_000, random_state=0)
    >>> bool(abs(clayton.price - mc.price) > clayton.standard_error)
    True
    """
    if copula.dim != 2:
        raise ValueError(f"a CMS spread option is bivariate; got dim={copula.dim}")
    if len(legs) != 2:
        raise ValueError(f"expected 2 legs, got {len(legs)}")
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    margins = [
        cms_margin(leg.forward, leg.vol, maturity, leg.tenor, leg.frequency, leg.model)
        for leg in legs
    ]
    rates = _terminal_prices(copula, margins, n, random_state)
    spread = rates[:, 0] - rates[:, 1]
    payoff = (
        np.maximum(spread - strike, 0.0) if kind == "call" else np.maximum(strike - spread, 0.0)
    )
    return _mc(notional * payoff, rate, maturity)


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
    r"""The implied-volatility smile of a basket, given its components and a copula.

    Prices a basket call at each strike by simulation, then converts each
    price to a Black-76 implied vol. Use it to see what smile your component
    smiles plus a dependence assumption imply for the basket, e.g. to compare
    with quoted index options.

    Parameters
    ----------
    copula : Copula
        Dependence between the ``d`` components.
    margins : scipy.stats frozen distribution or list of them, length d
        Terminal price distribution of each component (e.g.
        :class:`SmileMargin`). A single margin is reused for every asset.
    strikes : array_like of float, shape (k,)
        Basket strikes at which to compute the implied vol.
    maturity : float
        Time to expiry in years.
    weights : array_like of float, shape (d,), optional
        Basket weights. Default is equal weights ``1 / d``.
    rate : float, default 0.0
        Continuously compounded discount rate, as a decimal.
    n : int, default 200_000
        Number of simulated scenarios (shared across all strikes).
    random_state : int, numpy.random.Generator or None, default None
        Seed or generator, for reproducible results.

    Returns
    -------
    strikes : numpy.ndarray of float, shape (k,)
        The input strikes.
    vols : numpy.ndarray of float, shape (k,)
        Annualised implied volatility at each strike, as decimals.

    Raises
    ------
    ValueError
        If ``weights`` does not have length ``d``, or a simulated price cannot
        be inverted to a volatility (see :func:`implied_volatility`).

    Notes
    -----
    This is the calculation a single correlation number cannot do: give each
    component the marginal its **own smile** implies (see
    :class:`SmileMargin`), choose a dependence structure, and the basket smile
    falls out.

    The forward used in the inversion is the *simulated* basket mean, not the
    analytic one, so Monte-Carlo noise in the mean does not show up as a
    spurious skew. All strikes share the same simulated scenarios.

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
    if w.size != d:
        raise ValueError(f"weights has length {w.size}, expected {d} (one per component)")
    k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))

    basket = _terminal_prices(copula, margins, n, random_state) @ w
    forward = float(basket.mean())

    vols = np.empty(k.size)
    for i, strike in enumerate(k):
        price = _mc(np.maximum(basket - strike, 0.0), rate, maturity).price
        vols[i] = implied_volatility(price, forward, float(strike), maturity, rate)
    return k, vols
