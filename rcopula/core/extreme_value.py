r"""Extreme-value copulas.

A bivariate copula is *extreme-value* exactly when it can be written through a
**Pickands dependence function** :math:`A : [0,1] \to [1/2, 1]`:

.. math::

    C(u, v) = (uv)^{\,A\bigl(\log v / \log(uv)\bigr)}.

:math:`A` must be convex with :math:`\max(t, 1-t) \le A(t) \le 1`; the lower
envelope is comonotonicity and :math:`A \equiv 1` is independence. These are the
copulas that arise as limits of componentwise maxima, which makes them the
natural dependence models for joint extremes -- flood peak and volume, wind
speed and rainfall, simultaneous market crashes.

Every extreme-value copula has upper tail dependence
:math:`\lambda_U = 2(1 - A(1/2))` and **no** lower tail dependence.

The Gumbel-Hougaard copula is the only family that is both Archimedean and
extreme-value; it lives in :mod:`rcopula.core.archimedean` and is re-exposed
here through :func:`gumbel_pickands` for comparison.

References
----------
Pickands, J. (1981). Multivariate extreme value distributions.
    *Bulletin of the International Statistical Institute* 49, 859-878.
Gudendorf, G. and Segers, J. (2010). Extreme-value copulas. In *Copula Theory
    and Its Applications*, Lecture Notes in Statistics 198, 127-145. Springer.
    The survey this module follows for the density and the tau integral.
Galambos, J. (1975). Order statistics of samples from multivariate
    distributions. *JASA* 70(351), 674-680.
Husler, J. and Reiss, R.-D. (1989). Maxima of normal random vectors: between
    independence and complete dependence. *Statistics & Probability Letters*
    7(4), 283-286.
Demarta, S. and McNeil, A. J. (2005). The t copula and related copulas.
    *International Statistical Review* 73(1), 111-129. (The t-EV copula.)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm
from scipy.stats import t as student_t

from rcopula.core.base import Copula, TailDependence

__all__ = [
    "ExtremeValueCopula",
    "GalambosCopula",
    "HuslerReissCopula",
    "TEVCopula",
    "TawnCopula",
    "gumbel_pickands",
]

#: Step for the central-difference fallback used when a family does not supply
#: analytic Pickands derivatives. Chosen to balance truncation against roundoff
#: for the second derivative, where the error scales as h^2 + eps/h^2.
_FD_STEP = 1e-4


def gumbel_pickands(w: ArrayLike, theta: float) -> NDArray[np.float64]:
    r"""Evaluate the curve that describes the Gumbel copula's dependence, at given points.

    Every extreme-value copula is summarised by one curve on ``[0, 1]``, its
    *Pickands dependence function* ``A``: ``A = 1`` everywhere means the two
    variables are independent, and the lower it dips the stronger they move
    together. This function returns that curve for the Gumbel-Hougaard copula,
    so you can compare it with :meth:`ExtremeValueCopula.A` of the other
    families in this module.

    Parameters
    ----------
    w : float or array_like of float, any shape
        Points at which to evaluate ``A``, each in ``[0, 1]``.
    theta : float
        Gumbel dependence parameter, ``theta >= 1``. ``1`` is independence and
        larger values mean stronger dependence.

    Returns
    -------
    numpy.ndarray of float, same shape as ``w``
        Values of ``A(w)``, each in ``[1/2, 1]``. A 0-d array when ``w`` is a
        scalar.

    Notes
    -----
    :math:`A(t) = (t^\theta + (1-t)^\theta)^{1/\theta}`.

    Provided for comparison: Gumbel is the unique family that is simultaneously
    Archimedean and extreme-value. The copula itself is
    :class:`~rcopula.core.archimedean.GumbelCopula`.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.core.extreme_value import gumbel_pickands
    >>> float(np.round(gumbel_pickands(0.5, 2.0), 12))
    0.707106781187
    """
    t = np.asarray(w, dtype=np.float64)
    return (t**theta + (1.0 - t) ** theta) ** (1.0 / theta)


class ExtremeValueCopula(Copula):
    r"""Shared base for two-variable copulas built to model joint extremes.

    An *extreme-value copula* describes how the largest values of two
    quantities (yearly flood peak and flood volume, the worst daily losses of
    two assets) occur together. Such copulas have dependence in the upper tail
    (big values tend to happen together) and none in the lower tail. You do not
    instantiate this class directly; use one of :class:`GalambosCopula`,
    :class:`HuslerReissCopula`, :class:`TawnCopula` or :class:`TEVCopula`.

    Parameters
    ----------
    params : float or array_like of float, shape (n_params,), default nan
        Parameter value(s) of the family. ``nan`` means "to be estimated", so
        ``fit(GalambosCopula(), u)`` works without inventing a starting value.
    dim : int, default 2
        Number of variables. Must be ``2``: only the bivariate case is
        implemented.
    free : array_like of bool, shape (n_params,), or None, default None
        Which parameters a fit may change; ``False`` holds that parameter at
        its current value. ``None`` makes every parameter free.

    Attributes
    ----------
    dim : int
        Always ``2``.
    params : numpy.ndarray of float, shape (n_params,)
        The parameter vector (read-only).
    free : numpy.ndarray of bool, shape (n_params,)
        Mask of parameters free to be estimated.

    Raises
    ------
    ValueError
        If ``dim`` is not 2, or a parameter lies outside its admissible range.

    Notes
    -----
    Subclasses supply :meth:`A`; supplying :meth:`dA` and :meth:`d2A` as well
    makes the density exact rather than finite-differenced (the defaults here
    use central differences with step ``1e-4``).

    The copula is :math:`C(u, v) = (uv)^{A(\log v / \log(uv))}`, and every
    member has :math:`\lambda_L = 0`, :math:`\lambda_U = 2(1 - A(1/2))`.
    """

    def __init__(
        self, params: ArrayLike = np.nan, dim: int = 2, *, free: ArrayLike | None = None
    ) -> None:
        # NaN means "this family, to be estimated" -- the same convention as
        # ClaytonCopula() and GaussianCopula(), and what makes
        # ``fit(GalambosCopula(), u)`` the natural idiom. Without a default
        # these families were the only ones that could not be named without
        # also inventing a parameter value.
        if int(dim) != 2:
            raise ValueError(
                f"{type(self).__name__} is bivariate only, got dim={int(dim)}. "
                "Multivariate extreme-value copulas need a Pickands function on "
                "the simplex, which is not implemented."
            )
        super().__init__(params, 2, free=free)

    # -- Pickands function and its derivatives -------------------------

    @abstractmethod
    def A(self, w: ArrayLike) -> NDArray[np.float64]:
        r"""Evaluate the curve that fully determines this copula's dependence.

        This is the *Pickands dependence function* ``A``. It equals ``1``
        everywhere for independent variables and dips towards
        ``max(t, 1 - t)`` as dependence approaches perfect co-movement, so
        plotting it is a quick visual summary of the family and parameter.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]`` at which to evaluate ``A``.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A(w)``, each in ``[1/2, 1]``.

        Notes
        -----
        ``A`` is convex with :math:`\max(t, 1-t) \le A(t) \le 1`, and the copula
        is :math:`C(u, v) = (uv)^{A(\log v / \log(uv))}`.
        """

    def dA(self, w: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the slope (first derivative) of the dependence curve :meth:`A`.

        Mostly used internally for the density and for sampling. The default
        is a central finite difference with step ``1e-4``; families that know
        the derivative in closed form override it.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``. In the finite-difference default, points
            closer than the step to 0 or 1 are evaluated at the step instead.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A'(w)``, each in ``[-1, 1]``.
        """
        t = np.asarray(w, dtype=np.float64)
        h = _FD_STEP
        tc = np.clip(t, h, 1.0 - h)
        return (self.A(tc + h) - self.A(tc - h)) / (2.0 * h)

    def d2A(self, w: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the curvature (second derivative) of the dependence curve :meth:`A`.

        Used internally for the density and for Kendall's tau. The default is
        a central finite difference with step ``1e-4``; families with a closed
        form override it.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``. In the finite-difference default, points
            closer than the step to 0 or 1 are evaluated at the step instead.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A''(w)``. Non-negative, since ``A`` is convex.
        """
        t = np.asarray(w, dtype=np.float64)
        h = _FD_STEP
        tc = np.clip(t, h, 1.0 - h)
        return (self.A(tc + h) - 2.0 * self.A(tc) + self.A(tc - h)) / (h * h)

    # -- numerical core ------------------------------------------------

    def _cdf(self, u, params):
        x, y = u[:, 0], u[:, 1]
        log_uv = np.log(x) + np.log(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.log(y) / log_uv
        t = np.where(np.isfinite(t), t, 0.5)
        return np.exp(log_uv * self.A(t))

    def _logpdf(self, u, params):
        r"""Density of a bivariate extreme-value copula.

        With :math:`s = -\log(uv)` and :math:`t = \log v/\log(uv)`,

        .. math::

            c(u,v) = \frac{C(u,v)}{uv}\Bigl\{
                \bigl[A(t) - t A'(t)\bigr]\bigl[A(t) + (1-t)A'(t)\bigr]
                + \frac{t(1-t)}{s}\,A''(t)\Bigr\}

        (Gudendorf & Segers 2010, eq. 5). The first product is the contribution
        of the two conditional distributions; the ``A''`` term is what makes the
        copula non-degenerate.
        """
        x, y = u[:, 0], u[:, 1]
        log_uv = np.log(x) + np.log(y)
        s = -log_uv
        t = np.log(y) / log_uv

        first, second = self._factors(t)
        # The bracket is a density up to positive factors, so it cannot be
        # negative; deep in a corner it is a sum of terms at the precision floor
        # and rounding can make it so. Clipping gives log(0) = -inf, i.e. a
        # density of exactly zero, which is the correct limit there -- where
        # letting log() see a negative number gave nan.
        bracket = np.maximum(first * second + t * (1.0 - t) * self.d2A(t) / s, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return log_uv * self.A(t) - np.log(x) - np.log(y) + np.log(bracket)

    def _factors(self, t: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""The two conditional factors :math:`A - tA'` and :math:`A + (1-t)A'`.

        Both are non-negative for *every* Pickands function, because the
        conditional distributions they scale are. That follows from the two
        defining bounds :math:`\max(t, 1-t) \le A \le 1` and
        :math:`-1 \le A' \le 1`: for instance
        :math:`A + (1-t)A' \ge (1-t) + (1-t)(-1) = 0`, with equality exactly at
        the boundary.

        Written as stated they are differences of two quantities of order 1 that
        agree in the corner, so rounding can push them slightly negative -- which
        turned the log-density into ``nan``. Clipping at zero is the correct
        limit there, since the true value is provably non-negative and the
        density genuinely vanishes. Families with a cancellation-free closed
        form override this.
        """
        a, da = self.A(t), self.dA(t)
        return np.maximum(a - t * da, 0.0), np.maximum(a + (1.0 - t) * da, 0.0)

    def _cond_cdf(self, v: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        r""":math:`\partial C/\partial u`, the conditional distribution of ``V`` given ``U``."""
        log_u, log_v = np.log(u), np.log(v)
        log_uv = log_u + log_v
        t = log_v / log_uv
        a = self.A(t)
        da = self.dA(t)
        return np.exp(log_uv * a) / u * (a - t * da)

    def _rvs(self, size, params, rng):
        """Conditional inversion, solved by vectorised bisection.

        Extreme-value copulas have no general closed-form sampler, so invert
        ``dC/du = w`` numerically. Bisection rather than Brent because it
        vectorises over all draws at once, and 60 halvings already reach 1e-18.
        """
        u = rng.uniform(size=size)
        w = rng.uniform(size=size)

        lo = np.full(size, 1e-12)
        hi = np.full(size, 1.0 - 1e-12)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            too_small = self._cond_cdf(mid, u) < w
            lo = np.where(too_small, mid, lo)
            hi = np.where(too_small, hi, mid)
        return np.column_stack([u, 0.5 * (lo + hi)])

    # -- dependence measures -------------------------------------------

    def tau(self) -> float:
        r"""Return Kendall's tau, a rank-based measure of how strongly the variables move together.

        Kendall's tau is the probability that two random draws are ordered the
        same way in both variables minus the probability they are ordered
        oppositely. For extreme-value copulas it lies in ``[0, 1]``: ``0`` is
        independence and ``1`` is perfect co-movement.

        Returns
        -------
        float
            Kendall's tau of the copula.

        Raises
        ------
        ValueError
            If any parameter is still ``nan`` (not yet fitted or supplied).

        Notes
        -----
        :math:`\tau = \int_0^1 \frac{t(1-t)}{A(t)}\,\mathrm{d}A'(t)`, evaluated as
        :math:`\int_0^1 t(1-t) A''(t)/A(t)\,\mathrm{d}t` on a 400-point
        Gauss-Legendre grid.
        """
        self._require_specified()
        nodes, weights = np.polynomial.legendre.leggauss(400)
        t = 0.5 * (nodes + 1.0)
        wt = 0.5 * weights
        integrand = t * (1.0 - t) * self.d2A(t) / self.A(t)
        return float(np.sum(wt * integrand))

    def rho(self) -> float:
        r"""Return Spearman's rho, the correlation between the two variables' ranks.

        Spearman's rho is the ordinary correlation computed on the copula's
        uniform scale. For extreme-value copulas it lies in ``[0, 1]``: ``0``
        is independence and ``1`` is perfect co-movement.

        Returns
        -------
        float
            Spearman's rho of the copula.

        Raises
        ------
        ValueError
            If any parameter is still ``nan`` (not yet fitted or supplied).

        Notes
        -----
        :math:`\rho_S = 12\int_0^1 (A(t)+1)^{-2}\,\mathrm{d}t - 3`, evaluated on a
        400-point Gauss-Legendre grid.
        """
        self._require_specified()
        nodes, weights = np.polynomial.legendre.leggauss(400)
        t = 0.5 * (nodes + 1.0)
        wt = 0.5 * weights
        return float(12.0 * np.sum(wt / (self.A(t) + 1.0) ** 2) - 3.0)

    def lambda_(self) -> TailDependence:
        r"""Return how likely the two variables are to be extreme at the same time.

        These are the tail-dependence coefficients. ``upper`` is, roughly, the
        chance one variable is extremely large given the other is; ``lower``
        is the same for extremely small values. Extreme-value copulas always
        have ``lower = 0``.

        Returns
        -------
        TailDependence
            Named tuple ``(lower, upper)`` of floats in ``[0, 1]``; here always
            :math:`(0,\ 2(1 - A(1/2)))`.

        Raises
        ------
        ValueError
            If any parameter is still ``nan`` (not yet fitted or supplied).
        """
        self._require_specified()
        return TailDependence(lower=0.0, upper=float(2.0 * (1.0 - self.A(0.5))))

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> ExtremeValueCopula:
        """Build a copula of this family whose Kendall's tau equals a target value.

        Useful when you know (or have estimated) how strongly two variables
        move together and want the family parameter that matches it. The
        parameter is found numerically by root-finding on ``tau(theta)``.

        Parameters
        ----------
        tau : float
            Target Kendall's tau. Must be attainable by the family; for these
            families that means strictly between 0 and 1 (and below about
            0.418 for :class:`TawnCopula`).
        dim : int, default 2
            Accepted for interface compatibility with :class:`Copula`; not
            used, since these copulas are always bivariate.
        **kwargs
            Extra keyword arguments passed on to the constructor (for example
            ``free``).

        Returns
        -------
        ExtremeValueCopula
            A new instance of the calling class with the calibrated parameter.

        Raises
        ------
        ValueError
            If ``tau`` cannot be reached by the family.

        Examples
        --------
        >>> from rcopula import GalambosCopula
        >>> g = GalambosCopula.from_tau(0.5)
        >>> round(g.tau(), 6)
        0.5
        """
        lo, hi = cls._tau_bracket()
        try:
            theta = float(brentq(lambda th: cls(th, **kwargs).tau() - tau, lo, hi, xtol=1e-12))
        except ValueError as exc:
            raise ValueError(
                f"Kendall's tau = {tau} is not attainable by the {cls.__name__} family"
            ) from exc
        return cls(theta, **kwargs)

    @classmethod
    def _tau_bracket(cls) -> tuple[float, float]:
        raise NotImplementedError


class GalambosCopula(ExtremeValueCopula):
    r"""Galambos copula: a one-parameter model for variables whose large values occur together.

    A common choice for joint extremes (for example annual maxima). One
    parameter ``theta`` controls the strength: near ``0`` the variables are
    nearly independent, and as it grows they approach perfect co-movement in
    the upper tail. Bivariate only.

    Parameters
    ----------
    params : float, default nan
        Dependence parameter ``theta``, ``theta >= 0``. ``nan`` means "to be
        estimated".
    dim : int, default 2
        Must be ``2``.
    free : array_like of bool, shape (1,), or None, default None
        Whether ``theta`` may be changed by a fit. ``None`` means free.

    Attributes
    ----------
    theta : float
        The dependence parameter.

    Raises
    ------
    ValueError
        If ``dim`` is not 2 or ``theta`` is negative.

    Notes
    -----
    :math:`A(t) = 1 - \bigl(t^{-\theta} + (1-t)^{-\theta}\bigr)^{-1/\theta}`,
    with :math:`\theta > 0`. Independence at :math:`\theta \to 0`,
    comonotonicity as :math:`\theta \to \infty`.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import GalambosCopula
    >>> g = GalambosCopula(2.0)
    >>> float(np.round(g.A(0.5), 12))
    0.646446609407
    >>> float(np.round(g.lambda_().upper, 10))
    0.7071067812
    >>> float(round(g.tau(), 7))
    0.6311589
    """

    name = "Galambos"
    param_names = ("theta",)

    @property
    def theta(self) -> float:
        """The dependence parameter ``theta`` (float, ``>= 0``; larger is stronger)."""
        return float(self._params[0])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Allowed range of each parameter, as ``[(0.0, inf)]`` for ``theta``.

        Returns
        -------
        list of tuple of (float, float), length 1
            ``(lower, upper)`` bounds for ``theta``.
        """
        return [(0.0, np.inf)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> GalambosCopula:
        return GalambosCopula(float(np.atleast_1d(params)[0]), free=free)

    #: Endpoint guard. Everything below is evaluated in logs, so this only has
    #: to keep ``log t`` finite -- unlike the previous direct powers, which
    #: overflowed once ``theta * log10(1/t)`` passed ~308 and needed a
    #: theta-dependent bound.
    _EDGE = 1e-300

    def _clip(self, w: ArrayLike) -> NDArray[np.float64]:
        """Keep ``t`` strictly inside ``(0, 1)`` so its logarithm is finite."""
        return np.clip(np.asarray(w, dtype=np.float64), self._EDGE, 1.0 - 1e-16)

    def _log_g(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        r""":math:`\log\bigl(t^{-\theta} + (1-t)^{-\theta}\bigr)`, by ``logaddexp``."""
        th = self.theta
        return np.logaddexp(-th * np.log(t), -th * np.log1p(-t))

    def A(self, w):
        r"""Evaluate the Galambos dependence curve (Pickands function) at ``w``.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``; values are clipped just inside the interval.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A(w)`` in ``[1/2, 1]``.

        Notes
        -----
        :math:`1 - g^{-1/\theta}` with :math:`g = t^{-\theta} + (1-t)^{-\theta}`,
        written as ``-expm1`` so it never cancels.
        """
        t = self._clip(w)
        return -np.expm1(-self._log_g(t) / self.theta)

    def dA(self, w):
        r"""Evaluate the slope of the Galambos dependence curve, in closed form.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``; values are clipped just inside the interval.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A'(w)`` in ``[-1, 1]``.

        Notes
        -----
        :math:`A'(t) = -g^{-1/\theta - 1}\,(t^{-\theta-1} - (1-t)^{-\theta-1})`.

        The difference is formed in logs: the two powers span hundreds of orders
        of magnitude for large ``theta``, and their signed difference is
        ``sign * e^{max} * (1 - e^{-|gap|})``, which ``expm1`` evaluates exactly
        even when the gap is tiny (at ``t = 1/2`` it is zero, and so is ``A'``).
        """
        t = self._clip(w)
        th = self.theta
        log_t, log_s = np.log(t), np.log1p(-t)
        p, q = -(th + 1.0) * log_t, -(th + 1.0) * log_s
        gap = np.abs(p - q)
        with np.errstate(divide="ignore"):
            log_diff = np.maximum(p, q) + np.log(-np.expm1(-gap))
        magnitude = np.exp(log_diff - (1.0 / th + 1.0) * self._log_g(t))
        return np.where(p >= q, -magnitude, magnitude)

    def d2A(self, w):
        r"""Evaluate the curvature of the Galambos dependence curve, in closed form.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``; values are clipped just inside the interval.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A''(w)``, non-negative (may be ``inf`` at the very
            edges for large ``theta``).

        Notes
        -----
        :math:`A''(t) = (1 + \theta)\,\bigl(t(1-t)\bigr)^{-\theta-2}\,g^{-1/\theta-2}`.

        The direct second derivative is a difference of two enormous terms, and
        at ``theta = 30`` it came out **negative** -- impossible, since ``A`` is
        convex by definition, and it drove the copula density negative for half
        the unit square. Expanding that difference collapses it: with
        :math:`a = t^{-\theta}`, :math:`b = (1-t)^{-\theta}` the bracket is

        .. math::
            g\bigl(t^{-\theta-2} + (1-t)^{-\theta-2}\bigr) - \bigl(t^{-\theta-1}
                - (1-t)^{-\theta-1}\bigr)^2
            = ab\left(\tfrac{1}{t} + \tfrac{1}{1-t}\right)^2
            = \bigl(t(1-t)\bigr)^{-\theta-2},

        using :math:`t + (1-t) = 1`. What was a cancelling difference is a
        single positive term, so the result is non-negative by construction and
        overflow-free in logs.
        """
        t = self._clip(w)
        th = self.theta
        log_ts = np.log(t) + np.log1p(-t)
        log_out = np.log1p(th) - (th + 2.0) * log_ts - (1.0 / th + 2.0) * self._log_g(t)
        with np.errstate(over="ignore"):
            return np.asarray(np.exp(log_out))

    def _factors(self, t):
        r"""Exact, cancellation-free conditional factors.

        Substituting :math:`A = 1 - g^{-1/\theta}` and
        :math:`A' = -g^{-1/\theta-1}(t^{-\theta-1} - (1-t)^{-\theta-1})` into
        :math:`A + (1-t)A'` and using :math:`1 + (1-t)/t = 1/t` collapses it to
        a single term:

        .. math::
            A + (1-t)A' = 1 - t^{-(\theta+1)} g^{-(1/\theta+1)},

        and symmetrically for the other factor. Both are then ``-expm1`` of a
        quantity computed entirely in logs, so nothing cancels and neither can
        come out negative.
        """
        t = self._clip(t)
        th = self.theta
        scale = (1.0 / th + 1.0) * self._log_g(t)
        with np.errstate(over="ignore"):
            first = -np.expm1(-(th + 1.0) * np.log1p(-t) - scale)
            second = -np.expm1(-(th + 1.0) * np.log(t) - scale)
        # The exponent is <= 0 exactly, but only to within rounding; a positive
        # last bit turns -expm1 slightly negative.
        return np.maximum(first, 0.0), np.maximum(second, 0.0)

    @classmethod
    def _tau_bracket(cls) -> tuple[float, float]:
        return (1e-8, 25.0)


class HuslerReissCopula(ExtremeValueCopula):
    r"""Husler-Reiss copula: the joint-extremes model that arises from normally distributed data.

    If you take maxima of many correlated normal variables, their dependence
    converges to this copula. One parameter ``theta`` controls the strength:
    near ``0`` the extremes are nearly independent, and large values give
    near-perfect co-movement. Bivariate only.

    Parameters
    ----------
    params : float, default nan
        Dependence parameter ``theta``, ``theta >= 0``. ``nan`` means "to be
        estimated".
    dim : int, default 2
        Must be ``2``.
    free : array_like of bool, shape (1,), or None, default None
        Whether ``theta`` may be changed by a fit. ``None`` means free.

    Attributes
    ----------
    theta : float
        The dependence parameter.

    Raises
    ------
    ValueError
        If ``dim`` is not 2 or ``theta`` is negative.

    Notes
    -----
    :math:`A(t) = t\,\Phi\!\left(\tfrac{1}{\theta} + \tfrac{\theta}{2}\log\tfrac{t}{1-t}\right)
    + (1-t)\,\Phi\!\left(\tfrac{1}{\theta} - \tfrac{\theta}{2}\log\tfrac{t}{1-t}\right)`,
    with :math:`\theta > 0`. The extreme-value limit of the Gaussian copula.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import HuslerReissCopula
    >>> h = HuslerReissCopula(1.5)
    >>> float(np.round(h.A(0.5), 12))
    0.747507462453
    >>> float(np.round(h.lambda_().upper, 7))
    0.5049851
    """

    name = "HuslerReiss"
    param_names = ("theta",)

    @property
    def theta(self) -> float:
        """The dependence parameter ``theta`` (float, ``>= 0``; larger is stronger)."""
        return float(self._params[0])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Allowed range of each parameter, as ``[(0.0, inf)]`` for ``theta``.

        Returns
        -------
        list of tuple of (float, float), length 1
            ``(lower, upper)`` bounds for ``theta``.
        """
        return [(0.0, np.inf)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> HuslerReissCopula:
        return HuslerReissCopula(float(np.atleast_1d(params)[0]), free=free)

    def _z(self, t):
        th = self.theta
        ell = np.log(t / (1.0 - t))
        return 1.0 / th + 0.5 * th * ell, 1.0 / th - 0.5 * th * ell

    def A(self, w):
        """Evaluate the Husler-Reiss dependence curve (Pickands function) at ``w``.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``; values are clipped to ``[1e-12, 1 - 1e-12]``.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A(w)`` in ``[1/2, 1]``.
        """
        t = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        z1, z2 = self._z(t)
        return t * ndtr(z1) + (1.0 - t) * ndtr(z2)

    def dA(self, w):
        r"""Evaluate the slope of the Husler-Reiss dependence curve, in closed form.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``; values are clipped to ``[1e-12, 1 - 1e-12]``.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A'(w)`` in ``[-1, 1]``.

        Notes
        -----
        Exactly :math:`\Phi(z_1) - \Phi(z_2)`.

        The two density terms cancel: since
        :math:`\phi(z_1)/\phi(z_2) = e^{-\ell} = (1-t)/t`, we get
        :math:`t\phi(z_1) = (1-t)\phi(z_2)` and their contributions to
        :math:`A'` are equal and opposite.
        """
        t = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        z1, z2 = self._z(t)
        return ndtr(z1) - ndtr(z2)

    def d2A(self, w):
        r"""Evaluate the curvature of the Husler-Reiss dependence curve, in closed form.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``; values are clipped to ``[1e-12, 1 - 1e-12]``.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A''(w)``, positive.

        Notes
        -----
        Obtained by differentiating :meth:`dA` once more. With
        :math:`z_1' = \tfrac{\theta}{2t(1-t)}` and :math:`z_2' = -z_1'`,

        .. math::
            A''(t) = \phi(z_1) z_1' - \phi(z_2) z_2'
                   = \frac{\theta}{2t(1-t)}\bigl(\phi(z_1) + \phi(z_2)\bigr),

        which is manifestly positive -- as it must be, since ``A`` is convex.
        """
        t = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        th = self.theta
        z1, z2 = self._z(t)
        return 0.5 * th / (t * (1.0 - t)) * (norm.pdf(z1) + norm.pdf(z2))

    @classmethod
    def _tau_bracket(cls) -> tuple[float, float]:
        return (1e-8, 50.0)


class TawnCopula(ExtremeValueCopula):
    r"""Tawn copula: a simple joint-extremes model that only allows weak-to-moderate dependence.

    Its dependence curve is a parabola, which makes it easy to reason about,
    but it cannot represent strong dependence: Kendall's tau tops out at about
    0.418 (reached at ``theta = 1``). This is R's one-parameter form.
    Bivariate only.

    Parameters
    ----------
    params : float, default nan
        Dependence parameter ``theta`` in ``[0, 1]``. ``0`` is independence.
        ``nan`` means "to be estimated".
    dim : int, default 2
        Must be ``2``.
    free : array_like of bool, shape (1,), or None, default None
        Whether ``theta`` may be changed by a fit. ``None`` means free.

    Attributes
    ----------
    theta : float
        The dependence parameter.

    Raises
    ------
    ValueError
        If ``dim`` is not 2 or ``theta`` is outside ``[0, 1]``.

    Notes
    -----
    :math:`A(t) = 1 - \theta\,t(1-t)`, with :math:`\theta \in [0, 1]`. A
    quadratic perturbation of independence, so like FGM it reaches only weak
    dependence -- R notes it is valid only for :math:`\tau < 0.4184`.

    Examples
    --------
    >>> from rcopula import TawnCopula
    >>> t = TawnCopula(0.6)
    >>> float(t.A(0.5))
    0.85
    >>> float(round(t.tau(), 7))
    0.2275623
    >>> float(round(t.lambda_().upper, 12))
    0.3
    """

    name = "Tawn"
    param_names = ("theta",)

    @property
    def theta(self) -> float:
        """The dependence parameter ``theta`` (float in ``[0, 1]``; larger is stronger)."""
        return float(self._params[0])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Allowed range of each parameter, as ``[(0.0, 1.0)]`` for ``theta``.

        Returns
        -------
        list of tuple of (float, float), length 1
            ``(lower, upper)`` bounds for ``theta``.
        """
        return [(0.0, 1.0)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> TawnCopula:
        return TawnCopula(float(np.atleast_1d(params)[0]), free=free)

    def A(self, w):
        r"""Evaluate the Tawn dependence curve, :math:`1 - \theta\,w(1-w)`.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A(w)`` in ``[3/4, 1]``.
        """
        t = np.asarray(w, dtype=np.float64)
        return 1.0 - self.theta * t * (1.0 - t)

    def dA(self, w):
        r"""Evaluate the slope of the Tawn dependence curve, :math:`\theta(2w - 1)`.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A'(w)``.
        """
        t = np.asarray(w, dtype=np.float64)
        return self.theta * (2.0 * t - 1.0)

    def d2A(self, w):
        r"""Evaluate the curvature of the Tawn dependence curve, the constant :math:`2\theta`.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``; only the shape is used.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            ``2 * theta`` at every point.
        """
        t = np.asarray(w, dtype=np.float64)
        return np.full(t.shape, 2.0 * self.theta)

    @classmethod
    def _tau_bracket(cls) -> tuple[float, float]:
        return (1e-10, 1.0)


class TEVCopula(ExtremeValueCopula):
    r"""t-EV copula: the joint-extremes model that arises from Student-t distributed data.

    If you take maxima of many variables linked by a Student-t copula (a
    common model for financial returns), their dependence converges to this
    copula. It has two parameters: a correlation ``rho`` and the
    degrees of freedom ``df`` (smaller ``df`` means heavier joint tails).
    Bivariate only.

    Parameters
    ----------
    rho : float or array_like of float, shape (2,), default nan
        Correlation parameter in ``[-1, 1]``; ``nan`` means "to be estimated".
        A length-2 array is read as ``(rho, df)`` and then the ``df`` keyword
        is ignored (this is how copies are rebuilt internally).
    dim : int, default 2
        Must be ``2``.
    df : float, default 4.0
        Degrees of freedom, ``df > 0`` (at least ``1e-6``).
    free : array_like of bool, shape (2,), or None, default None
        Which of ``(rho, df)`` a fit may change. ``None`` makes both free.

    Attributes
    ----------
    rho_param : float
        The correlation parameter (named so as not to clash with the
        :meth:`rho` method).
    df : float
        The degrees of freedom.

    Raises
    ------
    ValueError
        If ``dim`` is not 2 or a parameter is outside its range.

    Notes
    -----
    :math:`A(t) = t\,T_{\nu+1}(z_t) + (1-t)\,T_{\nu+1}(z_{1-t})` with

    .. math::

        z_t = \sqrt{\tfrac{\nu+1}{1-\rho^2}}
              \left[\left(\tfrac{t}{1-t}\right)^{1/\nu} - \rho\right].

    Because the t copula has tail dependence for every finite ``df``, the t-EV
    copula is never the independence copula: :math:`\tau > 0` always.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import TEVCopula
    >>> c = TEVCopula(0.5, df=4)
    >>> float(np.round(c.A(0.5), 12))
    0.87341500245
    >>> float(round(c.tau(), 6))
    0.195867
    """

    name = "TEV"
    param_names = ("rho", "df")

    def __init__(
        self,
        rho: ArrayLike = np.nan,
        dim: int = 2,
        *,
        df: float = 4.0,
        free: ArrayLike | None = None,
    ) -> None:
        given = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        params = given if given.size == 2 else np.array([float(given[0]), float(df)])
        super().__init__(params, dim, free=free)

    @property
    def rho_param(self) -> float:
        """The correlation parameter ``rho`` (float in ``[-1, 1]``).

        Named ``rho_param`` because :meth:`rho` returns Spearman's rho.
        """
        return float(self._params[0])

    @property
    def df(self) -> float:
        """The degrees of freedom ``df`` (float, positive; smaller means heavier tails)."""
        return float(self._params[1])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Allowed ranges, ``[(-1.0, 1.0), (1e-6, inf)]`` for ``(rho, df)``.

        Returns
        -------
        list of tuple of (float, float), length 2
            ``(lower, upper)`` bounds for ``rho`` and ``df``.
        """
        return [(-1.0, 1.0), (1e-6, np.inf)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> TEVCopula:
        return TEVCopula(np.atleast_1d(params), free=free)

    def _z_and_dz(self, t):
        """The two t-quantile arguments and their derivatives in ``t``."""
        rho, nu = self.rho_param, self.df
        s = 1.0 - t
        k = np.sqrt((nu + 1.0) / (1.0 - rho * rho))
        a = 1.0 / nu
        z1 = k * ((t / s) ** a - rho)
        z2 = k * ((s / t) ** a - rho)
        dz1 = k * a * (t / s) ** (a - 1.0) / s**2
        dz2 = -k * a * (s / t) ** (a - 1.0) / t**2
        return z1, z2, dz1, dz2

    def A(self, w):
        """Evaluate the t-EV dependence curve (Pickands function) at ``w``.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``; values are clipped to ``[1e-12, 1 - 1e-12]``.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A(w)`` in ``[1/2, 1]``.
        """
        t = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        nu = self.df
        z1, z2, _, _ = self._z_and_dz(t)
        return t * student_t.cdf(z1, df=nu + 1.0) + (1.0 - t) * student_t.cdf(z2, df=nu + 1.0)

    def dA(self, w):
        r"""Evaluate the slope of the t-EV dependence curve, in closed form.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``; values are clipped to ``[1e-12, 1 - 1e-12]``.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A'(w)`` in ``[-1, 1]``.

        Notes
        -----
        Exactly :math:`T_{\nu+1}(z_1) - T_{\nu+1}(z_2)`.

        As for Husler-Reiss, the two density terms cancel identically:
        :math:`t f(z_1) z_1' + (1-t) f(z_2) z_2' = 0` for every ``t``. Relying on
        that rather than finite-differencing matters -- the central-difference
        fallback left the t-EV density wrong by ~2e-4.
        """
        t = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        nu = self.df
        z1, z2, _, _ = self._z_and_dz(t)
        return student_t.cdf(z1, df=nu + 1.0) - student_t.cdf(z2, df=nu + 1.0)

    def d2A(self, w):
        r"""Evaluate the curvature of the t-EV dependence curve, in closed form.

        Parameters
        ----------
        w : float or array_like of float, any shape
            Points in ``[0, 1]``; values are clipped to ``[1e-12, 1 - 1e-12]``.

        Returns
        -------
        numpy.ndarray of float, same shape as ``w``
            Values of ``A''(w)``, positive.

        Notes
        -----
        :math:`A''(t) = f(z_1) z_1' - f(z_2) z_2'`, both terms positive.
        """
        t = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        nu = self.df
        z1, z2, dz1, dz2 = self._z_and_dz(t)
        return student_t.pdf(z1, df=nu + 1.0) * dz1 - student_t.pdf(z2, df=nu + 1.0) * dz2

    @classmethod
    def _tau_bracket(cls) -> tuple[float, float]:
        return (-0.999, 0.999)

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> TEVCopula:
        """Build a t-EV copula whose Kendall's tau equals a target value, keeping ``df`` fixed.

        Only the correlation ``rho`` is solved for, numerically; the degrees
        of freedom are whatever you pass (default 4).

        Parameters
        ----------
        tau : float
            Target Kendall's tau. Must be attainable at the given ``df``; the
            t-EV copula always has ``tau > 0``.
        dim : int, default 2
            Accepted for interface compatibility; not used (always bivariate).
        **kwargs
            ``df`` (float, default 4.0) sets the degrees of freedom; any other
            keyword (for example ``free``) goes to the constructor.

        Returns
        -------
        TEVCopula
            A new copula with the calibrated ``rho`` and the given ``df``.

        Raises
        ------
        ValueError
            If ``tau`` is not attainable at that ``df``.

        Examples
        --------
        >>> from rcopula import TEVCopula
        >>> c = TEVCopula.from_tau(0.3, df=4)
        >>> round(c.tau(), 6), c.df
        (0.3, 4.0)
        """
        df = kwargs.pop("df", 4.0)
        lo, hi = cls._tau_bracket()
        try:
            rho = float(brentq(lambda r: cls(r, df=df).tau() - tau, lo, hi, xtol=1e-12))
        except ValueError as exc:
            raise ValueError(
                f"Kendall's tau = {tau} is not attainable by the t-EV family at df={df}"
            ) from exc
        return cls(rho, df=df, **kwargs)
