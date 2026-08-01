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
    r"""Pickands function of the Gumbel-Hougaard copula.

    :math:`A(t) = (t^\theta + (1-t)^\theta)^{1/\theta}`.

    Provided for comparison: Gumbel is the unique family that is simultaneously
    Archimedean and extreme-value.

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
    """Base class for bivariate extreme-value copulas.

    Subclasses supply :meth:`A`; supplying :meth:`_dA` and :meth:`_d2A` as well
    makes the density exact rather than finite-differenced.
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
        """The Pickands dependence function on ``[0, 1]``."""

    def dA(self, w: ArrayLike) -> NDArray[np.float64]:
        """First derivative of :meth:`A`."""
        t = np.asarray(w, dtype=np.float64)
        h = _FD_STEP
        tc = np.clip(t, h, 1.0 - h)
        return (self.A(tc + h) - self.A(tc - h)) / (2.0 * h)

    def d2A(self, w: ArrayLike) -> NDArray[np.float64]:
        """Second derivative of :meth:`A`. Non-negative, since ``A`` is convex."""
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
        r"""Kendall's tau, :math:`\int_0^1 \frac{t(1-t)}{A(t)}\,\mathrm{d}A'(t)`.

        Evaluated as :math:`\int_0^1 t(1-t) A''(t)/A(t)\,\mathrm{d}t` on a
        Gauss-Legendre grid.
        """
        self._require_specified()
        nodes, weights = np.polynomial.legendre.leggauss(400)
        t = 0.5 * (nodes + 1.0)
        wt = 0.5 * weights
        integrand = t * (1.0 - t) * self.d2A(t) / self.A(t)
        return float(np.sum(wt * integrand))

    def rho(self) -> float:
        r"""Spearman's rho, :math:`12\int_0^1 (A(t)+1)^{-2}\,\mathrm{d}t - 3`."""
        self._require_specified()
        nodes, weights = np.polynomial.legendre.leggauss(400)
        t = 0.5 * (nodes + 1.0)
        wt = 0.5 * weights
        return float(12.0 * np.sum(wt / (self.A(t) + 1.0) ** 2) - 3.0)

    def lambda_(self) -> TailDependence:
        r"""Always :math:`(0,\ 2(1 - A(1/2)))` for an extreme-value copula."""
        self._require_specified()
        return TailDependence(lower=0.0, upper=float(2.0 * (1.0 - self.A(0.5))))

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> ExtremeValueCopula:
        """Calibrate to a target Kendall's tau by root-finding on ``tau(theta)``."""
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
    r"""Galambos copula.

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
        return float(self._params[0])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
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
        r""":math:`1 - g^{-1/\theta}`, written as ``-expm1`` so it never cancels."""
        t = self._clip(w)
        return -np.expm1(-self._log_g(t) / self.theta)

    def dA(self, w):
        r""":math:`A'(t) = -g^{-1/\theta - 1}\,(t^{-\theta-1} - (1-t)^{-\theta-1})`.

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
        r""":math:`A''(t) = (1 + \theta)\,\bigl(t(1-t)\bigr)^{-\theta-2}\,g^{-1/\theta-2}`.

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
    r"""Husler-Reiss copula.

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
        return float(self._params[0])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, np.inf)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> HuslerReissCopula:
        return HuslerReissCopula(float(np.atleast_1d(params)[0]), free=free)

    def _z(self, t):
        th = self.theta
        ell = np.log(t / (1.0 - t))
        return 1.0 / th + 0.5 * th * ell, 1.0 / th - 0.5 * th * ell

    def A(self, w):
        t = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        z1, z2 = self._z(t)
        return t * ndtr(z1) + (1.0 - t) * ndtr(z2)

    def dA(self, w):
        r"""Exactly :math:`\Phi(z_1) - \Phi(z_2)`.

        The two density terms cancel: since
        :math:`\phi(z_1)/\phi(z_2) = e^{-\ell} = (1-t)/t`, we get
        :math:`t\phi(z_1) = (1-t)\phi(z_2)` and their contributions to
        :math:`A'` are equal and opposite.
        """
        t = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        z1, z2 = self._z(t)
        return ndtr(z1) - ndtr(z2)

    def d2A(self, w):
        r"""Differentiating :meth:`dA` once more.

        With :math:`z_1' = \tfrac{\theta}{2t(1-t)}` and :math:`z_2' = -z_1'`,

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
    r"""Tawn copula, in R's one-parameter form.

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
        return float(self._params[0])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> TawnCopula:
        return TawnCopula(float(np.atleast_1d(params)[0]), free=free)

    def A(self, w):
        t = np.asarray(w, dtype=np.float64)
        return 1.0 - self.theta * t * (1.0 - t)

    def dA(self, w):
        t = np.asarray(w, dtype=np.float64)
        return self.theta * (2.0 * t - 1.0)

    def d2A(self, w):
        t = np.asarray(w, dtype=np.float64)
        return np.full(t.shape, 2.0 * self.theta)

    @classmethod
    def _tau_bracket(cls) -> tuple[float, float]:
        return (1e-10, 1.0)


class TEVCopula(ExtremeValueCopula):
    r"""t-EV copula: the extreme-value limit of the Student-t copula.

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
        return float(self._params[0])

    @property
    def df(self) -> float:
        return float(self._params[1])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
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
        t = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        nu = self.df
        z1, z2, _, _ = self._z_and_dz(t)
        return t * student_t.cdf(z1, df=nu + 1.0) + (1.0 - t) * student_t.cdf(z2, df=nu + 1.0)

    def dA(self, w):
        r"""Exactly :math:`T_{\nu+1}(z_1) - T_{\nu+1}(z_2)`.

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
        r""":math:`A''(t) = f(z_1) z_1' - f(z_2) z_2'`, both terms positive."""
        t = np.clip(np.asarray(w, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        nu = self.df
        z1, z2, dz1, dz2 = self._z_and_dz(t)
        return student_t.pdf(z1, df=nu + 1.0) * dz1 - student_t.pdf(z2, df=nu + 1.0) * dz2

    @classmethod
    def _tau_bracket(cls) -> tuple[float, float]:
        return (-0.999, 0.999)

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> TEVCopula:
        """Calibrate ``rho`` to a target tau at fixed ``df``."""
        df = kwargs.pop("df", 4.0)
        lo, hi = cls._tau_bracket()
        try:
            rho = float(brentq(lambda r: cls(r, df=df).tau() - tau, lo, hi, xtol=1e-12))
        except ValueError as exc:
            raise ValueError(
                f"Kendall's tau = {tau} is not attainable by the t-EV family at df={df}"
            ) from exc
        return cls(rho, df=df, **kwargs)
