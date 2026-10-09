r"""Two-parameter Archimedean copulas: BB1 and BB7.

A one-parameter family ties the strength of dependence to its tail behaviour:
fix Clayton's :math:`\theta` and you have fixed both its Kendall's tau and its
lower-tail coefficient. The BB families of Joe and Hu (1996) add a second
parameter so the two tails can be set **separately**, which is what a
pair-copula in a vine often needs -- equity pairs, for example, usually crash
together more than they rally together, but rarely not at all in either tail.

Both are Archimedean, :math:`C(u, v) = \psi\bigl(\psi^{-1}(u) + \psi^{-1}(v)\bigr)`,
with a generator built by composing two familiar ones:

* **BB1** (Clayton-Gumbel), :math:`\psi(s) = (1 + s^{1/\delta})^{-1/\theta}`
  with :math:`\theta > 0,\ \delta \ge 1`. Lower-tail dependence
  :math:`2^{-1/(\theta\delta)}`, upper-tail :math:`2 - 2^{1/\delta}`.
  :math:`\delta = 1` is Clayton; :math:`\theta \to 0` is Gumbel.
* **BB7** (Joe-Clayton), :math:`\psi(s) = 1 - \bigl(1 - (1+s)^{-1/\delta}\bigr)^{1/\theta}`
  with :math:`\theta \ge 1,\ \delta > 0`. Lower-tail dependence
  :math:`2^{-1/\delta}`, upper-tail :math:`2 - 2^{1/\theta}`.
  :math:`\theta = 1` is Clayton; :math:`\delta \to 0` is Joe.

Everything a vine needs is in closed form: the density, the distribution
function and both conditional distributions (h-functions). The inverse
h-function is solved numerically by a safeguarded Newton iteration, since no
closed form exists. Sampling uses the frailty (mixture) representation of each
generator, so it is exact and vectorised.

============================  ================================================
:class:`BB1Copula`            Clayton-Gumbel: both tails, set separately.
:class:`BB7Copula`            Joe-Clayton: both tails, set separately.
============================  ================================================

Parameter names and ranges follow R's ``VineCopula`` package (families 7 and
9), so fitted values can be compared directly.

References
----------
Joe, H. and Hu, T. (1996). Multivariate distributions from mixtures of
    max-infinitely divisible distributions.
    *Journal of Multivariate Analysis* 57(2), 240-265.
Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall/CRC,
    sections 4.17 (BB1) and 4.23 (BB7) -- the formulas implemented here.
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer,
    Theorem 5.1.6 -- Kendall's tau of an Archimedean copula from its generator.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate

from rcopula.core.base import Copula, TailDependence
from rcopula.core.measures import rho_by_quadrature
from rcopula.special.stable import rsibuya, rstable_positive

__all__ = ["BB1Copula", "BB7Copula"]

#: Points are clipped this far inside the unit square before evaluation, as
#: :func:`rcopula.transforms.conditional_cdf` does.
_EPS = 1e-12


def _log_expm1(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """``log(exp(x) - 1)`` for ``x > 0``, accurate at both ends."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > 30.0, x + np.log1p(-np.exp(-np.minimum(x, 700.0))), np.log(np.expm1(x)))


def _log1mexp(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """``log(1 - exp(-x))`` for ``x > 0`` (Maechler 2012)."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x > np.log(2.0), np.log1p(-np.exp(-x)), np.log(-np.expm1(-x)))


def _points(u: ArrayLike) -> NDArray[np.float64]:
    arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
    if arr.shape[1] != 2:
        raise ValueError(f"expected points with two columns, got {arr.shape[1]}")
    return np.clip(arr, _EPS, 1.0 - _EPS)


class _BBCopula(Copula):
    """Shared machinery: validation, h-functions, inverse h, sampling hooks.

    Subclasses provide ``_log_h(x, cond, theta, delta)`` (log of
    :math:`P(U \\le x \\mid V = cond)`), ``_logpdf`` and ``_cdf``.
    """

    param_names = ("theta", "delta")

    def __init__(
        self,
        theta: float = np.nan,
        delta: float = np.nan,
        dim: int = 2,
        *,
        free: ArrayLike | None = None,
    ) -> None:
        if int(dim) != 2:
            raise ValueError(f"the {self.name} copula is bivariate only, got dim={int(dim)}")
        super().__init__([theta, delta], 2, free=free)

    @property
    def theta(self) -> float:
        """The first parameter, ``theta``.

        Returns
        -------
        float
            ``theta``, or ``nan`` if unspecified.
        """
        return float(self._params[0])

    @property
    def delta(self) -> float:
        """The second parameter, ``delta``.

        Returns
        -------
        float
            ``delta``, or ``nan`` if unspecified.
        """
        return float(self._params[1])

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> _BBCopula:
        values = np.atleast_1d(np.asarray(params, dtype=np.float64))
        return type(self)(float(values[0]), float(values[1]), free=free)

    def _validate_params(self) -> None:
        super()._validate_params()
        # The bounds are inclusive in the base check; the open ends are not
        # admissible (the formulas divide by them).
        for value, lo, nm in zip(self._params, self._excluded, self.param_names, strict=True):
            if lo is not None and not np.isnan(value) and value <= lo:
                raise ValueError(f"{self.name} copula: parameter {nm}={value!r} must exceed {lo}")

    #: Lower bounds that are *excluded* (``None`` where the bound is attained).
    _excluded: tuple[float | None, ...] = (None, None)

    # -- conditional distributions ---------------------------------------

    def _log_h(
        self, x: NDArray[np.float64], cond: NDArray[np.float64], theta: float, delta: float
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    def hfunc(self, u: ArrayLike, given: int = 1) -> NDArray[np.float64]:
        r"""Conditional probability of one coordinate given the other (the h-function).

        Closed form for this family, so it is exact and much faster than
        differentiating the CDF numerically.

        Parameters
        ----------
        u : array_like of float, shape (n, 2)
            Points in the unit square, one per row. Values are clipped to
            ``[1e-12, 1 - 1e-12]``.
        given : {0, 1}, default 1
            The column to condition on. ``given=1`` returns
            :math:`P(U_1 \le u_1 \mid U_2 = u_2) = \partial C/\partial u_2`;
            ``given=0`` returns :math:`P(U_2 \le u_2 \mid U_1 = u_1)`.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Values in ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``given`` is not 0 or 1, ``u`` does not have two columns, or a
            parameter is unspecified.

        Examples
        --------
        >>> import numpy as np
        >>> from rcopula import BB1Copula
        >>> cop = BB1Copula(0.8, 1.6)
        >>> h = cop.hfunc([[0.3, 0.6]])
        >>> bool(0.0 < h[0] < 1.0)
        True
        """
        if given not in (0, 1):
            raise ValueError(f"given must be 0 or 1, got {given}")
        self._require_specified()
        arr = _points(u)
        x, cond = arr[:, 1 - given], arr[:, given]
        theta, delta = float(self._params[0]), float(self._params[1])
        return np.clip(np.exp(self._log_h(x, cond, theta, delta)), 0.0, 1.0)

    def hinv(self, w: ArrayLike, cond: ArrayLike, given: int = 1) -> NDArray[np.float64]:
        r"""Invert the h-function: the coordinate value with a given conditional probability.

        Solves :math:`h(x \mid \text{cond}) = w` for ``x`` by Newton's method
        (the derivative of the h-function is the density), kept inside a
        bisection bracket so it cannot diverge.

        Parameters
        ----------
        w : float or array_like of float, shape (n,)
            Target conditional probabilities in ``[0, 1]``.
        cond : float or array_like of float, shape (n,)
            Values of the conditioning coordinate in ``[0, 1]``; broadcast
            against ``w``.
        given : {0, 1}, default 1
            Which coordinate ``cond`` is (see :meth:`hfunc`). The family is
            exchangeable, so the answer does not depend on it; it is accepted
            for a uniform interface.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            The solved values, in ``[1e-12, 1 - 1e-12]``.

        Raises
        ------
        ValueError
            If ``given`` is not 0 or 1 or a parameter is unspecified.

        Examples
        --------
        >>> import numpy as np
        >>> from rcopula import BB7Copula
        >>> cop = BB7Copula(1.5, 2.0)
        >>> x = cop.hinv([0.1, 0.5, 0.9], [0.3, 0.6, 0.8])
        >>> bool(np.allclose(cop.hfunc(np.column_stack([x, [0.3, 0.6, 0.8]])), [0.1, 0.5, 0.9]))
        True
        """
        if given not in (0, 1):
            raise ValueError(f"given must be 0 or 1, got {given}")
        self._require_specified()
        target, c = np.broadcast_arrays(
            np.atleast_1d(np.asarray(w, dtype=np.float64)),
            np.atleast_1d(np.asarray(cond, dtype=np.float64)),
        )
        target = np.clip(target, _EPS, 1.0 - _EPS)
        c = np.clip(c, _EPS, 1.0 - _EPS)
        theta, delta = float(self._params[0]), float(self._params[1])
        lo = np.full(target.shape, _EPS)
        hi = np.full(target.shape, 1.0 - _EPS)
        x = np.full(target.shape, 0.5)
        for _ in range(100):
            value = np.exp(self._log_h(x, c, theta, delta))
            below = value < target
            lo = np.where(below, x, lo)
            hi = np.where(below, hi, x)
            density = np.exp(self._logpdf(np.column_stack([x, c]), self._params))
            with np.errstate(divide="ignore", invalid="ignore"):
                step = x - (value - target) / density
            inside = np.isfinite(step) & (step > lo) & (step < hi)
            new = np.where(inside, step, 0.5 * (lo + hi))
            done = np.max(np.abs(new - x)) < 1e-14
            x = new
            if done:
                break
        return np.clip(x, _EPS, 1.0 - _EPS)

    # -- shared numerics ----------------------------------------------------

    def _frailty_rvs(
        self, frailty: NDArray[np.float64], rng: np.random.Generator, params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Marshall-Olkin: ``U_j = psi(E_j / V)`` with ``E_j`` standard exponential."""
        e = rng.standard_exponential(size=(frailty.size, 2))
        return self._psi(e / frailty[:, None], float(params[0]), float(params[1]))

    def _psi(self, s: NDArray[np.float64], theta: float, delta: float) -> NDArray[np.float64]:
        raise NotImplementedError

    def rho(self) -> float:
        """Return Spearman's rho, a rank correlation in ``[-1, 1]``, by numerical integration.

        Returns
        -------
        float
            Spearman's rho, between 0 and 1 for these positively dependent
            families.
        """
        self._require_specified()
        return rho_by_quadrature(self)


class BB1Copula(_BBCopula):
    r"""Clayton-Gumbel copula: two parameters that set the lower and upper tails separately.

    A two-variable Archimedean copula with tail dependence in **both** tails,
    each controlled separately -- which neither Clayton (lower only) nor
    Gumbel (upper only) can do, and which the Student t can do only
    symmetrically. A common choice for equity pairs in a vine. Positive
    dependence only; rotate it (:class:`~rcopula.RotatedCopula`) for negative.

    .. math::

        C(u, v) = \Bigl(1 + \bigl[(u^{-\theta} - 1)^{\delta}
                  + (v^{-\theta} - 1)^{\delta}\bigr]^{1/\delta}\Bigr)^{-1/\theta}

    Parameters
    ----------
    theta : float, default nan
        Lower-tail shape, ``theta > 0``. ``nan`` leaves it to be fitted.
    delta : float, default nan
        Upper-tail shape, ``delta >= 1``. ``nan`` leaves it to be fitted.
    dim : int, default 2
        Number of variables; must be exactly 2.
    free : array_like of bool, shape (2,), or None, default None
        Keyword-only. Which of ``(theta, delta)`` are estimated by fitting;
        ``None`` means both.

    Attributes
    ----------
    theta : float
        Lower-tail shape parameter.
    delta : float
        Upper-tail shape parameter.

    Raises
    ------
    ValueError
        If ``dim != 2``, ``theta <= 0`` or ``delta < 1``.

    Notes
    -----
    Kendall's tau is :math:`1 - 2/(\delta(\theta + 2))`; the tail coefficients
    are :math:`\lambda_L = 2^{-1/(\theta\delta)}` and
    :math:`\lambda_U = 2 - 2^{1/\delta}` (Joe 2014, section 4.17). With
    ``delta = 1`` it is exactly the Clayton copula with the same ``theta``.
    Parameters match R ``VineCopula``'s family 7 (``par = theta``,
    ``par2 = delta``).

    Examples
    --------
    >>> from rcopula import BB1Copula, ClaytonCopula
    >>> cop = BB1Copula(0.5, 1.5)
    >>> round(cop.tau(), 6)
    0.466667
    >>> lam = cop.lambda_()
    >>> round(lam.lower, 4), round(lam.upper, 4)
    (0.3969, 0.4126)

    With ``delta = 1`` it reduces to Clayton:

    >>> import numpy as np
    >>> pts = np.array([[0.2, 0.7], [0.6, 0.4]])
    >>> bool(np.allclose(BB1Copula(2.0, 1.0).pdf(pts), ClaytonCopula(2.0).pdf(pts)))
    True
    """

    name = "BB1"

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Admissible ranges of ``(theta, delta)``.

        Returns
        -------
        list of tuple of (float, float)
            ``[(0.0, inf), (1.0, inf)]``; ``theta = 0`` itself is excluded.
        """
        return [(0.0, np.inf), (1.0, np.inf)]

    _excluded = (0.0, None)

    @staticmethod
    def _log_x(u: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
        """``log(u^-theta - 1)``."""
        return _log_expm1(-theta * np.log(u))

    def _parts(self, u: NDArray[np.float64], theta: float, delta: float):
        x, y = u[:, 0], u[:, 1]
        lx, ly = self._log_x(x, theta), self._log_x(y, theta)
        log_s = np.logaddexp(delta * lx, delta * ly)
        log_w = log_s / delta
        return lx, ly, log_s, log_w

    def _cdf(self, u, params):
        theta, delta = float(params[0]), float(params[1])
        _, _, _, log_w = self._parts(np.clip(u, _EPS, 1.0 - _EPS), theta, delta)
        return np.exp(-np.logaddexp(0.0, log_w) / theta)

    def _log_h(self, x, cond, theta, delta):
        _, ly, log_s, log_w = self._parts(np.column_stack([x, cond]), theta, delta)
        return (
            -(1.0 / theta + 1.0) * np.logaddexp(0.0, log_w)
            + (1.0 / delta - 1.0) * log_s
            + (delta - 1.0) * ly
            - (theta + 1.0) * np.log(cond)
        )

    def _logpdf(self, u, params):
        theta, delta = float(params[0]), float(params[1])
        u = np.clip(u, _EPS, 1.0 - _EPS)
        x = u[:, 0]
        lx, _, log_s, log_w = self._parts(u, theta, delta)
        # w / (1 + w), computed without overflow.
        ratio = np.exp(log_w - np.logaddexp(0.0, log_w))
        return (
            self._log_h(x, u[:, 1], theta, delta)
            - (theta + 1.0) * np.log(x)
            + (delta - 1.0) * lx
            - log_s
            + np.log((theta + 1.0) * ratio + theta * (delta - 1.0))
        )

    def _psi(self, s, theta, delta):
        return np.exp(-np.log1p(s ** (1.0 / delta)) / theta)

    def _rvs(self, size, params, rng):
        theta, delta = float(params[0]), float(params[1])
        # V = M^delta * S: M ~ Gamma(1/theta) mixes a positive stable S whose
        # Laplace transform is exp(-s^(1/delta)), giving (1 + s^(1/delta))^(-1/theta).
        m = rng.gamma(1.0 / theta, size=size)
        stable = rstable_positive(size, 1.0 / delta, rng) if delta > 1.0 else np.ones(size)
        frailty = m**delta * stable
        return np.clip(
            self._frailty_rvs(frailty, rng, params), np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
        )

    def tau(self) -> float:
        r"""Return Kendall's tau, a rank correlation in ``[0, 1)``.

        Closed form: :math:`1 - 2/(\delta(\theta + 2))`.

        Returns
        -------
        float
            Kendall's tau.

        Examples
        --------
        >>> from rcopula import BB1Copula
        >>> round(BB1Copula(2.0, 2.0).tau(), 6)
        0.75
        """
        self._require_specified()
        return 1.0 - 2.0 / (self.delta * (self.theta + 2.0))

    def lambda_(self) -> TailDependence:
        r"""Return the lower and upper tail-dependence coefficients.

        :math:`\lambda_L = 2^{-1/(\theta\delta)}` and
        :math:`\lambda_U = 2 - 2^{1/\delta}`.

        Returns
        -------
        TailDependence
            Named tuple ``(lower, upper)`` of floats in ``[0, 1)``.
        """
        self._require_specified()
        return TailDependence(
            lower=float(2.0 ** (-1.0 / (self.theta * self.delta))),
            upper=float(2.0 - 2.0 ** (1.0 / self.delta)),
        )


class BB7Copula(_BBCopula):
    r"""Joe-Clayton copula: two parameters that set the upper and lower tails separately.

    A two-variable Archimedean copula with tail dependence in both tails, the
    upper governed by ``theta`` (Joe-like) and the lower by ``delta``
    (Clayton-like). An alternative to :class:`BB1Copula` with a different
    shape in between the tails. Positive dependence only; rotate it
    (:class:`~rcopula.RotatedCopula`) for negative.

    .. math::

        C(u, v) = 1 - \Bigl(1 - \bigl[(1 - \bar u^{\theta})^{-\delta}
                  + (1 - \bar v^{\theta})^{-\delta} - 1\bigr]^{-1/\delta}\Bigr)^{1/\theta},
        \qquad \bar u = 1 - u.

    Parameters
    ----------
    theta : float, default nan
        Upper-tail shape, ``theta >= 1``. ``nan`` leaves it to be fitted.
    delta : float, default nan
        Lower-tail shape, ``delta > 0``. ``nan`` leaves it to be fitted.
    dim : int, default 2
        Number of variables; must be exactly 2.
    free : array_like of bool, shape (2,), or None, default None
        Keyword-only. Which of ``(theta, delta)`` are estimated by fitting;
        ``None`` means both.

    Attributes
    ----------
    theta : float
        Upper-tail shape parameter.
    delta : float
        Lower-tail shape parameter.

    Raises
    ------
    ValueError
        If ``dim != 2``, ``theta < 1`` or ``delta <= 0``.

    Notes
    -----
    Tail coefficients :math:`\lambda_L = 2^{-1/\delta}` and
    :math:`\lambda_U = 2 - 2^{1/\theta}` (Joe 2014, section 4.23). Kendall's
    tau has no closed form and is computed from the generator by
    one-dimensional quadrature. With ``theta = 1`` it is exactly the Clayton
    copula with parameter ``delta``. Parameters match R ``VineCopula``'s
    family 9 (``par = theta``, ``par2 = delta``).

    Examples
    --------
    >>> from rcopula import BB7Copula, ClaytonCopula
    >>> cop = BB7Copula(1.5, 2.0)
    >>> lam = cop.lambda_()
    >>> round(lam.lower, 4), round(lam.upper, 4)
    (0.7071, 0.4126)
    >>> round(cop.tau(), 4)
    0.5359

    With ``theta = 1`` it reduces to Clayton:

    >>> import numpy as np
    >>> pts = np.array([[0.2, 0.7], [0.6, 0.4]])
    >>> bool(np.allclose(BB7Copula(1.0, 2.0).pdf(pts), ClaytonCopula(2.0).pdf(pts)))
    True
    """

    name = "BB7"

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Admissible ranges of ``(theta, delta)``.

        Returns
        -------
        list of tuple of (float, float)
            ``[(1.0, inf), (0.0, inf)]``; ``delta = 0`` itself is excluded.
        """
        return [(1.0, np.inf), (0.0, np.inf)]

    _excluded = (None, 0.0)

    @staticmethod
    def _log_a(u: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
        """``log(1 - (1-u)^theta)``."""
        return _log1mexp(-theta * np.log1p(-u))

    def _log_one_plus_s(
        self, la: NDArray[np.float64], lb: NDArray[np.float64], delta: float
    ) -> NDArray[np.float64]:
        """``log((1-ubar^theta)^-delta + (1-vbar^theta)^-delta - 1)``, from the log-A terms."""
        pa, pb = -delta * la, -delta * lb  # both >= 0
        top = np.maximum(pa, pb)
        return top + np.log(np.exp(pa - top) + np.exp(pb - top) - np.exp(-top))

    def _cdf(self, u, params):
        theta, delta = float(params[0]), float(params[1])
        u = np.clip(u, _EPS, 1.0 - _EPS)
        log1s = self._log_one_plus_s(
            self._log_a(u[:, 0], theta), self._log_a(u[:, 1], theta), delta
        )
        # 1 - z with z = (1+s)^(-1/delta)
        log_one_minus_z = _log1mexp(log1s / delta)
        return -np.expm1(log_one_minus_z / theta)

    def _log_h(self, x, cond, theta, delta):
        la, lb = self._log_a(x, theta), self._log_a(cond, theta)
        log1s = self._log_one_plus_s(la, lb, delta)
        log_one_minus_z = _log1mexp(log1s / delta)
        return (
            (1.0 / theta - 1.0) * log_one_minus_z
            - (1.0 / delta + 1.0) * log1s
            + (theta - 1.0) * np.log1p(-cond)
            - (delta + 1.0) * lb
        )

    def _logpdf(self, u, params):
        theta, delta = float(params[0]), float(params[1])
        u = np.clip(u, _EPS, 1.0 - _EPS)
        x, y = u[:, 0], u[:, 1]
        la = self._log_a(x, theta)
        log1s = self._log_one_plus_s(la, self._log_a(y, theta), delta)
        log_one_minus_z = _log1mexp(log1s / delta)
        # z / (1 - z)
        odds = np.exp(-log1s / delta - log_one_minus_z)
        return (
            self._log_h(x, y, theta, delta)
            + np.log(theta)
            + (theta - 1.0) * np.log1p(-x)
            - (delta + 1.0) * la
            - log1s
            + np.log((1.0 + delta) + (1.0 - 1.0 / theta) * odds)
        )

    def _psi(self, s, theta, delta):
        # 1 - (1 - (1+s)^(-1/delta))^(1/theta)
        inner = _log1mexp(np.log1p(s) / delta)
        return -np.expm1(inner / theta)

    def _rvs(self, size, params, rng):
        theta, delta = float(params[0]), float(params[1])
        # V | M ~ Gamma(M / delta): M is the Sibuya frailty of Joe's generator,
        # and Gamma(1/delta) is the frailty of Clayton's, composed.
        m = rsibuya(size, 1.0 / theta, rng) if theta > 1.0 else np.ones(size)
        frailty = rng.gamma(m / delta)
        return np.clip(
            self._frailty_rvs(frailty, rng, params), np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
        )

    def tau(self) -> float:
        r"""Return Kendall's tau, a rank correlation in ``[0, 1)``.

        No closed form exists; it is computed as
        :math:`1 + 4\int_0^1 \varphi(t)/\varphi'(t)\,dt` with the generator
        inverse :math:`\varphi(t) = (1 - (1-t)^\theta)^{-\delta} - 1`, by
        adaptive quadrature.

        Returns
        -------
        float
            Kendall's tau.

        Examples
        --------
        >>> from rcopula import BB7Copula
        >>> round(BB7Copula(1.0, 2.0).tau(), 8)  # theta = 1 is Clayton(2): tau = 0.5
        0.5
        """
        self._require_specified()
        theta, delta = self.theta, self.delta

        def ratio(t: float) -> float:
            a = 1.0 - (1.0 - t) ** theta
            if a <= 0.0:
                return 0.0
            return -(a - a ** (delta + 1.0)) / (delta * theta * (1.0 - t) ** (theta - 1.0))

        value, _ = integrate.quad(ratio, 0.0, 1.0, limit=200, epsabs=1e-13, epsrel=1e-12)
        return float(1.0 + 4.0 * value)

    def lambda_(self) -> TailDependence:
        r"""Return the lower and upper tail-dependence coefficients.

        :math:`\lambda_L = 2^{-1/\delta}` and :math:`\lambda_U = 2 - 2^{1/\theta}`.

        Returns
        -------
        TailDependence
            Named tuple ``(lower, upper)`` of floats in ``[0, 1)``.
        """
        self._require_specified()
        return TailDependence(
            lower=float(2.0 ** (-1.0 / self.delta)),
            upper=float(2.0 - 2.0 ** (1.0 / self.theta)),
        )
