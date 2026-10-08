r"""Archimedean copulas.

An Archimedean copula is built from a single univariate *generator*
:math:`\psi : [0, \infty) \to [0, 1]`:

.. math::

    C(u_1, \dots, u_d) = \psi\bigl(\psi^{-1}(u_1) + \dots + \psi^{-1}(u_d)\bigr)

with density

.. math::

    c(\mathbf{u}) = \frac{\bigl|\psi^{(d)}\bigl(\sum_j \psi^{-1}(u_j)\bigr)\bigr|}
                         {\prod_j \bigl|\psi'\bigl(\psi^{-1}(u_j)\bigr)\bigr|} .

The whole family therefore reduces to: the generator, its inverse, its first
derivative, and its :math:`d`-th derivative. Those four are what
:class:`ArchimedeanGenerator` requires, and they are deliberately *stateless*
functions of ``(t, theta)`` rather than methods on a parameterised object, so
that a likelihood optimiser never has to allocate anything in its inner loop.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer, Chapter 4
    and Table 4.1, for the generators and closed-form dependence measures.
Hofert, M., Mächler, M. and McNeil, A. J. (2012). Likelihood inference for
    Archimedean copulas in high dimensions under known margins.
    *Journal of Multivariate Analysis* 110, 133-150.
    For the numerically stable form of the d-th generator derivative.
McNeil, A. J. and Nešlehová, J. (2009). Multivariate Archimedean copulas,
    d-monotone functions and l1-norm symmetric distributions.
    *Annals of Statistics* 37(5B), 3059-3097.
    For when a generator actually yields a valid copula in dimension d.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import digamma, zeta

from rcopula.core.base import Copula, TailDependence
from rcopula.special.combinatorics import eulerian_all, stirling1_all, stirling2_all
from rcopula.special.debye import debye1, debye2
from rcopula.special.logexp import log1mexp, log1pexp, signed_logsumexp
from rcopula.special.stable import rlog_series, rsibuya, rstable_positive

#: Gauss-Legendre nodes per axis for the generic Spearman-rho quadrature.
_RHO_NODES = 256

#: Largest |theta| considered when bracketing an inversion. Beyond this every
#: family is numerically indistinguishable from the comonotone copula.
_THETA_MAX = 1e4

#: Euler-Mascheroni constant, used in Joe's digamma expression for tau.
_EULER_GAMMA = 0.5772156649015328606

#: Rungs per branch of the inversion bracketing ladder.
_LADDER_RUNGS = 24

__all__ = [
    "AMHCopula",
    "ArchimedeanCopula",
    "ArchimedeanGenerator",
    "ClaytonCopula",
    "FrankCopula",
    "GumbelCopula",
    "JoeCopula",
]


# ======================================================================
# Generators
# ======================================================================


class ArchimedeanGenerator(ABC):
    """The one-variable function that defines an Archimedean copula family.

    An Archimedean copula is fully determined by a single decreasing function
    of one variable, the *generator* :math:`\\psi`. This abstract class
    collects everything a family must know about its generator: the function
    itself, its inverse, the logs of its derivatives, its admissible
    parameter range, closed-form dependence measures and how to sample it.

    You only need this class to define a new Archimedean family; to *use* one,
    take :class:`ClaytonCopula`, :class:`GumbelCopula`, :class:`FrankCopula`,
    :class:`JoeCopula` or :class:`AMHCopula`, or wrap a generator in
    :class:`ArchimedeanCopula`.

    Every method is stateless: the parameter ``theta`` is passed in on each
    call rather than stored on the object, so a likelihood optimiser can
    evaluate many ``theta`` values without allocating anything.

    Attributes
    ----------
    name : str
        Human-readable family name, e.g. ``"Clayton"``.
    param_name : str
        Name of the single dependence parameter, ``"theta"`` by default.
    """

    name: str = "generator"
    param_name: str = "theta"

    @abstractmethod
    def bounds(self, dim: int) -> tuple[float, float]:
        """Return the allowed range of ``theta`` for a given dimension.

        Some families allow negative dependence only in two dimensions, so the
        range can shrink as ``dim`` grows.

        Parameters
        ----------
        dim : int
            Copula dimension, at least 2.

        Returns
        -------
        tuple of (float, float)
            ``(lower, upper)`` bounds for ``theta``; either may be infinite.
        """

    @abstractmethod
    def psi(self, t: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
        """Evaluate the generator :math:`\\psi(t)`.

        Parameters
        ----------
        t : numpy.ndarray of float, any shape
            Non-negative arguments.
        theta : float
            Dependence parameter, inside :meth:`bounds`.

        Returns
        -------
        numpy.ndarray of float, same shape as ``t``
            Values in ``[0, 1]``; :math:`\\psi(0) = 1` and :math:`\\psi` decreases
            towards 0.
        """

    @abstractmethod
    def ipsi(self, u: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
        """Evaluate the inverse generator :math:`\\psi^{-1}(u)`.

        Parameters
        ----------
        u : numpy.ndarray of float, any shape
            Values in ``[0, 1]``.
        theta : float
            Dependence parameter, inside :meth:`bounds`.

        Returns
        -------
        numpy.ndarray of float, same shape as ``u``
            Non-negative values; :math:`\\psi^{-1}(1) = 0`.
        """

    @abstractmethod
    def log_abs_dpsi(self, t: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
        """Return the log of the absolute first derivative, :math:`\\log|\\psi'(t)|`.

        Parameters
        ----------
        t : numpy.ndarray of float, any shape
            Non-negative arguments.
        theta : float
            Dependence parameter, inside :meth:`bounds`.

        Returns
        -------
        numpy.ndarray of float, same shape as ``t``
            ``log|psi'(t)|``; ``-inf`` where the derivative is zero.
        """

    @abstractmethod
    def log_abs_dpsi_d(self, t: NDArray[np.float64], theta: float, d: int) -> NDArray[np.float64]:
        """Return the log of the absolute ``d``-th derivative, :math:`\\log|\\psi^{(d)}(t)|`.

        This is the numerator of the ``d``-dimensional copula density.

        Parameters
        ----------
        t : numpy.ndarray of float, any shape
            Non-negative arguments.
        theta : float
            Dependence parameter, inside :meth:`bounds`.
        d : int
            Order of the derivative (the copula dimension), at least 1.

        Returns
        -------
        numpy.ndarray of float, same shape as ``t``
            ``log|psi^(d)(t)|``.
        """

    @abstractmethod
    def tau(self, theta: float) -> float:
        """Return Kendall's tau (a rank correlation in ``[-1, 1]``) implied by ``theta``.

        Parameters
        ----------
        theta : float
            Dependence parameter, inside :meth:`bounds`.

        Returns
        -------
        float
            Population Kendall's tau.
        """

    @abstractmethod
    def lambda_(self, theta: float) -> TailDependence:
        """Return how strongly extreme values co-occur (tail dependence) at ``theta``.

        Parameters
        ----------
        theta : float
            Dependence parameter, inside :meth:`bounds`.

        Returns
        -------
        TailDependence
            Named tuple ``(lower, upper)`` of floats in ``[0, 1]``.
        """

    @abstractmethod
    def rvs_frailty(self, size: int, theta: float, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw the random "frailty" variable used to simulate the copula.

        The frailty :math:`V_0` is the positive random variable whose Laplace
        transform is ``psi``; sampling uses :math:`U_j = \\psi(E_j / V_0)` with
        independent standard exponentials :math:`E_j`. Only called when
        :meth:`has_frailty` is true.

        Parameters
        ----------
        size : int
            Number of draws, non-negative.
        theta : float
            Dependence parameter, inside :meth:`bounds`.
        rng : numpy.random.Generator
            Source of randomness.

        Returns
        -------
        numpy.ndarray of float, shape (size,)
            Positive frailty draws.
        """

    def has_frailty(self, theta: float) -> bool:
        """Report whether the fast frailty sampler can be used at this ``theta``.

        Parameters
        ----------
        theta : float
            Dependence parameter, inside :meth:`bounds`.

        Returns
        -------
        bool
            ``True`` if the Marshall-Olkin frailty representation applies (the
            default rule is ``theta > 0``); ``False`` means the copula is
            sampled by conditional inversion instead.

        Notes
        -----
        Sampling an Archimedean copula as :math:`U_j = \\psi(E_j / V)` requires
        :math:`\\psi` to be *completely* monotone, i.e. a Laplace transform. That
        holds only on the positively-dependent half of Clayton, Frank and AMH.
        On the negative half -- which exists in ``d = 2`` and is the reason
        those families are used at all -- there is no frailty, and the copula
        has to be sampled by conditional inversion instead.
        """
        return theta > 0.0

    def is_independent(self, theta: float) -> bool:
        """Report whether this ``theta`` means the variables are independent.

        The independence point is where most generators divide by zero, so
        callers short-circuit there rather than evaluate.

        Parameters
        ----------
        theta : float
            Dependence parameter.

        Returns
        -------
        bool
            ``True`` if ``theta`` gives the independence copula. The default
            rule is ``theta == 0``; Gumbel and Joe use ``theta == 1``.
        """
        return theta == 0.0

    # -- optional log-space paths, for strong dependence ----------------
    #
    # At large ``theta`` the intermediate quantities leave double precision:
    # Clayton's ``psi^{-1}(u) = u^{-theta} - 1`` is ``inf`` for ``u = 0.02`` and
    # ``theta = 1000``, and its frailty ``Gamma(1/theta, 1)`` underflows to
    # exactly zero. Both are removable by never forming the large quantity --
    # only its logarithm. Generators that can do that override these; the
    # defaults reproduce the direct route exactly.

    def rvs_log_frailty(
        self, size: int, theta: float, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Draw the log of the frailty, :math:`\\log V_0`, without underflow.

        Used at strong dependence, where the frailty itself can be too small to
        represent. Default: the log of :meth:`rvs_frailty`.

        Parameters
        ----------
        size : int
            Number of draws, non-negative.
        theta : float
            Dependence parameter, inside :meth:`bounds`.
        rng : numpy.random.Generator
            Source of randomness.

        Returns
        -------
        numpy.ndarray of float, shape (size,)
            Log frailty draws (may be ``-inf`` in the default implementation
            if a draw underflows).
        """
        with np.errstate(divide="ignore"):
            return np.log(self.rvs_frailty(size, theta, rng))

    def psi_from_log_t(self, log_t: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
        """Evaluate the generator when only ``log t`` is available, :math:`\\psi(e^{\\log t})`.

        Lets families avoid forming a huge ``t``. Default: exponentiate, then
        call :meth:`psi`.

        Parameters
        ----------
        log_t : numpy.ndarray of float, any shape
            Natural log of the generator argument.
        theta : float
            Dependence parameter, inside :meth:`bounds`.

        Returns
        -------
        numpy.ndarray of float, same shape as ``log_t``
            Generator values in ``[0, 1]``.
        """
        with np.errstate(over="ignore"):
            return self.psi(np.exp(log_t), theta)

    def log_cdf(self, u: NDArray[np.float64], theta: float, dim: int) -> NDArray[np.float64] | None:
        """Return ``log C(u)`` by a numerically stable family-specific formula, if one exists.

        ``None`` means "no specialised form", and the caller falls back to
        ``psi(sum of psi^{-1})``.

        Parameters
        ----------
        u : numpy.ndarray of float, shape (n, dim)
            Points in the unit cube.
        theta : float
            Dependence parameter, inside :meth:`bounds`.
        dim : int
            Copula dimension.

        Returns
        -------
        numpy.ndarray of float, shape (n,), or None
            Log of the copula CDF at each row, or ``None`` (the default).
        """
        return None

    def log_pdf(self, u: NDArray[np.float64], theta: float, dim: int) -> NDArray[np.float64] | None:
        """Return ``log c(u)`` by a numerically stable family-specific formula, if one exists.

        ``None`` falls back to the generic
        ``log|psi^(d)| - sum log|psi'|``, which needs ``psi^{-1}`` evaluated
        explicitly and therefore inherits its overflow.

        Parameters
        ----------
        u : numpy.ndarray of float, shape (n, dim)
            Points strictly inside the unit cube.
        theta : float
            Dependence parameter, inside :meth:`bounds`.
        dim : int
            Copula dimension.

        Returns
        -------
        numpy.ndarray of float, shape (n,), or None
            Log density at each row, or ``None`` (the default).
        """
        return None

    def rho(self, theta: float) -> float:
        r"""Return Spearman's rho (a rank correlation in ``[-1, 1]``) implied by ``theta``.

        Spearman's rho is the ordinary correlation of the ranks of the two
        variables. It is always computed for the bivariate (``dim=2``) copula.

        Parameters
        ----------
        theta : float
            Dependence parameter, inside ``bounds(2)``.

        Returns
        -------
        float
            Population Spearman's rho, clipped to ``[-1, 1]``.

        Notes
        -----
        The generic implementation evaluates
        :math:`\rho = 12 \int_0^1\!\!\int_0^1 C(u,v)\,du\,dv - 3` on a tensor
        Gauss-Legendre grid. ``scipy.integrate.dblquad`` is a poor fit here: it
        calls the integrand one scalar at a time (so every evaluation pays the
        array-construction cost) and, being adaptive on a smooth-but-peaked
        integrand, it silently returns ~1e-3 accuracy. A fixed 256-node tensor
        rule is both exact to ~1e-14 and a single vectorised call.

        Families with a closed form (Frank, via Debye functions) override this.
        """
        nodes, weights = np.polynomial.legendre.leggauss(_RHO_NODES)
        # Map from [-1, 1] to [0, 1].
        x = 0.5 * (nodes + 1.0)
        w = 0.5 * weights

        cop = ArchimedeanCopula(self, theta, dim=2)
        uu, vv = np.meshgrid(x, x, indexing="ij")
        # Very strong dependence overflows the generator (e.g. Gumbel's
        # (-log u)**theta). The overflow is benign: the inverse generator tends
        # to infinity and the copula correctly tends to the comonotone bound M.
        with np.errstate(over="ignore", invalid="ignore"):
            c = cop.cdf(np.column_stack([uu.ravel(), vv.ravel()])).reshape(uu.shape)
        integral = float(np.einsum("i,j,ij->", w, w, np.nan_to_num(c)))
        # Quadrature of a near-comonotone copula can overshoot the bound by a
        # few ulps, and a rho outside [-1, 1] is worse than one that is merely
        # imprecise.
        return float(np.clip(12.0 * integral - 3.0, -1.0, 1.0))

    def _attained_at_bound(self, func: Any, target: float, dim: int) -> float | None:
        """A finite bound of theta whose measure is exactly ``target``, if any.

        The bracket search only looks strictly inside the theta range, so a
        target sitting *on* a bound -- tau = 0 for Joe or Gumbel, whose
        independence point is the lower bound theta = 1 -- used to be reported
        as unattainable.
        """
        for bound in self.bounds(dim):
            if not np.isfinite(bound):
                continue
            try:
                with np.errstate(all="ignore"):
                    value = float(func(bound))
            except (ZeroDivisionError, ValueError, FloatingPointError):
                continue
            if np.isfinite(value) and abs(value - target) <= 1e-12:
                return float(bound)
        return None

    def _bracket(
        self, func: Callable[[float], float], target: float, dim: int
    ) -> tuple[float, float]:
        """Find ``(a, b)`` bracketing the root of ``func(theta) = target``.

        A fixed wide bracket does not work: the upper end would sit at a theta
        where the generator overflows, and monotone families approach their
        limiting dependence so fast that most of the interval is numerically
        indistinguishable from comonotonicity. Instead, scan a geometric ladder
        of candidate thetas and return the first sign change.
        """
        lo, hi = self.bounds(dim)
        # The ladder has to resolve the neighbourhood of whichever bound the
        # family actually has. A plain geometric ladder from zero works for
        # Clayton and Frank but skips the whole interesting region of Gumbel,
        # whose independence point sits at theta = 1 rather than 0. So anchor
        # extra points just inside each finite bound.
        #
        # theta = 0 is deliberately excluded: it is the independence limit and
        # Clayton and Frank both divide by theta, so it must be approached,
        # never evaluated.
        # Deliberately coarse. Every rung costs one evaluation of `func`, and for
        # Spearman's rho that is a 256x256 quadrature -- a 60-point ladder made
        # `from_rho` take seconds. 24 rungs still isolates a sign change on a
        # monotone curve, and brentq refines from there.
        rungs = [
            -np.geomspace(_THETA_MAX, 1e-6, _LADDER_RUNGS),
            np.geomspace(1e-6, _THETA_MAX, _LADDER_RUNGS),
        ]
        if np.isfinite(lo):
            rungs.append(lo + np.geomspace(1e-9, _THETA_MAX, _LADDER_RUNGS))
        if np.isfinite(hi):
            rungs.append(hi - np.geomspace(1e-9, _THETA_MAX, _LADDER_RUNGS))

        ladder = np.unique(np.concatenate(rungs))
        ladder = ladder[(ladder > lo) & (ladder < hi)]

        def safe(t: float) -> float:
            """Overflow at extreme theta is expected; treat it as "no value here"
            rather than letting it abort the whole search."""
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    return float(func(t)) - target
            except (ZeroDivisionError, ValueError, FloatingPointError):
                return np.nan

        values = np.array([safe(float(t)) for t in ladder])
        finite = np.isfinite(values)
        ladder, values = ladder[finite], values[finite]

        sign_change = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
        if sign_change.size == 0:
            raise ValueError(
                f"target {target} is not attainable by the {self.name} family "
                f"in dimension {dim} (theta range ({lo}, {hi}))"
            )
        i = int(sign_change[0])
        return float(ladder[i]), float(ladder[i + 1])

    def itau(self, tau: float, dim: int = 2) -> float:
        """Find the ``theta`` that gives a target Kendall's tau (R's ``iTau``).

        Parameters
        ----------
        tau : float
            Target Kendall's tau, a rank correlation in ``[-1, 1]``. Must be
            reachable by this family in dimension ``dim``.
        dim : int, default 2
            Copula dimension; it can restrict the reachable range.

        Returns
        -------
        float
            The parameter ``theta`` whose population tau equals ``tau``.

        Raises
        ------
        ValueError
            If ``tau`` is outside the range this family can reach.

        Notes
        -----
        The generic implementation is a bracketed root-find on the monotone
        ``tau(theta)`` curve; families with a closed form override it.
        """
        edge = self._attained_at_bound(self.tau, tau, dim)
        if edge is not None:
            return edge
        a, b = self._bracket(self.tau, tau, dim)
        return float(brentq(lambda th: self.tau(th) - tau, a, b, xtol=1e-14, rtol=8.9e-16))

    def irho(self, rho: float, dim: int = 2) -> float:
        """Find the ``theta`` that gives a target Spearman's rho (R's ``iRho``).

        Parameters
        ----------
        rho : float
            Target Spearman's rho, a rank correlation in ``[-1, 1]``. Must be
            reachable by this family.
        dim : int, default 2
            Copula dimension; it restricts the admissible ``theta`` range.

        Returns
        -------
        float
            The parameter ``theta`` whose population rho equals ``rho``.

        Raises
        ------
        ValueError
            If ``rho`` is outside the range this family can reach.

        Notes
        -----
        A bracketed root-find on the monotone ``rho(theta)`` curve. Each
        evaluation of ``rho`` is a 256x256 quadrature for families without a
        closed form, so this is slower than :meth:`itau`.
        """
        edge = self._attained_at_bound(self.rho, rho, dim)
        if edge is not None:
            return edge
        a, b = self._bracket(self.rho, rho, dim)
        return float(brentq(lambda th: self.rho(th) - rho, a, b, xtol=1e-12, rtol=8.9e-16))


class _ClaytonGenerator(ArchimedeanGenerator):
    r"""Clayton: :math:`\psi(t) = (1 + t)^{-1/\theta}`.

    Nelsen family (4.2.1). Lower-tail dependent, upper-tail independent — the
    canonical choice when joint *crashes* matter more than joint booms.
    """

    name = "Clayton"

    def bounds(self, dim: int) -> tuple[float, float]:
        # Negative dependence is only attainable in d = 2; for d > 2 the
        # generator stops being d-monotone (McNeil & Neslehova 2009).
        return (-1.0, np.inf) if dim == 2 else (0.0, np.inf)

    def psi(self, t, theta):
        # For theta < 0 the generator has *finite support*: psi(t) = 0 once
        # 1 + t <= 0. That region is where C hits the Frechet lower bound, and
        # it is reached for perfectly ordinary (u, v), so it has to be handled
        # rather than left to produce nan from a fractional power of a negative.
        return np.maximum(1.0 + t, 0.0) ** (-1.0 / theta)

    def ipsi(self, u, theta):
        return u ** (-theta) - 1.0

    def log_abs_dpsi(self, t, theta):
        # log|theta|, not log(theta): theta is negative on the whole
        # negative-dependence half of the family, where Clayton is still a
        # perfectly good bivariate copula.
        inside = 1.0 + t > 0.0
        safe = np.where(inside, t, 0.0)
        value = -np.log(abs(theta)) - (1.0 / theta + 1.0) * np.log1p(safe)
        return np.where(inside, value, -np.inf)

    def log_abs_dpsi_d(self, t, theta, d):
        # psi^(d)(t) = (-1)^d prod_{k=0}^{d-1}(1/theta + k) (1+t)^{-1/theta-d}.
        # For theta < 0 the factors change sign, so take the magnitude of each;
        # the density needs |psi^(d)| and the signs cancel against those of psi'.
        k = np.arange(d)
        log_coef = float(np.sum(np.log(np.abs(1.0 / theta + k))))
        inside = 1.0 + t > 0.0
        safe = np.where(inside, t, 0.0)
        value = log_coef - (1.0 / theta + d) * np.log1p(safe)
        return np.where(inside, value, -np.inf)

    def tau(self, theta):
        return theta / (theta + 2.0)

    def itau(self, tau, dim=2):
        if not -1.0 <= tau < 1.0:
            raise ValueError(f"Clayton requires tau in [-1, 1), got {tau}")
        if dim > 2 and tau < 0.0:
            # theta < 0 is only a valid (2-monotone) generator in two dimensions.
            raise ValueError(
                f"Clayton in dimension {dim} requires tau in [0, 1), got {tau}; "
                "negative dependence is only attainable for dim=2"
            )
        return 2.0 * tau / (1.0 - tau)

    def lambda_(self, theta):
        return TailDependence(lower=2.0 ** (-1.0 / theta) if theta > 0 else 0.0, upper=0.0)

    def rvs_frailty(self, size, theta, rng):
        return rng.gamma(1.0 / theta, 1.0, size)

    def rvs_log_frailty(self, size, theta, rng):
        r"""``log V`` for ``V ~ Gamma(1/theta, 1)``, without underflow.

        For large ``theta`` the shape ``a = 1/theta`` is tiny and the draw is
        astronomically small -- at ``theta = 1000`` NumPy returns exactly zero
        for about half of them, which sent ``t = E/V`` to infinity and half the
        sample to ``u = 0``.

        The boosting identity :math:`G\,U^{1/a} \sim \mathrm{Gamma}(a, 1)` for
        :math:`G \sim \mathrm{Gamma}(a+1, 1)` fixes it exactly: in logs it reads
        :math:`\log V = \log G + \theta \log U`, and the second term is simply a
        large negative number rather than an underflow.
        """
        a = 1.0 / theta
        boosted = rng.gamma(a + 1.0, 1.0, size)
        with np.errstate(divide="ignore"):
            return np.log(boosted) + theta * np.log(rng.uniform(size=size))

    def psi_from_log_t(self, log_t, theta):
        """``(1 + t)^{-1/theta}`` from ``log t``, via ``log1pexp``."""
        return np.asarray(np.exp(-log1pexp(log_t) / theta))

    def log_cdf(self, u, theta, dim):
        r"""``log C = -(1/theta) log(sum_j u_j^{-theta} - (d-1))``, shifted.

        Only for ``theta > 0``, where every :math:`u_j^{-\theta} \ge 1` and the
        sum can overflow -- at ``theta = 1000`` a coordinate of 0.02 alone gives
        ``inf``, and the CDF's margins came out wrong by 0.475. Factoring out
        the largest exponent leaves a bracket bounded below by :math:`e^{-m}`,
        so nothing cancels either.

        Returns ``None`` on the negative branch, where all the terms are at most
        1, nothing overflows, and the direct form additionally has to represent
        the region where ``C`` is exactly zero.
        """
        if theta <= 0.0:
            return None
        # A zero coordinate forces C = 0. Callers are supposed to have handled
        # that already, but guarding costs nothing and log(0) would poison the
        # whole row.
        zero = np.any(u <= 0.0, axis=1)
        x = -theta * np.log(np.where(u > 0.0, u, 1.0))  # >= 0
        return np.asarray(np.where(zero, -np.inf, -self._log1p_t(x, dim) / theta))

    @staticmethod
    def _log1p_t(x: NDArray[np.float64], dim: int) -> NDArray[np.float64]:
        r"""``log(sum_j u_j^{-theta} - (d-1))`` from ``x_j = -theta log u_j``.

        Factoring out the largest exponent keeps the sum in range however large
        theta is, and leaves a bracket bounded below by :math:`e^{-m}` (each
        :math:`x_j \ge 0`, so the sum is at least ``d``), so nothing cancels.
        """
        m = x.max(axis=1)
        bracket = np.sum(np.exp(x - m[:, None]), axis=1) - (dim - 1) * np.exp(-m)
        large = m + np.log(bracket)
        # Near independence (theta -> 0) every x_j is tiny and the bracket is
        # 1 + tiny - tiny: all its digits cancel, and the density multiplies the
        # result by 1/theta. At theta = 1e-17 that turned a log-density of ~0
        # into +4 per observation. Writing the sum as 1 + sum(expm1(x_j)) keeps
        # every digit; expm1 cannot overflow below the cut-off.
        small = np.log1p(np.sum(np.expm1(np.minimum(x, 1.0)), axis=1))
        return np.asarray(np.where(m <= 1.0, small, large))

    def log_pdf(self, u, theta, dim):
        r"""The Clayton density in logs.

        .. math::
            \log c = \sum_{k=0}^{d-1}\log\!\left(\tfrac1\theta + k\right)
                + d\log\theta
                - \left(\tfrac1\theta + d\right)\log(1 + t)
                + \left(\tfrac1\theta + 1\right)\sum_j x_j

        with :math:`x_j = -\theta\log u_j` and :math:`1 + t` as in
        :meth:`log_cdf`. The generic route reaches the same number by way of
        :math:`\psi^{-1}(u) = u^{-\theta} - 1`, which is ``inf`` for
        ``u = 0.05`` once ``theta`` passes about 300 -- and ``inf - inf`` is
        ``nan``. Nothing here ever forms that quantity.

        Only for ``theta > 0``: the negative branch has a support boundary that
        the direct form already represents correctly, and does not overflow.
        """
        if theta <= 0.0:
            return None
        k = np.arange(dim)
        x = -theta * np.log(u)
        return np.asarray(
            float(np.sum(np.log(1.0 / theta + k)))
            + dim * np.log(theta)
            - (1.0 / theta + dim) * self._log1p_t(x, dim)
            + (1.0 / theta + 1.0) * x.sum(axis=1)
        )


class _GumbelGenerator(ArchimedeanGenerator):
    r"""Gumbel-Hougaard: :math:`\psi(t) = \exp(-t^{1/\theta})`.

    Nelsen family (4.2.4). Upper-tail dependent, lower-tail independent, and the
    only Archimedean family that is also an extreme-value copula.
    """

    name = "Gumbel"

    def has_frailty(self, theta):
        return True

    def is_independent(self, theta):
        return theta == 1.0

    def bounds(self, dim: int) -> tuple[float, float]:
        return (1.0, np.inf)

    def psi(self, t, theta):
        return np.exp(-(t ** (1.0 / theta)))

    def ipsi(self, u, theta):
        return (-np.log(u)) ** theta

    def log_abs_dpsi(self, t, theta):
        alpha = 1.0 / theta
        # |psi'(t)| = alpha t^{alpha-1} exp(-t^alpha)
        return np.log(alpha) + (alpha - 1.0) * np.log(t) - t**alpha

    def log_abs_dpsi_d(self, t, theta, d):
        r"""|psi^(d)(t)| = exp(-t^alpha) / t^d * sum_k a_{d,k}(theta) t^{k*alpha}.

        The polynomial coefficients combine Stirling numbers of both kinds
        (Hofert et al. 2012):
        ``a_{d,k} = (-1)^{d-k} sum_{j=k}^{d} alpha^j s(d,j) S(j,k)``.
        They alternate in sign but the polynomial itself is positive, so the sum
        is formed in linear space after factoring out ``exp(-t^alpha)``.
        """
        alpha = 1.0 / theta
        coefs = _gumbel_poly_coefs(d, alpha)  # length d, for k = 1..d
        ta = t**alpha
        k = np.arange(1, d + 1)
        poly = np.sum(coefs * ta[..., None] ** k, axis=-1)
        return -ta - d * np.log(t) + np.log(poly)

    def tau(self, theta):
        return 1.0 - 1.0 / theta

    def itau(self, tau, dim=2):
        if not 0.0 <= tau < 1.0:
            raise ValueError(f"Gumbel requires tau in [0, 1), got {tau}")
        return 1.0 / (1.0 - tau)

    def lambda_(self, theta):
        return TailDependence(lower=0.0, upper=2.0 - 2.0 ** (1.0 / theta))

    def rvs_frailty(self, size, theta, rng):
        return rstable_positive(size, 1.0 / theta, rng)


class _FrankGenerator(ArchimedeanGenerator):
    r"""Frank: :math:`\psi(t) = -\log\bigl(1 - (1 - e^{-\theta})e^{-t}\bigr)/\theta`.

    Nelsen family (4.2.5). The only Archimedean family that is *radially
    symmetric*, and the only common one admitting the full dependence range
    :math:`\tau \in (-1, 1)` — but with no tail dependence at either end.
    """

    name = "Frank"

    def bounds(self, dim: int) -> tuple[float, float]:
        return (-np.inf, np.inf) if dim == 2 else (0.0, np.inf)

    def psi(self, t, theta):
        r"""``-log(1 - h e^{-t}) / theta``, evaluated through ``logaddexp``.

        Writing :math:`1 - h e^{-t} = (1 - e^{-t}) + e^{-\theta - t}` makes it a
        sum of two non-negative terms, so its logarithm is a ``logaddexp`` and
        neither cancels nor overflows.

        The naive form dies for ``theta`` beyond about 35: there ``h`` rounds to
        exactly 1, and for small ``t`` the product ``h e^{-t}`` rounds to exactly
        1 as well, so ``1 - h e^{-t}`` evaluates to **zero** and ``psi`` returns
        infinity for perfectly ordinary arguments. At ``theta = 50``, which is
        only ``tau = 0.92``, the generator stopped inverting itself for every
        ``u`` above about 0.6, breaking the margins of the CDF by 0.22.
        Forming the sum directly instead would then overflow for
        ``theta < -709``; taking the logarithm first avoids both.
        """
        t = np.asarray(t, dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_omz = np.logaddexp(np.log(-np.expm1(-t)), -theta - t)
        return -log_omz / theta

    def ipsi(self, u, theta):
        r"""``log( expm1(-theta) / expm1(-theta u) )``, computed stably.

        Both ``expm1`` terms carry the sign of ``-theta``, so their ratio is
        positive whichever way ``theta`` points. Writing that ratio as
        :math:`1 + s` with

        .. math::
            s = \frac{e^{-\theta u}\,\mathrm{expm1}(-\theta(1-u))}
                     {\mathrm{expm1}(-\theta u)}

        and evaluating :math:`\log(1 + s)` through :func:`~rcopula.special.logexp.log1pexp`
        of :math:`\log s` keeps every regime accurate:

        * as ``u -> 1`` the answer is small and ``log1p`` protects it -- the
          naive quotient loses about eight digits at ``theta = 20``;
        * for ``theta < 0`` the same quotient saturates to exactly ``-1``, so the
          previous ``-log1p(r)`` form returned ``+inf`` for **every** negative
          ``theta`` below about -40, taking the CDF's margins with it;
        * ``log1pexp`` returns its argument when large, so nothing overflows
          however extreme ``theta`` gets.

        ``theta = 0`` is the independence limit, where the ratio is 0/0; the
        copula short-circuits there before reaching this method.
        """
        u = np.asarray(u, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_s = (
                -theta * u
                + np.log(np.abs(np.expm1(-theta * (1.0 - u))))
                - np.log(np.abs(np.expm1(-theta * u)))
            )
        return np.asarray(log1pexp(log_s))

    @staticmethod
    def _z_and_1mz(
        t: NDArray[np.float64], theta: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""Return ``z = h e^{-t}`` and ``1 - z``, both computed stably.

        Forming ``1 - z`` by subtraction loses precision whenever ``z`` is close
        to 1, which happens for large ``theta`` and small ``t`` — precisely the
        upper corner of the unit square. Since

        .. math::
            1 - h e^{-t} = 1 - (1 - e^{-\theta}) e^{-t}
                         = -\mathrm{expm1}(-t) + e^{-\theta - t},

        the result is a sum of two non-negative terms and never cancels. At
        ``theta = 20`` this recovers roughly eight digits in the density.
        """
        h = -np.expm1(-theta)
        z = h * np.exp(-t)
        one_minus_z = -np.expm1(-t) + np.exp(-theta - t)
        return z, one_minus_z

    def log_abs_dpsi(self, t, theta):
        # For theta < 0, h = 1 - e^{-theta} is negative and so is z. The
        # derivative's magnitude is |z| / (|1 - z| |theta|); logging z directly
        # would give nan across the entire negative-dependence half of the
        # family, which is the half Frank exists for.
        z, omz = self._z_and_1mz(t, theta)
        return np.log(np.abs(z)) - np.log(omz) - np.log(abs(theta))

    def log_abs_dpsi_d(self, t, theta, d):
        r"""|psi^(d)(t)| = |Li_{-(d-1)}(h e^{-t})| / |theta|.

        Expanding ``psi`` as a geometric-type series gives
        ``psi^(d)(t) = ((-1)^d / theta) sum_k k^{d-1} (h e^{-t})^k``, i.e. a
        polylogarithm of negative integer order, which has the closed form
        used by :func:`_polylog_neg_int` (Eulerian numbers).

        For ``theta < 0`` the argument ``z = h e^{-t}`` is negative and can be
        large in magnitude, so the polylogarithm changes sign with the order.
        The rational closed form is the analytic continuation and stays valid
        there; only its *sign* varies, and the magnitude is what a log density
        needs.
        """
        z, omz = self._z_and_1mz(t, theta)
        return _log_polylog_neg_int(z, d - 1, omz) - np.log(abs(theta))

    def tau(self, theta):
        if theta == 0.0:
            return 0.0
        return 1.0 - 4.0 * (1.0 - float(debye1(theta))) / theta

    def rho(self, theta):
        if theta == 0.0:
            return 0.0
        return 1.0 - 12.0 * (float(debye1(theta)) - float(debye2(theta))) / theta

    def lambda_(self, theta):
        return TailDependence(lower=0.0, upper=0.0)

    #: Largest ``|theta|`` searched by :meth:`itau` / :meth:`irho`. Frank's tau
    #: approaches 1 like ``1 - 4/theta`` (rho like ``1 - 6/theta``), so this
    #: reaches ``|tau|`` within about ``4e-12`` of 1 -- far beyond anything a
    #: sample can estimate -- while every quantity involved stays finite.
    _INVERSION_THETA_MAX = 1e12

    def _invert(
        self, measure: Callable[[float], float], target: float, label: str, dim: int
    ) -> float:
        """Root-find ``measure(theta) = target`` over the family's full range."""
        if not -1.0 < target < 1.0:
            raise ValueError(f"Frank requires {label} in (-1, 1), got {target}")
        if dim > 2 and target < 0.0:
            raise ValueError(
                f"Frank in dimension {dim} requires {label} in [0, 1), got {target}; "
                "negative dependence is only attainable for dim=2"
            )
        if target == 0.0:
            return 0.0
        edge = self._INVERSION_THETA_MAX
        lo, hi = (1e-12, edge) if target > 0 else (-edge, -1e-12)
        extreme = measure(hi if target > 0 else lo)
        if abs(target) >= abs(extreme):
            raise ValueError(
                f"{label} = {target} is too close to {'+' if target > 0 else '-'}1 to be "
                f"attained by the Frank family (the most extreme {label} it reaches in "
                f"floating point, at |theta| = {edge:g}, is {extreme!r})"
            )
        return float(brentq(lambda th: measure(th) - target, lo, hi, xtol=1e-14, rtol=8.9e-16))

    def itau(self, tau, dim=2):
        """Invert Kendall's tau for Frank, searching ``|theta|`` up to ``1e12``.

        Raises a ``ValueError`` naming the limit when ``tau`` is so close to
        ``+-1`` that no representable ``theta`` reaches it, and when a negative
        ``tau`` is asked for with ``dim > 2``.
        """
        return self._invert(self.tau, tau, "tau", dim)

    def irho(self, rho: float, dim: int = 2) -> float:
        """Invert Spearman's rho for Frank, using its closed-form Debye expression.

        Same search range and errors as :meth:`itau`.
        """
        return self._invert(self.rho, rho, "rho", dim)

    def rvs_frailty(self, size, theta, rng):
        # log(1 - p) is exactly -theta; pass it, because 1 - e^{-theta}
        # saturates to 1 past theta ~ 37 and log1p(-p) would be -inf.
        return rlog_series(size, -np.expm1(-theta), rng, log1mp=-theta)


class _JoeGenerator(ArchimedeanGenerator):
    r"""Joe: :math:`\psi(t) = 1 - (1 - e^{-t})^{1/\theta}`.

    Nelsen family (4.2.6). Upper-tail dependent like Gumbel but with a heavier
    upper tail and no lower-tail dependence.
    """

    name = "Joe"

    def has_frailty(self, theta):
        return True

    def is_independent(self, theta):
        return theta == 1.0

    def bounds(self, dim: int) -> tuple[float, float]:
        return (1.0, np.inf)

    def psi(self, t, theta):
        alpha = 1.0 / theta
        return -np.expm1(log1mexp(t) * alpha)

    def ipsi(self, u, theta):
        r"""``-log(1 - (1-u)^theta)``.

        Both naive forms fail somewhere: ``log1p(-(1-u)**theta)`` cancels as
        ``u -> 0`` (the power tends to 1), while ``log(-expm1(...))`` cancels as
        ``u -> 1`` (the power tends to 0). Writing ``(1-u)^theta = e^{-a}`` with
        ``a = -theta*log1p(-u) > 0`` turns the whole thing into
        ``-log1mexp(a)``, and :func:`log1mexp` already switches branches at the
        right point.
        """
        return -log1mexp(-theta * np.log1p(-u))

    def log_abs_dpsi(self, t, theta):
        alpha = 1.0 / theta
        # |psi'(t)| = alpha (1 - e^{-t})^{alpha - 1} e^{-t}.
        # log(1 - e^{-t}) MUST go through log1mexp: for the small t produced by
        # u close to 1 with large theta (t ~ 1e-15 is routine), evaluating it as
        # log1p(-exp(-t)) leaves only two correct digits.
        return np.log(alpha) + (alpha - 1.0) * log1mexp(t) - t

    def log_abs_dpsi_d(self, t, theta, d):
        r"""|psi^(d)(t)| via the exact coefficient recursion.

        Writing :math:`y = 1 - e^{-t}` turns differentiation into the operator
        :math:`(1-y)\,\mathrm{d}/\mathrm{d}y`, under which the ansatz

        .. math::
            \psi^{(n)}(t) = \sum_k c_{n,k}\, y^{\alpha-k} (1-y)^{k}

        is closed, with

        .. math::
            c_{n+1,k+1} \mathrel{+}= (\alpha - k)\, c_{n,k}, \qquad
            c_{n+1,k}   \mathrel{-}= k\, c_{n,k},

        starting from :math:`c_{0,0} = -1`. Unlike the Sibuya-series form this
        is a *finite* sum with no truncation error. The coefficients alternate
        in sign, so the sum is accumulated with a sign-aware log-sum-exp.
        """
        alpha = 1.0 / theta
        c = _joe_poly_coefs(d, alpha)  # c[k] for k = 0..d
        k = np.arange(d + 1)

        log_y = log1mexp(t)  # log(1 - e^{-t}), accurate for tiny t
        # log |c_k| + (alpha - k) log y + k * (-t)
        with np.errstate(divide="ignore"):
            log_terms = (
                np.log(np.abs(c)) + np.multiply.outer(log_y, alpha - k) - np.multiply.outer(t, k)
            )
        log_abs, _ = signed_logsumexp(log_terms, np.sign(c), axis=-1)
        return log_abs

    def tau(self, theta):
        r"""Closed form via digamma.

        The defining series
        :math:`\tau = 1 - 4\sum_{k\ge1} [k(\theta k+2)(\theta(k-1)+2)]^{-1}`
        converges only like :math:`k^{-3}`, so reaching float64 accuracy by
        direct summation would need ~1e7 terms. Partial fractions in ``k`` with
        ``a = 2/theta`` collapse it to digamma values instead.
        """
        if theta == 1.0:
            return 0.0
        a = 2.0 / theta
        # Sum = -B (psi(1+a) + gamma) - C (psi(a) + gamma), B = 1/a, C = 1/(1-a).
        b_term = (digamma(1.0 + a) + _EULER_GAMMA) / a
        x = a - 1.0
        if abs(x) < 0.1:
            # C = 1/(1-a) blows up at a = 1 (theta = 2) but the product with
            # (psi(a) + gamma), which vanishes there, stays finite. Substituting
            # psi(1+x) + gamma = sum_{n>=2} (-1)^n zeta(n) x^{n-1} gives
            # (psi(a)+gamma)/(1-a) = -zeta(2) + zeta(3) x - zeta(4) x^2 + ...
            n = np.arange(2, 22)
            c_term = float(np.sum((-1.0) ** (n - 1) * zeta(n.astype(float)) * x ** (n - 2)))
        else:
            c_term = (digamma(a) + _EULER_GAMMA) / (1.0 - a)
        return 1.0 - 4.0 / theta**2 * (-b_term - c_term)

    def lambda_(self, theta):
        return TailDependence(lower=0.0, upper=2.0 - 2.0 ** (1.0 / theta))

    def rvs_frailty(self, size, theta, rng):
        return rsibuya(size, 1.0 / theta, rng)


class _AMHGenerator(ArchimedeanGenerator):
    r"""Ali-Mikhail-Haq: :math:`\psi(t) = (1-\theta)/(e^{t} - \theta)`.

    Nelsen family (4.2.3). No tail dependence, and only weak dependence is
    reachable at all: :math:`\tau \in [(5 - 8\log 2)/3,\ 1/3]`, roughly
    ``[-0.1817, 0.3333]``.

    Note a deliberate divergence from R: R's ``amhCopula`` is restricted to
    ``d = 2``, even though its ``copAMH`` generator object is d-dimensional.
    Here ``dim > 2`` is supported (with ``theta >= 0``, as negative theta is
    only d-monotone in two dimensions).
    """

    name = "AMH"

    def bounds(self, dim: int) -> tuple[float, float]:
        return (-1.0, 1.0) if dim == 2 else (0.0, 1.0)

    def psi(self, t, theta):
        return (1.0 - theta) / (np.exp(t) - theta)

    def ipsi(self, u, theta):
        return np.log((1.0 - theta * (1.0 - u)) / u)

    def log_abs_dpsi(self, t, theta):
        return self.log_abs_dpsi_d(t, theta, 1)

    def log_abs_dpsi_d(self, t, theta, d):
        r"""|psi^(d)(t)| = ((1-theta)/theta) * Li_{-d}(theta e^{-t}).

        Expanding ``psi(t) = (1-theta) sum_{k>=1} theta^{k-1} e^{-kt}`` and
        differentiating termwise gives a polylogarithm of negative integer
        order, which :func:`_polylog_neg_int` evaluates in closed form.

        The ``1/theta`` is a *removable* singularity -- at ``theta = 0`` AMH is
        the independence copula and the derivative is simply ``e^{-t}`` -- so
        the division is done symbolically instead, via
        :func:`_polylog_neg_int_over_z`, which absorbs the ``theta`` that
        ``Li_{-d}(theta e^{-t})`` contributes. Dividing numerically would raise
        at ``theta = 0``, and ``theta = 0`` sits squarely inside the admissible
        interval, so an optimiser walks straight into it.
        """
        z = theta * np.exp(-t)
        # (1 - theta) e^{-t} is positive for every admissible theta < 1, so its
        # log is taken directly. Only Li_{-d}(z)/z can change sign -- for
        # theta < 0, where z < 0 and the Eulerian polynomial may be negative --
        # and _log_polylog_neg_int_over_z already returns the log of its
        # absolute value, so no nan arises from logging a negative polylog.
        return np.log1p(-theta) - t + _log_polylog_neg_int_over_z(z, d)

    def tau(self, theta):
        r""":math:`\tau = 1 - 2[(1-\theta)^2 \log(1-\theta) + \theta] / (3\theta^2)`."""
        if theta == 0.0:
            return 0.0
        if abs(theta) < 1e-4:
            # The closed form is 0/0 at theta = 0; use the Taylor series, which
            # R also does (its `tauAMH` expands to order 7).
            t = theta
            return (
                2.0 * t / 9.0 + t**2 / 18.0 + t**3 / 30.0 + 2.0 * t**4 / 105.0 + 5.0 * t**5 / 378.0
            )
        return 1.0 - 2.0 * ((1.0 - theta) ** 2 * np.log1p(-theta) + theta) / (3.0 * theta**2)

    def lambda_(self, theta):
        return TailDependence(lower=0.0, upper=0.0)

    def rvs_frailty(self, size, theta, rng):
        if theta == 0.0:
            return np.ones(size)
        # Geometric on {1, 2, ...} with success probability 1 - theta.
        return rng.geometric(1.0 - theta, size).astype(np.float64)


def _joe_poly_coefs(d: int, alpha: float) -> NDArray[np.float64]:
    """Coefficients ``c_{d,k}``, ``k = 0..d``, of the Joe generator derivative."""
    c = np.zeros(d + 1)
    c[0] = -1.0
    for _ in range(d):
        nxt = np.zeros(d + 1)
        for k in range(d):
            if c[k] == 0.0:
                continue
            nxt[k + 1] += (alpha - k) * c[k]
            nxt[k] -= k * c[k]
        c = nxt
    return c


# ======================================================================
# Coefficient helpers
# ======================================================================


def _gumbel_poly_coefs(d: int, alpha: float) -> NDArray[np.float64]:
    """``a_{d,k}(theta)`` for ``k = 1..d`` in the Gumbel d-th derivative."""
    s1 = stirling1_all(d)  # s(d, j), j = 1..d
    out = np.empty(d, dtype=np.float64)
    for k in range(1, d + 1):
        j = np.arange(k, d + 1)
        s2 = np.array([stirling2_all(int(jj))[k - 1] for jj in j])
        out[k - 1] = (-1.0) ** (d - k) * np.sum(alpha**j * s1[k - 1 : d] * s2)
    return out


def _polylog_neg_int(
    z: NDArray[np.float64], n: int, one_minus_z: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    r"""``Li_{-n}(z)`` for integer ``n >= 0`` and real ``z < 1``.

    Uses the Eulerian-number closed form

    .. math::

        \mathrm{Li}_{-n}(z) = \frac{1}{(1-z)^{n+1}}
            \sum_{k=0}^{n-1} A(n, k)\, z^{n-k}, \qquad n \ge 1,

    with :math:`\mathrm{Li}_0(z) = z/(1-z)`. This turns what would be an
    infinite sum into a degree-``n`` polynomial.

    The rational closed form is the analytic continuation of the series
    :math:`\sum_{k \ge 1} k^n z^k`, so it is valid for every real ``z < 1``,
    not only inside the unit disc: the Frank generator with ``theta < 0``
    evaluates it at negative ``z`` of arbitrarily large magnitude, and AMH with
    ``theta < 0`` at ``z`` in ``(-1, 0)``. For ``z < 0`` the value can be
    negative, depending on ``n``.

    Parameters
    ----------
    z : ndarray
        Argument, any real value below 1.
    n : int
        Non-negative order.
    one_minus_z : ndarray, optional
        A separately-computed ``1 - z``. The denominator is raised to the power
        ``n + 1``, so any cancellation in ``1 - z`` is *amplified* ``n + 1``
        times; callers that can form it accurately should pass it in.
    """
    omz = (1.0 - z) if one_minus_z is None else one_minus_z
    if n == 0:
        return z / omz
    a = eulerian_all(n)  # A(n, k), k = 0..n-1
    k = np.arange(n)
    num = np.sum(a * z[..., None] ** (n - k), axis=-1)
    return num / omz ** (n + 1)


def _log_polylog_neg_int(
    z: NDArray[np.float64], n: int, one_minus_z: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    r"""``log |Li_{-n}(z)|``, without forming ``Li_{-n}(z)`` first.

    The Eulerian closed form divides a bounded numerator (its coefficients sum
    to ``n!``) by ``(1 - z)^{n+1}``. When ``z`` is within rounding of 1 that
    denominator is subnormal and the quotient overflows: the Frank copula at
    ``theta = 700`` produced ``inf`` for a quarter of the unit square, and its
    density with it. Taking the logarithm of each part separately keeps
    everything in range -- the numerator's log is ordinary, the denominator's is
    just ``(n+1) log(1-z)``.
    """
    omz = (1.0 - z) if one_minus_z is None else one_minus_z
    # A subnormal numerator or denominator means the density is zero there,
    # which is a legitimate answer rather than a warning.
    with np.errstate(divide="ignore"):
        if n == 0:
            return np.log(np.abs(z)) - np.log(omz)
        a = eulerian_all(n)  # A(n, k), k = 0..n-1
        k = np.arange(n)
        num = np.sum(a * z[..., None] ** (n - k), axis=-1)
        return np.log(np.abs(num)) - (n + 1) * np.log(omz)


def _log_polylog_neg_int_over_z(
    z: NDArray[np.float64], n: int, one_minus_z: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """``log |Li_{-n}(z) / z|``; see :func:`_polylog_neg_int_over_z`."""
    omz = (1.0 - z) if one_minus_z is None else one_minus_z
    with np.errstate(divide="ignore"):
        if n == 0:
            return -np.log(omz)
        a = eulerian_all(n)
        k = np.arange(n)
        num = np.sum(a * z[..., None] ** (n - k - 1), axis=-1)
        return np.log(np.abs(num)) - (n + 1) * np.log(omz)


def _polylog_neg_int_over_z(
    z: NDArray[np.float64], n: int, one_minus_z: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    r"""``Li_{-n}(z) / z``, which is finite at ``z = 0``.

    Every term of :math:`\mathrm{Li}_{-n}(z) = \sum_{k\ge1} k^n z^k` carries at
    least one factor of ``z``, so the ratio is a polynomial and tends to 1 as
    ``z \to 0``. Cancelling the factor symbolically -- dropping the exponent by
    one in the Eulerian closed form -- lets callers divide by a parameter that
    is proportional to ``z`` without ever forming ``0/0``.
    """
    omz = (1.0 - z) if one_minus_z is None else one_minus_z
    if n == 0:
        return 1.0 / omz
    a = eulerian_all(n)  # A(n, k), k = 0..n-1
    k = np.arange(n)
    num = np.sum(a * z[..., None] ** (n - k - 1), axis=-1)
    return num / omz ** (n + 1)


# ======================================================================
# The copula
# ======================================================================


class ArchimedeanCopula(Copula):
    """A copula with one dependence parameter, built from an Archimedean generator.

    Archimedean copulas describe dependence through a single decreasing
    function (the generator, see :class:`ArchimedeanGenerator`) and a single
    number ``theta``. They are exchangeable: swapping variables does not change
    the copula. Most users want one of the named families
    (:class:`ClaytonCopula`, :class:`GumbelCopula`, :class:`FrankCopula`,
    :class:`JoeCopula`, :class:`AMHCopula`); use this class directly only to
    plug in a generator of your own.

    Parameters
    ----------
    generator : ArchimedeanGenerator
        The generator defining the family.
    theta : float, default nan
        Dependence parameter, inside ``generator.bounds(dim)``. ``nan`` means
        "not yet known": the copula can then be fitted but not evaluated.
    dim : int, default 2
        Number of variables, at least 2.
    free : array_like of bool, shape (1,), or None, default None
        Whether ``theta`` is estimated (``True``) or held fixed (``False``)
        by :func:`~rcopula.fit.fit`. ``None`` means free.

    Attributes
    ----------
    generator : ArchimedeanGenerator
        The generator passed in.
    name : str
        Family name, taken from the generator.
    theta : float
        The dependence parameter.

    Raises
    ------
    ValueError
        If ``dim < 2`` or ``theta`` is outside the admissible range.
    """

    def __init__(
        self,
        generator: ArchimedeanGenerator,
        theta: float = np.nan,
        dim: int = 2,
        *,
        free: ArrayLike | None = None,
    ) -> None:
        self.generator = generator
        self.name = generator.name
        self.param_names = (generator.param_name,)
        super().__init__([theta], dim, free=free)

    @property
    def theta(self) -> float:
        """The dependence parameter ``theta`` as a plain float (``nan`` if not yet set).

        Returns
        -------
        float
            ``params[0]``.
        """
        return float(self._params[0])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """The allowed range of ``theta`` for this copula's dimension.

        Returns
        -------
        list of tuple of (float, float)
            A one-element list ``[(lower, upper)]`` from the generator's
            :meth:`~ArchimedeanGenerator.bounds`.
        """
        return [self.generator.bounds(self._dim)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> ArchimedeanCopula:
        return ArchimedeanCopula(
            self.generator, float(np.atleast_1d(params)[0]), self._dim, free=free
        )

    # -- numerical core ------------------------------------------------

    def _logpdf(self, u, params):
        theta = float(params[0])
        d = self._dim
        g = self.generator
        # The independence point is a removable singularity for most generators
        # -- Clayton and Frank both divide by theta -- and it sits inside the
        # admissible interval, so an optimiser reaches it.
        if g.is_independent(theta):
            return np.zeros(u.shape[0])
        log_p = g.log_pdf(u, theta, d)
        if log_p is not None:
            return log_p
        t_j = g.ipsi(u, theta)
        t = t_j.sum(axis=1)
        return g.log_abs_dpsi_d(t, theta, d) - g.log_abs_dpsi(t_j, theta).sum(axis=1)

    def _cdf(self, u, params):
        theta = float(params[0])
        g = self.generator
        if g.is_independent(theta):
            return np.asarray(u.prod(axis=1))
        log_c = g.log_cdf(u, theta, self._dim)
        if log_c is not None:
            return np.asarray(np.exp(log_c))
        return g.psi(g.ipsi(u, theta).sum(axis=1), theta)

    def _rvs(self, size, params, rng):
        theta = float(params[0])
        g = self.generator
        if g.is_independent(theta):
            return rng.uniform(size=(size, self._dim))
        if not g.has_frailty(theta):
            return self._rvs_conditional(size, theta, rng)
        # Marshall-Olkin (1988): U_j = psi(E_j / V), with V the frailty. Done in
        # logs, because at strong dependence V underflows and E/V overflows
        # while log(E) - log(V) stays perfectly ordinary.
        log_v = g.rvs_log_frailty(size, theta, rng)[:, None]
        with np.errstate(divide="ignore"):
            log_e = np.log(rng.exponential(1.0, size=(size, self._dim)))
        out = g.psi_from_log_t(log_e - log_v, theta)
        # Backstop: near-comonotone parameters put draws within one ulp of the
        # boundary, and a copula observation must lie strictly inside the cube
        # or every downstream log density is -inf.
        return np.clip(out, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))

    def _rvs_conditional(
        self, size: int, theta: float, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        r"""Sample by conditional inversion, for parameters with no frailty.

        Clayton on :math:`[-1, 0)`, Frank on :math:`(-\infty, 0)` and AMH on
        :math:`[-1, 0)` are perfectly good bivariate copulas but their
        generators are not Laplace transforms, so the Marshall-Olkin
        construction does not apply. Draw :math:`U_1` and an independent
        :math:`W`, then solve

        .. math::
            h(u_2 \mid u_1) = \frac{\psi'(\psi^{-1}(u_2) + \psi^{-1}(u_1))}
                                   {\psi'(\psi^{-1}(u_1))} = W

        for :math:`u_2`. The h-function is increasing in :math:`u_2`, so 60
        bisections take the bracket to machine precision.
        """
        if self._dim != 2:
            raise ValueError(
                f"{self.name} copula with theta={theta:g} has no frailty representation, "
                "and conditional sampling is implemented for dim=2 only"
            )
        g = self.generator
        u1 = rng.uniform(size=size)
        w = rng.uniform(size=size)
        t1 = g.ipsi(u1, theta)
        log_denominator = g.log_abs_dpsi(t1, theta)

        lo = np.full(size, 1e-12)
        hi = np.full(size, 1.0 - 1e-12)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                h = np.exp(g.log_abs_dpsi(g.ipsi(mid, theta) + t1, theta) - log_denominator)
                below = np.nan_to_num(h, nan=0.0) < w
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)
        return np.column_stack([u1, 0.5 * (lo + hi)])

    # -- dependence measures -------------------------------------------

    def tau(self) -> float:
        """Return Kendall's tau, a rank correlation in ``[-1, 1]`` implied by ``theta``.

        Kendall's tau is the probability that two random draws are ordered the
        same way in both variables, minus the probability they are ordered
        oppositely. It depends only on the copula, not on the margins.

        Returns
        -------
        float
            Population Kendall's tau of the bivariate margins.

        Raises
        ------
        ValueError
            If ``theta`` is ``nan`` (not yet set).
        """
        self._require_specified()
        return float(self.generator.tau(self.theta))

    def rho(self) -> float:
        """Return Spearman's rho, the correlation of the ranks implied by ``theta``.

        Returns
        -------
        float
            Population Spearman's rho of the bivariate margins, in ``[-1, 1]``.

        Raises
        ------
        ValueError
            If ``theta`` is ``nan`` (not yet set).
        """
        self._require_specified()
        return float(self.generator.rho(self.theta))

    def lambda_(self) -> TailDependence:
        """Return how likely extreme values are to occur together (tail dependence).

        ``lower`` is the chance that one variable is extremely small given the
        other is; ``upper`` is the same for extremely large values, both in the
        limit.

        Returns
        -------
        TailDependence
            Named tuple ``(lower, upper)`` of floats in ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``theta`` is ``nan`` (not yet set).
        """
        self._require_specified()
        return self.generator.lambda_(self.theta)

    # -- generator passthroughs (R's psi / iPsi) -----------------------

    def psi(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the generator function :math:`\\psi(t)` at this copula's ``theta``.

        The Python spelling of R's ``psi``.

        Parameters
        ----------
        t : array_like of float, any shape
            Non-negative arguments.

        Returns
        -------
        numpy.ndarray of float, same shape as ``t``
            Generator values in ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``theta`` is ``nan`` (not yet set), as for :meth:`tau`.

        Examples
        --------
        >>> from rcopula import ClaytonCopula
        >>> c = ClaytonCopula(theta=1.0)
        >>> c.psi([0.0, 1.0]).tolist()
        [1.0, 0.5]
        """
        self._require_specified()
        return self.generator.psi(np.asarray(t, dtype=np.float64), self.theta)

    def ipsi(self, u: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the inverse generator :math:`\\psi^{-1}(u)` at this copula's ``theta``.

        The Python spelling of R's ``iPsi``.

        Parameters
        ----------
        u : array_like of float, any shape
            Values in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray of float, same shape as ``u``
            Non-negative values.

        Raises
        ------
        ValueError
            If ``theta`` is ``nan`` (not yet set), as for :meth:`tau`.

        Examples
        --------
        >>> from rcopula import ClaytonCopula
        >>> c = ClaytonCopula(theta=1.0)
        >>> c.ipsi([1.0, 0.5]).tolist()
        [0.0, 1.0]
        """
        self._require_specified()
        return self.generator.ipsi(np.asarray(u, dtype=np.float64), self.theta)


class _ConcreteArchimedean(ArchimedeanCopula):
    """Base for the concrete one-parameter families.

    Each subclass binds a single generator instance, so users write
    ``ClaytonCopula(2.0)`` rather than ``ArchimedeanCopula(clayton_gen, 2.0)``.
    These are written out as real classes rather than produced by a factory:
    a factory would defeat static type checking, IDE completion and pickling
    for the sake of saving a dozen lines.
    """

    generator_instance: ArchimedeanGenerator = None  # type: ignore[assignment]

    def __init__(
        self,
        theta: float = np.nan,
        dim: int = 2,
        *,
        free: ArrayLike | None = None,
    ) -> None:
        super().__init__(self.generator_instance, theta, dim, free=free)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Promote the generator's identity to class attributes so that a copy
        # made without running __init__ still validates correctly.
        super().__init_subclass__(**kwargs)
        if getattr(cls, "generator_instance", None) is not None:
            cls.name = cls.generator_instance.name
            cls.param_names = (cls.generator_instance.param_name,)

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> _ConcreteArchimedean:
        return type(self)(float(np.atleast_1d(params)[0]), self._dim, free=free)

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> _ConcreteArchimedean:
        """Build a copula of this family whose Kendall's tau equals a target value (R's ``iTau``).

        Handy when you know roughly how strongly variables move together (as a
        rank correlation) and want the matching ``theta``.

        Parameters
        ----------
        tau : float
            Target Kendall's tau, a rank correlation in ``[-1, 1]``. Must be
            reachable by the family in dimension ``dim``.
        dim : int, default 2
            Number of variables, at least 2.
        **kwargs : Any
            Passed to the constructor (e.g. ``free``).

        Returns
        -------
        ClaytonCopula, GumbelCopula, FrankCopula, JoeCopula or AMHCopula
            A new instance of the calling class.

        Raises
        ------
        ValueError
            If ``tau`` cannot be reached by this family in dimension ``dim``.
        """
        return cls(cls.generator_instance.itau(tau, dim), dim, **kwargs)

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> _ConcreteArchimedean:
        """Build a copula of this family whose Spearman's rho equals a target value (R's ``iRho``).

        Spearman's rho is the correlation of the ranks. Slower than
        :meth:`from_tau` for families without a closed form, since it inverts
        a numerical integral.

        Parameters
        ----------
        rho : float
            Target Spearman's rho, in ``[-1, 1]``. Must be reachable by the
            family.
        dim : int, default 2
            Number of variables, at least 2.
        **kwargs : Any
            Passed to the constructor (e.g. ``free``).

        Returns
        -------
        ClaytonCopula, GumbelCopula, FrankCopula, JoeCopula or AMHCopula
            A new instance of the calling class.

        Raises
        ------
        ValueError
            If ``rho`` cannot be reached by this family.
        """
        return cls(cls.generator_instance.irho(rho, dim), dim, **kwargs)


class ClaytonCopula(_ConcreteArchimedean):
    r"""Clayton copula: dependence that is strongest when values are jointly small.

    Use it when joint *lows* matter more than joint highs -- for example
    losses that crash together but rise independently. Larger ``theta`` means
    stronger dependence; ``theta = 0`` is independence.

    Parameters
    ----------
    theta : float, default nan
        Dependence parameter. Must lie in ``[-1, inf)`` for ``dim=2`` and in
        ``[0, inf)`` for ``dim > 2``. ``nan`` means "to be estimated".
    dim : int, default 2
        Number of variables, at least 2.
    free : array_like of bool, shape (1,), or None, default None
        Whether ``theta`` is estimated by :func:`~rcopula.fit.fit`
        (``None`` means free).

    Attributes
    ----------
    theta : float
        The dependence parameter.
    dim : int
        Number of variables.

    Raises
    ------
    ValueError
        If ``dim < 2`` or ``theta`` is outside the admissible range.

    Notes
    -----
    Generator :math:`\psi(t) = (1+t)^{-1/\theta}`, with
    :math:`\tau = \theta/(\theta+2)` and lower tail dependence
    :math:`\lambda_L = 2^{-1/\theta}`.

    ``theta`` ranges over :math:`[-1, \infty)` in ``dim=2`` and
    :math:`(0, \infty)` beyond, because the generator ceases to be
    ``d``-monotone for negative ``theta`` in higher dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import ClaytonCopula
    >>> c = ClaytonCopula(theta=2.0, dim=3)
    >>> float(c.tau())
    0.5
    >>> bool(c.lambda_().lower == 2 ** -0.5)
    True
    >>> u = c.rvs(1000, random_state=0)
    >>> u.shape
    (1000, 3)
    >>> bool(np.all((u > 0) & (u < 1)))
    True

    Calibrating to a target dependence, R's ``iTau``:

    >>> ClaytonCopula.from_tau(0.5).theta
    2.0
    """

    generator_instance = _ClaytonGenerator()


class GumbelCopula(_ConcreteArchimedean):
    r"""Gumbel-Hougaard copula: dependence that is strongest when values are jointly large.

    Use it when joint *highs* (e.g. simultaneous large claims or floods)
    matter most. It allows only positive dependence; ``theta = 1`` is
    independence and larger ``theta`` is stronger. It is also an extreme-value
    copula.

    Parameters
    ----------
    theta : float, default nan
        Dependence parameter, ``theta >= 1``. ``nan`` means "to be estimated".
    dim : int, default 2
        Number of variables, at least 2.
    free : array_like of bool, shape (1,), or None, default None
        Whether ``theta`` is estimated by :func:`~rcopula.fit.fit`
        (``None`` means free).

    Attributes
    ----------
    theta : float
        The dependence parameter.
    dim : int
        Number of variables.

    Raises
    ------
    ValueError
        If ``dim < 2`` or ``theta < 1``.

    Notes
    -----
    Generator :math:`\psi(t) = \exp(-t^{1/\theta})`, with
    :math:`\tau = 1 - 1/\theta` and upper tail dependence
    :math:`\lambda_U = 2 - 2^{1/\theta}`. Requires ``theta >= 1``.

    Examples
    --------
    >>> from rcopula import GumbelCopula
    >>> GumbelCopula.from_tau(0.5).theta
    2.0
    >>> g = GumbelCopula(theta=2.0)
    >>> float(g.tau())
    0.5
    >>> float(round(g.lambda_().upper, 10))
    0.5857864376
    """

    generator_instance = _GumbelGenerator()


class FrankCopula(_ConcreteArchimedean):
    r"""Frank copula: symmetric dependence with no extra clustering of extremes.

    Use it when dependence looks the same for lows and highs and extremes are
    not unusually likely to coincide. In two dimensions it covers both
    negative (``theta < 0``) and positive (``theta > 0``) dependence;
    ``theta = 0`` is independence.

    Parameters
    ----------
    theta : float, default nan
        Dependence parameter. Any real number for ``dim=2``; ``theta >= 0``
        for ``dim > 2``. ``nan`` means "to be estimated".
    dim : int, default 2
        Number of variables, at least 2.
    free : array_like of bool, shape (1,), or None, default None
        Whether ``theta`` is estimated by :func:`~rcopula.fit.fit`
        (``None`` means free).

    Attributes
    ----------
    theta : float
        The dependence parameter.
    dim : int
        Number of variables.

    Raises
    ------
    ValueError
        If ``dim < 2`` or ``theta`` is outside the admissible range.

    Notes
    -----
    Generator :math:`\psi(t) = -\log(1 - (1-e^{-\theta})e^{-t})/\theta`.
    Radially symmetric, no tail dependence, and the full range
    :math:`\tau \in (-1, 1)` is attainable in ``dim=2``.

    Examples
    --------
    >>> from rcopula import FrankCopula
    >>> f = FrankCopula(theta=5.0)
    >>> float(round(f.tau(), 10))
    0.4567009582
    >>> float(round(f.rho(), 10))
    0.6434871081
    >>> f.lambda_()
    TailDependence(lower=0.0, upper=0.0)
    """

    generator_instance = _FrankGenerator()


class JoeCopula(_ConcreteArchimedean):
    r"""Joe copula: dependence concentrated in joint large values, more so than Gumbel.

    Use it, like :class:`GumbelCopula`, when joint highs matter, especially
    if upper-tail clustering is pronounced relative to the overall
    dependence. Positive dependence only; ``theta = 1`` is independence.

    Parameters
    ----------
    theta : float, default nan
        Dependence parameter, ``theta >= 1``. ``nan`` means "to be estimated".
    dim : int, default 2
        Number of variables, at least 2.
    free : array_like of bool, shape (1,), or None, default None
        Whether ``theta`` is estimated by :func:`~rcopula.fit.fit`
        (``None`` means free).

    Attributes
    ----------
    theta : float
        The dependence parameter.
    dim : int
        Number of variables.

    Raises
    ------
    ValueError
        If ``dim < 2`` or ``theta < 1``.

    Notes
    -----
    Generator :math:`\psi(t) = 1 - (1 - e^{-t})^{1/\theta}`, ``theta >= 1``.
    Upper-tail dependent, :math:`\lambda_U = 2 - 2^{1/\theta}`.

    Examples
    --------
    >>> from rcopula import JoeCopula
    >>> j = JoeCopula(theta=2.0)
    >>> float(round(j.tau(), 10))
    0.3550659332
    >>> float(round(j.lambda_().upper, 10))
    0.5857864376
    """

    generator_instance = _JoeGenerator()


class AMHCopula(_ConcreteArchimedean):
    r"""Ali-Mikhail-Haq (AMH) copula: a simple family for weak dependence only.

    Use it when dependence is mild: Kendall's tau can only range from about
    -0.18 to 1/3. ``theta = 0`` is independence.

    Parameters
    ----------
    theta : float, default nan
        Dependence parameter. Must lie in ``[-1, 1]`` for ``dim=2`` and in
        ``[0, 1]`` for ``dim > 2``. ``nan`` means "to be estimated".
    dim : int, default 2
        Number of variables, at least 2. (R's ``amhCopula`` allows only 2.)
    free : array_like of bool, shape (1,), or None, default None
        Whether ``theta`` is estimated by :func:`~rcopula.fit.fit`
        (``None`` means free).

    Attributes
    ----------
    theta : float
        The dependence parameter.
    dim : int
        Number of variables.

    Raises
    ------
    ValueError
        If ``dim < 2`` or ``theta`` is outside the admissible range.

    Notes
    -----
    Generator :math:`\psi(t) = (1-\theta)/(e^{t}-\theta)`. Reaches only weak
    dependence, :math:`\tau \in [-0.1817, 1/3]`, and has no tail dependence.

    Examples
    --------
    >>> from rcopula import AMHCopula
    >>> a = AMHCopula(theta=0.5)
    >>> float(round(a.tau(), 10))
    0.128764787
    >>> a.lambda_()
    TailDependence(lower=0.0, upper=0.0)
    """

    generator_instance = _AMHGenerator()
