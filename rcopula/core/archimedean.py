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
from rcopula.special.logexp import log1mexp, signed_logsumexp
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
    """A stateless Archimedean generator, parameterised by ``theta``."""

    name: str = "generator"
    param_name: str = "theta"

    @abstractmethod
    def bounds(self, dim: int) -> tuple[float, float]:
        """Admissible ``(lower, upper)`` for ``theta`` in the given dimension."""

    @abstractmethod
    def psi(self, t: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
        """The generator :math:`\\psi(t)`."""

    @abstractmethod
    def ipsi(self, u: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
        """The inverse generator :math:`\\psi^{-1}(u)`."""

    @abstractmethod
    def log_abs_dpsi(self, t: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
        """:math:`\\log|\\psi'(t)|`."""

    @abstractmethod
    def log_abs_dpsi_d(self, t: NDArray[np.float64], theta: float, d: int) -> NDArray[np.float64]:
        """:math:`\\log|\\psi^{(d)}(t)|`, the ``d``-th derivative."""

    @abstractmethod
    def tau(self, theta: float) -> float:
        """Population Kendall's tau."""

    @abstractmethod
    def lambda_(self, theta: float) -> TailDependence:
        """Tail-dependence coefficients."""

    @abstractmethod
    def rvs_frailty(self, size: int, theta: float, rng: np.random.Generator) -> NDArray[np.float64]:
        """Draw the frailty :math:`V_0` whose Laplace transform is ``psi``."""

    def has_frailty(self, theta: float) -> bool:
        """Whether the Marshall-Olkin frailty representation applies at ``theta``.

        Sampling an Archimedean copula as :math:`U_j = \\psi(E_j / V)` requires
        :math:`\\psi` to be *completely* monotone, i.e. a Laplace transform. That
        holds only on the positively-dependent half of Clayton, Frank and AMH.
        On the negative half -- which exists in ``d = 2`` and is the reason
        those families are used at all -- there is no frailty, and the copula
        has to be sampled by conditional inversion instead.
        """
        return theta > 0.0

    def is_independent(self, theta: float) -> bool:
        """Whether this ``theta`` gives the independence copula.

        The degenerate point where most generators divide by zero, so callers
        short-circuit rather than evaluate.
        """
        return theta == 0.0

    def rho(self, theta: float) -> float:
        r"""Population Spearman's rho.

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
        return 12.0 * integral - 3.0

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
        """Invert Kendall's tau for ``theta`` (R's ``iTau``).

        The generic implementation is a bracketed root-find on the monotone
        ``tau(theta)`` curve; families with a closed form override it.
        """
        a, b = self._bracket(self.tau, tau, dim)
        return float(brentq(lambda th: self.tau(th) - tau, a, b, xtol=1e-14, rtol=8.9e-16))

    def irho(self, rho: float, dim: int = 2) -> float:
        """Invert Spearman's rho for ``theta`` (R's ``iRho``)."""
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
        return 2.0 * tau / (1.0 - tau)

    def lambda_(self, theta):
        return TailDependence(lower=2.0 ** (-1.0 / theta) if theta > 0 else 0.0, upper=0.0)

    def rvs_frailty(self, size, theta, rng):
        return rng.gamma(1.0 / theta, 1.0, size)


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
        h = -np.expm1(-theta)  # 1 - exp(-theta)
        return -np.log1p(-h * np.exp(-t)) / theta

    def ipsi(self, u, theta):
        r"""``-log( (1 - e^{-theta u}) / (1 - e^{-theta}) )``, computed stably.

        The naive quotient loses precision as ``u -> 1``: both numerator and
        denominator approach ``1``, so their ratio approaches ``1`` and the
        subsequent ``log`` cancels catastrophically. At ``theta = 20`` that costs
        about eight digits.

        Rewriting the ratio as ``1 + r`` with
        ``r = e^{-theta} * expm1(theta(1-u)) / expm1(-theta)`` and using
        ``log1p`` keeps the small quantity small throughout.
        """
        r = np.exp(-theta) * np.expm1(theta * (1.0 - u)) / np.expm1(-theta)
        return -np.log1p(r)

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
        return np.log(np.abs(_polylog_neg_int(z, d - 1, omz))) - np.log(abs(theta))

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

    def itau(self, tau, dim=2):
        if not -1.0 < tau < 1.0:
            raise ValueError(f"Frank requires tau in (-1, 1), got {tau}")
        if tau == 0.0:
            return 0.0
        lo, hi = (1e-12, 1e3) if tau > 0 else (-1e3, -1e-12)
        return float(brentq(lambda th: self.tau(th) - tau, lo, hi, xtol=1e-14, rtol=8.9e-16))

    def irho(self, rho: float, dim: int = 2) -> float:
        """Invert Spearman's rho for Frank, using its closed-form Debye expression."""
        if not -1.0 < rho < 1.0:
            raise ValueError(f"Frank requires rho in (-1, 1), got {rho}")
        if rho == 0.0:
            return 0.0
        lo, hi = (1e-12, 1e3) if rho > 0 else (-1e3, -1e-12)
        return float(brentq(lambda th: self.rho(th) - rho, lo, hi, xtol=1e-14, rtol=8.9e-16))

    def rvs_frailty(self, size, theta, rng):
        return rlog_series(size, -np.expm1(-theta), rng)


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
        # For theta < 0 both (1-theta) e^{-t} and Li_{-d}(z)/z can carry signs
        # that cancel. Take the absolute value of the *product*, not of each
        # factor: logging a negative polylog gives nan.
        return np.log1p(-theta) - t + np.log(np.abs(_polylog_neg_int_over_z(z, d)))

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
    r"""``Li_{-n}(z)`` for integer ``n >= 0`` and ``0 < z < 1``.

    Uses the Eulerian-number closed form

    .. math::

        \mathrm{Li}_{-n}(z) = \frac{1}{(1-z)^{n+1}}
            \sum_{k=0}^{n-1} A(n, k)\, z^{n-k}, \qquad n \ge 1,

    with :math:`\mathrm{Li}_0(z) = z/(1-z)`. This turns what would be an
    infinite sum into a degree-``n`` polynomial.

    Parameters
    ----------
    z : ndarray
        Argument in ``(0, 1)``.
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
    """A one-parameter Archimedean copula built from a generator."""

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
        """The dependence parameter."""
        return float(self._params[0])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
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
        t_j = g.ipsi(u, theta)
        t = t_j.sum(axis=1)
        return g.log_abs_dpsi_d(t, theta, d) - g.log_abs_dpsi(t_j, theta).sum(axis=1)

    def _cdf(self, u, params):
        theta = float(params[0])
        g = self.generator
        if g.is_independent(theta):
            return np.asarray(u.prod(axis=1))
        return g.psi(g.ipsi(u, theta).sum(axis=1), theta)

    def _rvs(self, size, params, rng):
        theta = float(params[0])
        g = self.generator
        if g.is_independent(theta):
            return rng.uniform(size=(size, self._dim))
        if not g.has_frailty(theta):
            return self._rvs_conditional(size, theta, rng)
        # Marshall-Olkin (1988): U_j = psi(E_j / V), with V the frailty.
        v = g.rvs_frailty(size, theta, rng)[:, None]
        e = rng.exponential(1.0, size=(size, self._dim))
        return g.psi(e / v, theta)

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
        self._require_specified()
        return float(self.generator.tau(self.theta))

    def rho(self) -> float:
        self._require_specified()
        return float(self.generator.rho(self.theta))

    def lambda_(self) -> TailDependence:
        self._require_specified()
        return self.generator.lambda_(self.theta)

    # -- generator passthroughs (R's psi / iPsi) -----------------------

    def psi(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the generator."""
        return self.generator.psi(np.asarray(t, dtype=np.float64), self.theta)

    def ipsi(self, u: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the inverse generator."""
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
        """Calibrate to a target Kendall's tau (R's ``iTau``)."""
        return cls(cls.generator_instance.itau(tau, dim), dim, **kwargs)

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> _ConcreteArchimedean:
        """Calibrate to a target Spearman's rho (R's ``iRho``)."""
        return cls(cls.generator_instance.irho(rho, dim), dim, **kwargs)


class ClaytonCopula(_ConcreteArchimedean):
    r"""Clayton copula.

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
    r"""Gumbel-Hougaard copula.

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
    r"""Frank copula.

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
    r"""Joe copula.

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
    r"""Ali-Mikhail-Haq copula.

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
