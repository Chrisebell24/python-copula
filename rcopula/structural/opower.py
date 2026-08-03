r"""The outer power transformation (R's ``opower``).

Every Archimedean family here is one-parameter, and one parameter buys one
shape: Clayton is always lower-tail dependent and never upper, Gumbel always the
reverse. The outer power transformation adds a second parameter that acts on the
*tail behaviour* while leaving the family's character intact.

Given a generator :math:`\psi` and :math:`\alpha \ge 1`, the transformed
generator is

.. math::  \psi_\alpha(t) = \psi\bigl(t^{1/\alpha}\bigr),

whose inverse is :math:`\psi_\alpha^{-1}(u) = \{\psi^{-1}(u)\}^{\alpha}`, so

.. math::

    C_\alpha(u_1, \dots, u_d)
      = \psi\Bigl(\bigl\{\textstyle\sum_j (\psi^{-1}(u_j))^{\alpha}\bigr\}^{1/\alpha}\Bigr).

Two facts make it worth having. Kendall's tau moves by a closed form,

.. math::  \tau_\alpha = 1 - \frac{1 - \tau_\psi}{\alpha},

so the transformation is a dial from the base family's dependence up towards
comonotonicity. And it *creates upper tail dependence where there was none*:
applied to the independence generator :math:`\psi(t) = e^{-t}` it produces
exactly the Gumbel copula with parameter :math:`\alpha`, which is the cleanest
possible statement of what it does.

============================  ================================================
:class:`OuterPowerCopula`     The transformed family.
:func:`opower`                Convenience constructor.
============================  ================================================

.. warning::

   The density is implemented for :math:`d = 2` only. In higher dimensions it
   needs the :math:`d`-th derivative of a *composition* of generators, which is
   the same open problem that leaves
   :class:`~rcopula.structural.nested.NestedArchimedean` without one. The CDF,
   Kendall's tau and tail dependence work in any dimension.

Examples
--------
>>> import rcopula as rc
>>> from rcopula.structural.opower import opower
>>> base = rc.ClaytonCopula(2.0)
>>> lifted = opower(base, 1.5)
>>> round(base.tau(), 4), round(lifted.tau(), 4)
(0.5, 0.6667)

Clayton has no upper tail dependence; the transformation supplies some:

>>> round(base.lambda_().upper, 4), round(lifted.lambda_().upper, 4)
(0.0, 0.4126)

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer,
    Theorem 4.5.1 and exercise 4.23 -- the transformation and its tau.
Hofert, M. (2010). *Sampling Nested Archimedean Copulas*. PhD thesis, Ulm
    University. The outer power transformation and its frailty.
Hofert, M. and Mächler, M. (2011). Nested Archimedean copulas meet R:
    the nacopula package. *J. Statistical Software* 39(9), 1-20.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.archimedean import ArchimedeanCopula
from rcopula.core.base import Copula, TailDependence

__all__ = ["OuterPowerCopula", "opower"]

#: Radius at which the diagonal limits defining tail dependence are evaluated.
#: Small enough to be in the limit for these generators, large enough that
#: `ipsi` has not lost its significant digits.
_TAIL_RADIUS = 1e-6


class OuterPowerCopula(Copula):
    r"""An Archimedean copula with its generator raised to an outer power.

    Parameters
    ----------
    base : ArchimedeanCopula
        Supplies :math:`\psi`. Its parameter is the first parameter here.
    alpha : float
        The outer power, :math:`\alpha \ge 1`. At :math:`\alpha = 1` the
        transformation is the identity and this *is* the base copula.
    free : array_like of bool, optional

    Notes
    -----
    Parameters are ordered ``(theta, alpha)``, so ``fit`` estimates both.

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from rcopula.structural.opower import OuterPowerCopula
    >>> cop = OuterPowerCopula(rc.ClaytonCopula(1.5), 2.0)
    >>> cop.dim, [round(float(p), 3) for p in cop.params]
    (2, [1.5, 2.0])

    At alpha = 1 it reduces to the base copula exactly:

    >>> identity = OuterPowerCopula(rc.ClaytonCopula(1.5), 1.0)
    >>> u = np.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.2]])
    >>> bool(np.allclose(identity.cdf(u), rc.ClaytonCopula(1.5).cdf(u)))
    True
    """

    def __init__(
        self,
        base: ArchimedeanCopula,
        alpha: float = np.nan,
        *,
        free: ArrayLike | None = None,
    ) -> None:
        if not isinstance(base, ArchimedeanCopula):
            raise TypeError(
                f"the outer power transformation acts on an Archimedean generator; "
                f"got {type(base).__name__}"
            )
        self.base = base
        self.generator = base.generator
        self.name = f"outer-power {base.generator.name}"
        self.param_names = (base.generator.param_name, "alpha")
        super().__init__([float(np.asarray(base.params)[0]), float(alpha)], base.dim, free=free)

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return [self.generator.bounds(self._dim), (1.0, np.inf)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> OuterPowerCopula:
        values = np.atleast_1d(np.asarray(params, dtype=float))
        # with_params is declared to return the base Copula type; an Archimedean
        # copula's reconstruction is always Archimedean, which the signature
        # cannot express.
        rebuilt = cast("ArchimedeanCopula", self.base.with_params([values[0]]))
        return OuterPowerCopula(rebuilt, float(values[1]), free=free)

    @property
    def alpha(self) -> float:
        """The outer power."""
        return float(self._params[1])

    # -- the transformed generator ----------------------------------------

    def _psi(self, t: NDArray[np.float64], theta: float, alpha: float) -> NDArray[np.float64]:
        return np.asarray(self.generator.psi(t ** (1.0 / alpha), theta), dtype=float)

    def _ipsi(self, u: NDArray[np.float64], theta: float, alpha: float) -> NDArray[np.float64]:
        return np.asarray(self.generator.ipsi(u, theta), dtype=float) ** alpha

    def _cdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        theta, alpha = float(params[0]), float(params[1])
        inverted = self._ipsi(u, theta, alpha)
        return self._psi(np.sum(inverted, axis=1), theta, alpha)

    def _logpdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._dim != 2:
            raise NotImplementedError(
                "the density of an outer-power copula is implemented for dim=2 only: "
                "in higher dimensions it needs the d-th derivative of a composition "
                "of generators, which is the same obstacle that leaves nested "
                "Archimedean copulas without a density. Its CDF, tau and tail "
                "dependence work in any dimension."
            )
        theta, alpha = float(params[0]), float(params[1])
        generator = self.generator

        # With x = ipsi(u), y = ipsi(v), T = x^a + y^a and r = T^(1/a),
        #
        #   c(u,v) = x^(a-1) y^(a-1) / (psi'(x) psi'(y))
        #            * T^(1/a - 2) * [ psi''(r) r - (a - 1) psi'(r) ].
        #
        # Both bracketed terms are positive: psi'' > 0, psi' < 0 and a >= 1. It
        # is assembled in logs because x^(a-1) and T^(1/a-2) each overflow long
        # before the density does.
        x = np.asarray(generator.ipsi(u[:, 0], theta), dtype=float)
        y = np.asarray(generator.ipsi(u[:, 1], theta), dtype=float)
        total = x**alpha + y**alpha
        radius = total ** (1.0 / alpha)

        log_first = (alpha - 1.0) * (np.log(x) + np.log(y))
        log_first -= generator.log_abs_dpsi(x, theta) + generator.log_abs_dpsi(y, theta)
        log_first += (1.0 / alpha - 2.0) * np.log(total)

        log_curv = generator.log_abs_dpsi_d(radius, theta, 2) + np.log(radius)
        log_slope = generator.log_abs_dpsi(radius, theta)
        bracket = np.exp(log_curv) + (alpha - 1.0) * np.exp(log_slope)
        return np.asarray(log_first + np.log(bracket), dtype=float)

    # -- measures ----------------------------------------------------------

    def tau(self) -> float:
        r"""Kendall's tau, :math:`1 - (1 - \tau_\psi)/\alpha`.

        Examples
        --------
        >>> import rcopula as rc
        >>> from rcopula.structural.opower import opower
        >>> round(opower(rc.ClaytonCopula(2.0), 2.0).tau(), 4)
        0.75
        """
        self._require_specified()
        return float(1.0 - (1.0 - self.base.with_params([self.params[0]]).tau()) / self.alpha)

    def rho(self) -> float:
        """Spearman's rho, by quadrature on the CDF."""
        from rcopula.core.measures import rho_by_quadrature

        self._require_specified()
        return float(rho_by_quadrature(self))

    def lambda_(self) -> TailDependence:
        r"""Tail dependence, from the diagonal limits.

        Evaluated numerically rather than from a closed form: the transformation
        changes the regular variation of :math:`\psi'` at each end differently
        for each base family, and a formula asserted for all of them would be
        wrong for some.

        Examples
        --------
        The transformation creates upper tail dependence in a family that has
        none:

        >>> import rcopula as rc
        >>> from rcopula.structural.opower import opower
        >>> rc.ClaytonCopula(2.0).lambda_().upper
        0.0
        >>> bool(opower(rc.ClaytonCopula(2.0), 2.0).lambda_().upper > 0.5)
        True
        """
        self._require_specified()
        radius = _TAIL_RADIUS
        low = np.array([[radius, radius]])
        high = np.array([[1.0 - radius, 1.0 - radius]])
        lower = float(np.asarray(self.cdf(low))[0] / radius)
        upper = float((1.0 - 2.0 * (1.0 - radius) + np.asarray(self.cdf(high))[0]) / radius)
        return TailDependence(
            lower=float(np.clip(lower, 0.0, 1.0)), upper=float(np.clip(upper, 0.0, 1.0))
        )

    def _rvs(
        self, size: int, params: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        if self._dim != 2:
            raise NotImplementedError(
                "sampling an outer-power copula is implemented for dim=2 only; it "
                "goes through the conditional distribution, which needs the density."
            )
        from rcopula.transforms import inverse_rosenblatt

        return np.asarray(inverse_rosenblatt(self, rng.uniform(size=(int(size), 2))), dtype=float)

    def describe(self) -> str:
        """One-line summary."""
        theta, alpha = (float(p) for p in self._params)
        return (
            f"Outer-power {self.generator.name} copula, dim {self._dim}, "
            f"theta={theta:g}, alpha={alpha:g}"
        )


def opower(base: ArchimedeanCopula, alpha: float, **kwargs: Any) -> OuterPowerCopula:
    r"""Raise an Archimedean generator to an outer power (R's ``opower``).

    Parameters
    ----------
    base : ArchimedeanCopula
    alpha : float
        The power, at least 1.

    Returns
    -------
    OuterPowerCopula

    Examples
    --------
    The cleanest statement of what the transformation does: applied to the
    independence generator it *is* the Gumbel copula.

    >>> import numpy as np, rcopula as rc
    >>> from rcopula.structural.opower import opower
    >>> nearly_independent = rc.ClaytonCopula(1e-10)
    >>> lifted = opower(nearly_independent, 2.5)
    >>> u = np.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.2]])
    >>> bool(np.max(np.abs(lifted.cdf(u) - rc.GumbelCopula(2.5).cdf(u))) < 1e-6)
    True

    And it moves Kendall's tau by a closed form:

    >>> base = rc.FrankCopula(4.0)
    >>> round(base.tau(), 4), round(opower(base, 3.0).tau(), 4)
    (0.3881, 0.796)
    """
    return OuterPowerCopula(base, alpha, **kwargs)
