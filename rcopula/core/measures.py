r"""Dependence measures by quadrature, for copulas with no closed form.

Structural copulas -- rotations, mixtures, Khoudraji devices -- generally have no
analytic Kendall's tau. Both measures are integrals of the copula itself,

.. math::
    \rho = 12\int_0^1\!\!\int_0^1 C(u,v)\,du\,dv - 3, \qquad
    \tau = 4\int_0^1\!\!\int_0^1 C(u,v)\,dC(u,v) - 1,

so a tensor-product rule evaluates them directly.

The rule is **tanh-sinh** (double exponential) rather than Gauss-Legendre, and
the reason is specific: a copula with tail dependence has a density that blows
up at a corner of the unit square. Gauss-Legendre assumes an analytic integrand
and degrades to algebraic convergence when that fails -- for a Gumbel copula it
reaches only ``4e-4`` with 128 nodes per axis and improves like :math:`n^{-2}`.
The tanh-sinh substitution
:math:`u = \tfrac{1}{2}(1 + \tanh(\tfrac{\pi}{2}\sinh t))` sends the node
spacing to zero doubly exponentially at each end, which absorbs endpoint
singularities; the same Gumbel case reaches ``5e-10`` with 159 nodes.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer,
    Theorems 5.1.1 and 5.1.3 -- the integral representations used here.
Takahasi, H. and Mori, M. (1974). Double exponential formulas for numerical
    integration. *Publications of RIMS, Kyoto University* 9(3), 721-741.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from rcopula.core.base import Copula

__all__ = ["rho_by_quadrature", "tau_by_partials", "tau_by_quadrature"]

#: Half-width of the tanh-sinh node index. 140 gives roughly 150 usable nodes
#: per axis and ~1e-9 even on families with a divergent corner density.
_LEVEL = 140

#: The substitution saturates in double precision past |t| ~ 3, so the step is
#: chosen to cover that range with the requested number of nodes.
_HALF_RANGE = 3.0


@lru_cache(maxsize=4)
def _nodes(level: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Tanh-sinh nodes and weights on the open interval ``(0, 1)``.

    With :math:`u(t) = \tfrac12(1 + \tanh(\tfrac\pi2 \sinh t))`,

    .. math::
        u'(t) = \frac{\pi\cosh t}{4\cosh^2(\tfrac\pi2\sinh t)},

    which underflows long before ``u`` reaches 0 or 1, so the nodes stay
    strictly interior and the weights stay finite.
    """
    step = _HALF_RANGE / level
    t = np.arange(-level, level + 1) * step
    inner = 0.5 * np.pi * np.sinh(t)
    with np.errstate(over="ignore"):
        u = 0.5 * (1.0 + np.tanh(inner))
        w = step * 0.25 * np.pi * np.cosh(t) / np.cosh(inner) ** 2
    keep = (u > 0.0) & (u < 1.0) & np.isfinite(w) & (w > 0.0)
    return u[keep], w[keep]


@lru_cache(maxsize=4)
def _tensor(level: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Flattened ``(m^2, 2)`` evaluation points and their product weights."""
    x, w = _nodes(level)
    uu, vv = np.meshgrid(x, x, indexing="ij")
    return np.column_stack([uu.ravel(), vv.ravel()]), np.outer(w, w).ravel()


def rho_by_quadrature(copula: Copula, level: int = _LEVEL) -> float:
    r"""Spearman's rho as :math:`12\int\int C\,du\,dv - 3`.

    Needs only the CDF, so it works for every bivariate copula -- including
    those with no density at all.

    Examples
    --------
    Reproduces the analytic value for families that have one:

    >>> from rcopula import ClaytonCopula, GumbelCopula
    >>> from rcopula.core.measures import rho_by_quadrature
    >>> for cop in (ClaytonCopula(2.0), GumbelCopula(2.5)):
    ...     print(bool(abs(rho_by_quadrature(cop) - cop.rho()) < 1e-8))
    True
    True
    """
    if copula.dim != 2:
        raise ValueError(f"quadrature is bivariate; got dim={copula.dim}")
    points, weights = _tensor(level)
    value = 12.0 * float(np.sum(weights * copula.cdf(points))) - 3.0
    return float(np.clip(value, -1.0, 1.0))


def tau_by_quadrature(copula: Copula, level: int = _LEVEL) -> float:
    r"""Kendall's tau as :math:`4\int\int C\,dC - 1 = 4\int\int C\,c\,du\,dv - 1`.

    Needs the density as well as the CDF. Unlike Spearman's rho, tau is **not**
    linear in the copula -- which is exactly why mixtures and rotations cannot
    simply average their components' values and have to come here instead.

    Examples
    --------
    >>> from rcopula import GumbelCopula, JoeCopula
    >>> from rcopula.core.measures import tau_by_quadrature
    >>> for cop in (GumbelCopula(2.5), JoeCopula(3.0)):
    ...     print(bool(abs(tau_by_quadrature(cop) - cop.tau()) < 1e-7))
    True
    True
    """
    if copula.dim != 2:
        raise ValueError(f"quadrature is bivariate; got dim={copula.dim}")
    points, weights = _tensor(level)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        density = copula.pdf(points)
    value = 4.0 * float(np.sum(weights * copula.cdf(points) * np.nan_to_num(density))) - 1.0
    return float(np.clip(value, -1.0, 1.0))


#: Gauss-Legendre nodes per axis for :func:`tau_by_partials`. The integrand
#: there is bounded in ``[0, 1]``, so an ordinary rule is appropriate and 400
#: nodes reproduce closed-form tau to eight digits.
_PARTIAL_NODES = 400

#: Step for differencing the CDF. Large enough that the difference is not
#: dominated by rounding in ``cdf``, small enough that the derivative is local.
_PARTIAL_STEP = 1e-6


def tau_by_partials(copula: Copula, nodes: int = _PARTIAL_NODES) -> float:
    r"""Kendall's tau as :math:`1 - 4\int\int \partial_1 C\,\partial_2 C\,du\,dv`.

    The same quantity as :func:`tau_by_quadrature`, integrated a different way,
    and better in two situations that matter.

    * **It needs only the CDF.** The :math:`\int\int C\,dC` form silently
      assumes the copula is absolutely continuous: a copula with a singular
      component -- Marshall-Olkin, or Clayton below zero -- carries mass that no
      density represents, and integrating the density alone misses it. Here the
      partial derivatives are conditional distribution functions, which exist
      regardless.
    * **The integrand is bounded in** :math:`[0, 1]`. Under strong dependence
      the density concentrates into a near-delta on the diagonal and no
      quadrature can resolve it: at Plackett ``theta = 1e4`` the density form
      returns 1.0 against a true 0.9756. The bounded form gets 0.97572.

    The trade is a numerical derivative, so it carries about 4e-8 rather than
    1e-8 -- immaterial next to what it buys.

    Examples
    --------
    Agrees with the closed forms:

    >>> from rcopula import ClaytonCopula, GumbelCopula
    >>> from rcopula.core.measures import tau_by_partials
    >>> for cop in (ClaytonCopula(2.0), GumbelCopula(3.0)):
    ...     print(bool(abs(tau_by_partials(cop) - cop.tau()) < 1e-7))
    True
    True

    And works where the density form cannot -- Marshall-Olkin puts mass on a
    curve, so its density integrates to less than one:

    >>> from rcopula import MarshallOlkinCopula
    >>> cop = MarshallOlkinCopula(0.7, 0.4)
    >>> bool(abs(tau_by_partials(cop) - cop.tau()) < 1e-3)
    True
    """
    if copula.dim != 2:
        raise ValueError(f"quadrature is bivariate; got dim={copula.dim}")
    x, w = np.polynomial.legendre.leggauss(nodes)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    uu, vv = np.meshgrid(x, x, indexing="ij")
    u, v = uu.ravel(), vv.ravel()

    step = _PARTIAL_STEP
    hi_u, lo_u = np.minimum(u + step, 1.0), np.maximum(u - step, 0.0)
    hi_v, lo_v = np.minimum(v + step, 1.0), np.maximum(v - step, 0.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        d1 = (copula.cdf(np.column_stack([hi_u, v])) - copula.cdf(np.column_stack([lo_u, v]))) / (
            hi_u - lo_u
        )
        d2 = (copula.cdf(np.column_stack([u, hi_v])) - copula.cdf(np.column_stack([u, lo_v]))) / (
            hi_v - lo_v
        )

    integrand = np.nan_to_num(d1 * d2).reshape(uu.shape)
    return float(np.clip(1.0 - 4.0 * np.einsum("i,j,ij->", w, w, integrand), -1.0, 1.0))
