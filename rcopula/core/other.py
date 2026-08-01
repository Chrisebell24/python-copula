r"""Copulas that are neither Archimedean nor elliptical.

Collected here: the independence copula, the two Frechet-Hoeffding bounds, and
three one-off parametric families that belong to neither big class --
Plackett, Farlie-Gumbel-Morgenstern, and Marshall-Olkin.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
    Section 2.5 for the Frechet-Hoeffding bounds, 3.3.1 for Plackett,
    Example 3.12 for FGM, and 3.1.1 for Marshall-Olkin.
Plackett, R. L. (1965). A class of bivariate distributions.
    *JASA* 60(310), 516-522.
Johnson, M. E. (1987). *Multivariate Statistical Simulation*. Wiley.
    The conditional sampler used for the Plackett copula.
Marshall, A. W. and Olkin, I. (1967). A multivariate exponential distribution.
    *JASA* 62(317), 30-44.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from rcopula.core.base import Copula, TailDependence

__all__ = [
    "FGMCopula",
    "FrechetLowerCopula",
    "FrechetUpperCopula",
    "IndependenceCopula",
    "MarshallOlkinCopula",
    "PlackettCopula",
]


class IndependenceCopula(Copula):
    r"""The independence copula :math:`\Pi(u) = \prod_j u_j`.

    The reference point for every dependence measure: all of ``tau``, ``rho``,
    ``beta`` and both tail-dependence coefficients are zero.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import IndependenceCopula
    >>> c = IndependenceCopula(dim=3)
    >>> float(c.cdf([[0.5, 0.5, 0.5]])[0])
    0.125
    >>> float(c.pdf([[0.2, 0.7, 0.9]])[0])
    1.0
    >>> float(c.tau()), float(c.rho())
    (0.0, 0.0)
    """

    name = "Independence"
    param_names: tuple[str, ...] = ()

    def __init__(self, dim: int = 2, **kwargs: object) -> None:
        super().__init__(np.empty(0), dim)

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return []

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> IndependenceCopula:
        return IndependenceCopula(self._dim)

    def _logpdf(self, u, params):
        return np.zeros(u.shape[0])

    def _cdf(self, u, params):
        return np.prod(u, axis=1)

    def _rvs(self, size, params, rng):
        return rng.uniform(size=(size, self._dim))

    def tau(self) -> float:
        return 0.0

    def rho(self) -> float:
        return 0.0

    def lambda_(self) -> TailDependence:
        return TailDependence(lower=0.0, upper=0.0)


class FrechetUpperCopula(Copula):
    r"""The Frechet-Hoeffding upper bound :math:`M(\mathbf{u}) = \min_j u_j`.

    Comonotonicity: every margin is an increasing function of every other. No
    density exists -- all the mass sits on the diagonal -- so :meth:`pdf` raises.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import FrechetUpperCopula
    >>> m = FrechetUpperCopula(dim=3)
    >>> float(m.cdf([[0.3, 0.7, 0.5]])[0])
    0.3
    >>> float(m.tau()), float(m.rho())
    (1.0, 1.0)
    >>> m.lambda_()
    TailDependence(lower=1.0, upper=1.0)

    Every margin of a draw is identical:

    >>> u = m.rvs(5, random_state=0)
    >>> bool(np.all(u[:, 0] == u[:, 2]))
    True
    """

    name = "FrechetUpper"
    param_names: tuple[str, ...] = ()

    def __init__(self, dim: int = 2, **kwargs: object) -> None:
        super().__init__(np.empty(0), dim)

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return []

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> FrechetUpperCopula:
        return FrechetUpperCopula(self._dim)

    def _logpdf(self, u, params):
        raise NotImplementedError(
            "the Frechet-Hoeffding upper bound is singular: all its mass lies on "
            "the diagonal, so it has no density with respect to Lebesgue measure"
        )

    def _cdf(self, u, params):
        return np.min(u, axis=1)

    def _rvs(self, size, params, rng):
        return np.repeat(rng.uniform(size=(size, 1)), self._dim, axis=1)

    def tau(self) -> float:
        return 1.0

    def rho(self) -> float:
        return 1.0

    def lambda_(self) -> TailDependence:
        return TailDependence(lower=1.0, upper=1.0)


class FrechetLowerCopula(Copula):
    r"""The Frechet-Hoeffding lower bound :math:`W(u, v) = \max(u + v - 1, 0)`.

    Countermonotonicity. **Bivariate only** -- in three or more dimensions
    :math:`W` is not a copula at all (its C-volume can be negative), which is
    why R restricts ``lowfhCopula`` the same way. Singular, so no density.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import FrechetLowerCopula
    >>> w = FrechetLowerCopula()
    >>> float(w.cdf([[0.3, 0.4]])[0])
    0.0
    >>> float(w.cdf([[0.7, 0.8]])[0])
    0.5
    >>> float(w.tau()), float(w.rho())
    (-1.0, -1.0)

    Draws are exactly countermonotone:

    >>> u = w.rvs(5, random_state=0)
    >>> bool(np.allclose(u[:, 0] + u[:, 1], 1.0))
    True

    Higher dimensions are refused rather than returning something invalid:

    >>> FrechetLowerCopula(dim=3)
    Traceback (most recent call last):
        ...
    ValueError: the Frechet-Hoeffding lower bound is only a copula for dim=2, got dim=3
    """

    name = "FrechetLower"
    param_names: tuple[str, ...] = ()

    def __init__(self, dim: int = 2, **kwargs: object) -> None:
        if int(dim) != 2:
            raise ValueError(
                f"the Frechet-Hoeffding lower bound is only a copula for dim=2, got dim={int(dim)}"
            )
        super().__init__(np.empty(0), 2)

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return []

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> FrechetLowerCopula:
        return FrechetLowerCopula()

    def _logpdf(self, u, params):
        raise NotImplementedError(
            "the Frechet-Hoeffding lower bound is singular: all its mass lies on "
            "the antidiagonal, so it has no density with respect to Lebesgue measure"
        )

    def _cdf(self, u, params):
        return np.maximum(u.sum(axis=1) - 1.0, 0.0)

    def _rvs(self, size, params, rng):
        v = rng.uniform(size=size)
        return np.column_stack([v, 1.0 - v])

    def tau(self) -> float:
        return -1.0

    def rho(self) -> float:
        return -1.0

    def lambda_(self) -> TailDependence:
        return TailDependence(lower=0.0, upper=0.0)


class PlackettCopula(Copula):
    r"""Plackett copula (bivariate).

    Defined by a constant cross-product ratio :math:`\theta > 0`:

    .. math::

        C(u,v) = \frac{S - \sqrt{S^2 - 4uv\theta(\theta-1)}}{2(\theta - 1)},
        \qquad S = 1 + (\theta - 1)(u + v).

    Spans the full dependence range -- :math:`\theta \to 0` gives the
    countermonotone bound, :math:`\theta = 1` independence,
    :math:`\theta \to \infty` the comonotone bound -- with no tail dependence
    anywhere. Spearman's rho has a closed form; Kendall's tau does not.

    Examples
    --------
    >>> from rcopula import PlackettCopula
    >>> p = PlackettCopula(3.0)
    >>> float(round(p.rho(), 10))
    0.352081567
    >>> p.lambda_()
    TailDependence(lower=0.0, upper=0.0)

    ``theta = 1`` is exactly independence:

    >>> float(PlackettCopula(1.0).cdf([[0.3, 0.4]])[0])
    0.12
    >>> float(PlackettCopula(1.0).pdf([[0.3, 0.4]])[0])
    1.0
    """

    name = "Plackett"
    param_names = ("theta",)

    def __init__(
        self, theta: float = np.nan, dim: int = 2, *, free: ArrayLike | None = None
    ) -> None:
        if int(dim) != 2:
            raise ValueError(f"the Plackett copula is bivariate only, got dim={int(dim)}")
        super().__init__([theta], 2, free=free)

    @property
    def theta(self) -> float:
        """The cross-product ratio."""
        return float(self._params[0])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, np.inf)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> PlackettCopula:
        return PlackettCopula(float(np.atleast_1d(params)[0]), free=free)

    @staticmethod
    def _eta_s_d(x, y, theta):
        eta = theta - 1.0
        s = 1.0 + eta * (x + y)
        d = np.sqrt(np.maximum(s * s - 4.0 * x * y * theta * eta, 0.0))
        return eta, s, d

    def _cdf(self, u, params):
        theta = float(params[0])
        if theta == 1.0:
            return u[:, 0] * u[:, 1]
        eta, s, d = self._eta_s_d(u[:, 0], u[:, 1], theta)
        return (s - d) / (2.0 * eta)

    def _logpdf(self, u, params):
        theta = float(params[0])
        if theta == 1.0:
            return np.zeros(u.shape[0])
        x, y = u[:, 0], u[:, 1]
        eta, _, d = self._eta_s_d(x, y, theta)
        num = theta * (1.0 + eta * (x + y - 2.0 * x * y))
        return np.log(num) - 3.0 * np.log(d)

    def _rvs(self, size, params, rng):
        theta = float(params[0])
        u = rng.uniform(size=size)
        w = rng.uniform(size=size)
        return np.column_stack([u, self._conditional_inverse(u, w, theta)])

    @staticmethod
    def _conditional_inverse(u, w, theta):
        """Invert ``dC/du = w`` for ``v`` (Johnson 1987, section 11.9)."""
        if theta == 1.0:
            return w
        a = w * (1.0 - w)
        b = theta + a * (theta - 1.0) ** 2
        c = 2.0 * a * (u * theta**2 + 1.0 - u) + theta * (1.0 - 2.0 * a)
        d = np.sqrt(np.maximum(theta * (theta + 4.0 * a * u * (1.0 - u) * (1.0 - theta) ** 2), 0.0))
        return (c - (1.0 - 2.0 * w) * d) / (2.0 * b)

    def tau(self) -> float:
        """Kendall's tau by quadrature -- Plackett admits no closed form.

        Uses :math:`\\tau = 4\\int\\!\\!\\int C\\,c\\;du\\,dv - 1`.
        """
        self._require_specified()
        # 100 nodes is already converged to 12 digits (checked against adaptive
        # quadrature at n = 100, 200, 400, 800 and 1600).
        nodes, weights = np.polynomial.legendre.leggauss(100)
        x = 0.5 * (nodes + 1.0)
        wt = 0.5 * weights
        uu, vv = np.meshgrid(x, x, indexing="ij")
        grid = np.column_stack([uu.ravel(), vv.ravel()])
        integrand = (self.cdf(grid) * self.pdf(grid)).reshape(uu.shape)
        return float(4.0 * np.einsum("i,j,ij->", wt, wt, integrand) - 1.0)

    def rho(self) -> float:
        r""":math:`\rho = \frac{\theta+1}{\theta-1} - \frac{2\theta\log\theta}{(\theta-1)^2}`."""
        self._require_specified()
        theta = self.theta
        if abs(theta - 1.0) < 1e-5:
            # 0/0 at theta = 1; expand instead. The limit is 0.
            e = theta - 1.0
            return float(e / 3.0 - e * e / 6.0)
        return float(
            (theta + 1.0) / (theta - 1.0) - 2.0 * theta * np.log(theta) / (theta - 1.0) ** 2
        )

    def lambda_(self) -> TailDependence:
        return TailDependence(lower=0.0, upper=0.0)

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> PlackettCopula:
        """Calibrate to a target Spearman's rho by inverting the closed form."""
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {rho}")
        if rho == 0.0:
            return cls(1.0)
        lo, hi = (1.0, 1e8) if rho > 0 else (1e-8, 1.0)
        return cls(float(brentq(lambda th: cls(th).rho() - rho, lo, hi, xtol=1e-13)))


class FGMCopula(Copula):
    r"""Farlie-Gumbel-Morgenstern copula (bivariate).

    .. math::  C(u,v) = uv\bigl(1 + \theta(1-u)(1-v)\bigr), \qquad |\theta| \le 1.

    A perturbation of independence, and a *weak* one: dependence is capped at
    :math:`|\tau| \le 2/9` and :math:`|\rho| \le 1/3`. Useful as a tractable toy
    and for testing procedures near independence, rarely as a serious model.

    Examples
    --------
    >>> from rcopula import FGMCopula
    >>> f = FGMCopula(0.5)
    >>> float(round(f.tau(), 12)), float(round(f.rho(), 12))
    (0.111111111111, 0.166666666667)
    >>> f.lambda_()
    TailDependence(lower=0.0, upper=0.0)

    Even maximal theta gives only weak dependence:

    >>> float(round(FGMCopula(1.0).tau(), 12))
    0.222222222222
    """

    name = "FGM"
    param_names = ("theta",)

    def __init__(
        self, theta: float = np.nan, dim: int = 2, *, free: ArrayLike | None = None
    ) -> None:
        if int(dim) != 2:
            raise ValueError(
                f"this FGM implementation is bivariate only, got dim={int(dim)}. "
                "The d-dimensional form needs 2**d - d - 1 parameters "
                "(1013 already at d=10) and is not yet supported."
            )
        super().__init__([theta], 2, free=free)

    @property
    def theta(self) -> float:
        return float(self._params[0])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return [(-1.0, 1.0)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> FGMCopula:
        return FGMCopula(float(np.atleast_1d(params)[0]), free=free)

    def _cdf(self, u, params):
        theta = float(params[0])
        x, y = u[:, 0], u[:, 1]
        return x * y * (1.0 + theta * (1.0 - x) * (1.0 - y))

    def _logpdf(self, u, params):
        theta = float(params[0])
        x, y = u[:, 0], u[:, 1]
        return np.log(1.0 + theta * (1.0 - 2.0 * x) * (1.0 - 2.0 * y))

    def _rvs(self, size, params, rng):
        theta = float(params[0])
        u = rng.uniform(size=size)
        w = rng.uniform(size=size)
        a = theta * (1.0 - 2.0 * u)
        # Solve a v^2 - (1 + a) v + w = 0 for the root lying in (0, 1).
        safe_a = np.where(np.abs(a) < 1e-12, 1.0, a)
        root = (1.0 + a - np.sqrt(np.maximum((1.0 + a) ** 2 - 4.0 * a * w, 0.0))) / (2.0 * safe_a)
        v = np.where(np.abs(a) < 1e-12, w, root)
        return np.column_stack([u, v])

    def tau(self) -> float:
        r""":math:`\tau = 2\theta/9`."""
        self._require_specified()
        return 2.0 * self.theta / 9.0

    def rho(self) -> float:
        r""":math:`\rho = \theta/3`."""
        self._require_specified()
        return self.theta / 3.0

    def lambda_(self) -> TailDependence:
        return TailDependence(lower=0.0, upper=0.0)

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> FGMCopula:
        if not -2.0 / 9.0 <= tau <= 2.0 / 9.0:
            raise ValueError(f"FGM attains only tau in [-2/9, 2/9], got {tau}")
        return cls(9.0 * tau / 2.0)

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> FGMCopula:
        if not -1.0 / 3.0 <= rho <= 1.0 / 3.0:
            raise ValueError(f"FGM attains only rho in [-1/3, 1/3], got {rho}")
        return cls(3.0 * rho)


class MarshallOlkinCopula(Copula):
    r"""Marshall-Olkin copula (bivariate).

    .. math::

        C(u,v) = \min\bigl(u\, v^{1-\alpha_2},\ u^{1-\alpha_1}\, v\bigr),
        \qquad \alpha_1, \alpha_2 \in [0, 1].

    Arises from a shared-shock model: two components fail from their own private
    shocks or from a common one. **Not absolutely continuous** -- it places mass
    on the curve :math:`u^{\alpha_1} = v^{\alpha_2}`, so :meth:`pdf` describes
    only the continuous part and is undefined there. Asymmetric unless
    :math:`\alpha_1 = \alpha_2`, and upper-tail dependent with
    :math:`\lambda_U = \min(\alpha_1, \alpha_2)`.

    Examples
    --------
    >>> from rcopula import MarshallOlkinCopula
    >>> mo = MarshallOlkinCopula(0.2, 0.8)
    >>> float(round(mo.tau(), 10))
    0.1904761905
    >>> mo.lambda_()
    TailDependence(lower=0.0, upper=0.2)

    Asymmetry is real -- swapping the arguments changes the value:

    >>> a = float(mo.cdf([[0.3, 0.7]])[0])
    >>> b = float(mo.cdf([[0.7, 0.3]])[0])
    >>> bool(abs(a - b) > 1e-3)
    True
    """

    name = "MarshallOlkin"
    param_names = ("alpha1", "alpha2")

    def __init__(
        self,
        alpha1: ArrayLike = np.nan,
        alpha2: float = np.nan,
        dim: int = 2,
        *,
        free: ArrayLike | None = None,
    ) -> None:
        if int(dim) != 2:
            raise ValueError(
                f"this Marshall-Olkin implementation is bivariate only, got dim={int(dim)}"
            )
        given = np.atleast_1d(np.asarray(alpha1, dtype=np.float64))
        params = given if given.size == 2 else np.array([float(given[0]), float(alpha2)])
        super().__init__(params, 2, free=free)

    @property
    def alpha(self) -> NDArray[np.float64]:
        """The two shock parameters."""
        return self._params

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0), (0.0, 1.0)]

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> MarshallOlkinCopula:
        return MarshallOlkinCopula(np.atleast_1d(params), free=free)

    def _cdf(self, u, params):
        a1, a2 = float(params[0]), float(params[1])
        x, y = u[:, 0], u[:, 1]
        return np.minimum(x * y ** (1.0 - a2), x ** (1.0 - a1) * y)

    def _logpdf(self, u, params):
        a1, a2 = float(params[0]), float(params[1])
        x, y = u[:, 0], u[:, 1]
        # Absolutely continuous part only; the singular mass carried on
        # u**a1 == v**a2 is not representable as a density.
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.log1p(-a2) - a2 * np.log(y)
            right = np.log1p(-a1) - a1 * np.log(x)
            return np.where(x**a1 > y**a2, left, right)

    def _rvs(self, size, params, rng):
        a1, a2 = float(params[0]), float(params[1])
        v = rng.uniform(size=(size, 3))
        # Marshall & Olkin (1967): two private shocks and one common shock.
        with np.errstate(divide="ignore"):
            first = v[:, 0] ** (1.0 / (1.0 - a1)) if a1 < 1.0 else np.zeros(size)
            second = v[:, 1] ** (1.0 / (1.0 - a2)) if a2 < 1.0 else np.zeros(size)
            u1 = np.maximum(first, v[:, 2] ** (1.0 / a1) if a1 > 0 else 0.0)
            u2 = np.maximum(second, v[:, 2] ** (1.0 / a2) if a2 > 0 else 0.0)
        return np.column_stack([u1, u2])

    def tau(self) -> float:
        r""":math:`\tau = \alpha_1\alpha_2/(\alpha_1 + \alpha_2 - \alpha_1\alpha_2)`."""
        self._require_specified()
        a1, a2 = self.alpha
        denom = a1 + a2 - a1 * a2
        return 0.0 if denom == 0 else float(a1 * a2 / denom)

    def rho(self) -> float:
        r""":math:`\rho = 3\alpha_1\alpha_2/(2\alpha_1 + 2\alpha_2 - \alpha_1\alpha_2)`."""
        self._require_specified()
        a1, a2 = self.alpha
        denom = 2.0 * a1 + 2.0 * a2 - a1 * a2
        return 0.0 if denom == 0 else float(3.0 * a1 * a2 / denom)

    def lambda_(self) -> TailDependence:
        self._require_specified()
        return TailDependence(lower=0.0, upper=float(min(self.alpha)))
