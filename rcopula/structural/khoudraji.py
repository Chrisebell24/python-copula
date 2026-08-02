r"""Khoudraji's device: making an asymmetric copula out of symmetric ones.

Every Archimedean copula is **exchangeable** -- :math:`C(u,v) = C(v,u)` -- and so
is every elliptical one with equal correlations. That is a real restriction.
Exchangeability says the dependence between two variables looks the same viewed
from either side, and plenty of data disagrees: an equity index and its
volatility, a river's peak and its volume, a claim and its settlement cost.

Khoudraji (1995) gives a construction that breaks it, using two copulas and a
shape parameter per coordinate:

.. math::

    C(\mathbf u) = C_1\bigl(u_1^{1-a_1}, \dots, u_d^{1-a_d}\bigr)\;
                   C_2\bigl(u_1^{a_1},   \dots, u_d^{a_d}\bigr),
    \qquad a_j \in [0, 1].

Equal shapes give back an exchangeable copula; unequal ones tilt the dependence
towards one coordinate. At :math:`\mathbf a = 0` it is :math:`C_1`, at
:math:`\mathbf a = 1` it is :math:`C_2`, and in between it interpolates in a way
that is genuinely asymmetric rather than a mixture.

The usual recipe takes :math:`C_1` to be the independence copula and
:math:`C_2` something with tail dependence, which produces a family whose
dependence concentrates in one corner and one direction. That is what the
example below does.

Two facts make the construction practical rather than merely definable:

* **Sampling is exact and trivial.** Draw :math:`\mathbf U \sim C_1` and
  :math:`\mathbf V \sim C_2` independently and set
  :math:`W_j = \max(U_j^{1/(1-a_j)},\, V_j^{1/a_j})`. The maximum of two
  independent events gives the product of their probabilities, which is the
  definition above -- no inversion, no rejection.
* **The extreme-value class is closed under it.** If both components are
  extreme-value then so is the result, with a Pickands function assembled from
  theirs (see :meth:`KhoudrajiCopula.pickands`). This is *the* standard way to
  build asymmetric extreme-value copulas, and it is what gives the tail
  dependence coefficient in closed form.

References
----------
Khoudraji, A. (1995). Contributions a l'etude des copules et a la modelisation
    de valeurs extremes bivariees. PhD thesis, Universite Laval.
Genest, C., Ghoudi, K. and Rivest, L.-P. (1998). Discussion of
    "Understanding relationships using copulas" by Frees and Valdez.
    *North American Actuarial Journal* 2(3), 143-149.
    Where the device reached a wider audience.
Liebscher, E. (2008). Construction of asymmetric multivariate copulas.
    *Journal of Multivariate Analysis* 99(10), 2234-2250.
    The general form, of which this is the two-component case.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.base import Copula, TailDependence
from rcopula.core.extreme_value import ExtremeValueCopula
from rcopula.core.measures import rho_by_quadrature, tau_by_partials

__all__ = ["KhoudrajiCopula"]


class KhoudrajiCopula(Copula):
    """An asymmetric copula built from two symmetric ones.

    Parameters
    ----------
    copula1, copula2 : Copula
        The two components, of the same dimension. ``copula1`` is raised to the
        ``1 - a`` powers and ``copula2`` to the ``a`` powers, so ``shapes`` near
        zero recovers ``copula1`` and near one recovers ``copula2``.
    shapes : array_like
        One shape in ``[0, 1]`` per coordinate. **Equal shapes leave the result
        exchangeable**, so asymmetry requires them to differ.

    Notes
    -----
    Parameters are the two components' parameters followed by the shapes, so
    :func:`~rcopula.fit.fit` and :func:`~rcopula.select_copula` work on the
    whole object. The density is available in two dimensions; above that only
    the CDF and the sampler are, which is the same trichotomy R applies.

    Examples
    --------
    The canonical construction -- independence tilted by a Gumbel:

    >>> import numpy as np
    >>> from rcopula import GumbelCopula, IndependenceCopula
    >>> from rcopula.structural import KhoudrajiCopula
    >>> cop = KhoudrajiCopula(IndependenceCopula(2), GumbelCopula(3.0), [0.4, 0.95])

    It is genuinely asymmetric: swapping the arguments changes the value, which
    no Archimedean or exchangeable elliptical copula can do.

    >>> a = float(cop.cdf([[0.3, 0.7]])[0])
    >>> b = float(cop.cdf([[0.7, 0.3]])[0])
    >>> bool(abs(a - b) > 0.01)
    True

    Equal shapes hand back exchangeability:

    >>> even = KhoudrajiCopula(IndependenceCopula(2), GumbelCopula(3.0), [0.6, 0.6])
    >>> bool(abs(even.cdf([[0.3, 0.7]])[0] - even.cdf([[0.7, 0.3]])[0]) < 1e-12)
    True

    Both components extreme-value makes the result extreme-value too, so the
    tail dependence is exact rather than estimated:

    >>> float(round(cop.lambda_().upper, 10))
    0.3769268825
    """

    name = "Khoudraji"

    def __init__(
        self,
        copula1: Copula,
        copula2: Copula,
        shapes: ArrayLike,
        *,
        free: ArrayLike | None = None,
    ) -> None:
        if copula1.dim != copula2.dim:
            raise ValueError(
                f"components must share a dimension, got {copula1.dim} and {copula2.dim}"
            )
        a = np.atleast_1d(np.asarray(shapes, dtype=np.float64)).ravel()
        if a.size == 1:
            a = np.full(copula1.dim, float(a[0]))
        if a.size != copula1.dim:
            raise ValueError(f"shapes has length {a.size} but the copulas have dim={copula1.dim}")

        self.copula1, self.copula2 = copula1, copula2
        self._n1, self._n2 = copula1.params.size, copula2.params.size
        self.name = f"Khoudraji({copula1.name}, {copula2.name})"
        self.param_names = (
            tuple(f"c1.{nm}" for nm in copula1.param_names)
            + tuple(f"c2.{nm}" for nm in copula2.param_names)
            + tuple(f"shape{j + 1}" for j in range(copula1.dim))
        )
        super().__init__(
            np.concatenate([copula1.params, copula2.params, a]), copula1.dim, free=free
        )

    # -- plumbing ------------------------------------------------------

    @property
    def shapes(self) -> NDArray[np.float64]:
        """The per-coordinate shape parameters."""
        return self._params[self._n1 + self._n2 :]

    @property
    def is_exchangeable(self) -> bool:
        """Equal shapes leave the construction symmetric, defeating its purpose."""
        a = self.shapes
        return bool(np.allclose(a, a[0]))

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return [
            *self.copula1.param_bounds,
            *self.copula2.param_bounds,
            *([(0.0, 1.0)] * self._dim),
        ]

    def _split(self, params: NDArray[np.float64]) -> tuple[Copula, Copula, NDArray[np.float64]]:
        """Rebuild the two components and the shapes from a flat vector."""
        p1 = params[: self._n1]
        p2 = params[self._n1 : self._n1 + self._n2]
        a = params[self._n1 + self._n2 :]
        one = self.copula1.with_params(p1) if self._n1 else self.copula1
        two = self.copula2.with_params(p2) if self._n2 else self.copula2
        return one, two, a

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> KhoudrajiCopula:
        one, two, a = self._split(np.atleast_1d(np.asarray(params, dtype=np.float64)))
        return KhoudrajiCopula(one, two, a, free=free)

    def _validate_params(self) -> None:
        super()._validate_params()
        if np.isnan(self._params).any():
            return
        one, two, _ = self._split(self._params)
        one._validate_params()
        two._validate_params()

    # -- evaluation ----------------------------------------------------

    def _cdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        one, two, a = self._split(params)
        return np.asarray(one.cdf(u ** (1.0 - a)) * two.cdf(u**a))

    def _logpdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Density by the product rule, bivariate only.

        With :math:`x_j = u_j^{1-a_j}` and :math:`y_j = u_j^{a_j}`,

        .. math::
            \frac{\partial^2 C}{\partial u\,\partial v}
              = p r\, A_{12} B + p s\, A_1 B_2 + q r\, A_2 B_1 + q s\, A B_{12},

        where :math:`A, B` are the two component CDFs, subscripts are partial
        derivatives, and :math:`p, q, r, s` are the chain-rule factors
        :math:`\partial x_1/\partial u`, :math:`\partial y_1/\partial u`,
        :math:`\partial x_2/\partial v`, :math:`\partial y_2/\partial v`.

        The four terms are the four ways of assigning one derivative each to the
        two factors of the product. The component partials come from
        :func:`~rcopula.transforms.conditional_cdf`, which is analytic for
        Archimedean and elliptical families.
        """
        from rcopula.transforms import conditional_cdf

        if self._dim != 2:
            raise NotImplementedError(
                f"the Khoudraji density is implemented for dim=2; this copula has "
                f"dim={self._dim}. Its CDF and sampler work in any dimension."
            )
        one, two, a = self._split(params)
        a1, a2 = float(a[0]), float(a[1])
        x, y = u[:, 0], u[:, 1]

        left = np.column_stack([x ** (1.0 - a1), y ** (1.0 - a2)])
        right = np.column_stack([x**a1, y**a2])

        # Chain-rule factors for each component's two arguments.
        p, q = (1.0 - a1) * x ** (-a1), a1 * x ** (a1 - 1.0)
        r, s = (1.0 - a2) * y ** (-a2), a2 * y ** (a2 - 1.0)

        big_a, big_b = one.cdf(left), two.cdf(right)
        a_1 = conditional_cdf(one, left, given=0)
        a_2 = conditional_cdf(one, left, given=1)
        b_1 = conditional_cdf(two, right, given=0)
        b_2 = conditional_cdf(two, right, given=1)
        a_12 = one.pdf(left)
        b_12 = two.pdf(right)

        density = (
            p * r * a_12 * big_b + p * s * a_1 * b_2 + q * r * a_2 * b_1 + q * s * big_a * b_12
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(np.maximum(density, 0.0))

    def _rvs(
        self, size: int, params: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        r"""Exact sampler, no inversion required.

        :math:`W_j = \max(U_j^{1/(1-a_j)}, V_j^{1/a_j})` with independent
        :math:`\mathbf U \sim C_1` and :math:`\mathbf V \sim C_2`: the maximum of
        two independent events has the product of their probabilities, which is
        the definition of the copula.
        """
        one, two, a = self._split(params)
        first = one.rvs(size, random_state=rng)
        second = two.rvs(size, random_state=rng)
        with np.errstate(divide="ignore"):
            left = np.where(a < 1.0, first ** (1.0 / np.where(a < 1.0, 1.0 - a, 1.0)), 0.0)
            right = np.where(a > 0.0, second ** (1.0 / np.where(a > 0.0, a, 1.0)), 0.0)
        return np.clip(np.maximum(left, right), np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))

    # -- extreme-value structure ---------------------------------------

    @property
    def is_extreme_value(self) -> bool:
        """Whether the construction lands back in the extreme-value class.

        It does exactly when both components do -- and the Gumbel copula counts,
        being the one family that is both Archimedean and extreme-value, as does
        the independence copula.
        """
        from rcopula.core.archimedean import GumbelCopula
        from rcopula.core.other import IndependenceCopula

        def is_ev(cop: Copula) -> bool:
            return isinstance(cop, ExtremeValueCopula | GumbelCopula | IndependenceCopula)

        return self._dim == 2 and is_ev(self.copula1) and is_ev(self.copula2)

    def pickands(self, w: ArrayLike) -> NDArray[np.float64]:
        r"""Pickands dependence function, when both components are extreme-value.

        Substituting :math:`C_i(u,v) = (uv)^{A_i(t_i)}` into the definition and
        collecting the exponent of :math:`\log(uv)` gives

        .. math::
            A(t) = \bigl[(1-a_1)(1-t) + (1-a_2)t\bigr] A_1(t')
                 + \bigl[a_1(1-t) + a_2 t\bigr] A_2(t''),

        a convex combination of the components' Pickands functions evaluated at
        reweighted arguments. So the extreme-value class is closed under the
        device -- which is the whole reason it is the standard route to an
        *asymmetric* extreme-value copula.

        Examples
        --------
        Reproduces the copula's own CDF, which is the point:

        >>> import numpy as np
        >>> from rcopula import GumbelCopula, IndependenceCopula
        >>> from rcopula.structural import KhoudrajiCopula
        >>> cop = KhoudrajiCopula(IndependenceCopula(2), GumbelCopula(2.5), [0.3, 0.9])
        >>> u, v = 0.4, 0.7
        >>> t = np.log(v) / (np.log(u) + np.log(v))
        >>> direct = float(cop.cdf([[u, v]])[0])
        >>> viaA = float(np.exp((np.log(u) + np.log(v)) * cop.pickands(t)))
        >>> bool(abs(direct - viaA) < 1e-12)
        True
        """
        if not self.is_extreme_value:
            raise NotImplementedError(
                f"{self.name} is not an extreme-value copula: the Pickands "
                "representation needs both components to be extreme-value"
            )
        t = np.asarray(w, dtype=np.float64)
        a1, a2 = float(self.shapes[0]), float(self.shapes[1])

        first = (1.0 - a1) * (1.0 - t) + (1.0 - a2) * t
        second = a1 * (1.0 - t) + a2 * t
        with np.errstate(divide="ignore", invalid="ignore"):
            t_one = np.where(first > 0.0, (1.0 - a2) * t / np.where(first > 0.0, first, 1.0), 0.5)
            t_two = np.where(second > 0.0, a2 * t / np.where(second > 0.0, second, 1.0), 0.5)
        return np.asarray(
            first * _pickands_of(self.copula1, t_one) + second * _pickands_of(self.copula2, t_two)
        )

    # -- dependence ----------------------------------------------------

    def tau(self) -> float:
        """Kendall's tau, by quadrature -- the device admits no closed form."""
        self._require_specified()
        if self._dim != 2:
            raise NotImplementedError("Kendall's tau is implemented for dim=2")
        return tau_by_partials(self)

    def rho(self) -> float:
        """Spearman's rho, by quadrature."""
        self._require_specified()
        if self._dim != 2:
            raise NotImplementedError("Spearman's rho is implemented for dim=2")
        return rho_by_quadrature(self)

    def lambda_(self) -> TailDependence:
        r"""Tail dependence.

        Exact when both components are extreme-value, where
        :math:`\lambda_U = 2(1 - A(1/2))` and :math:`\lambda_L = 0`. Otherwise
        the limit depends on how each component behaves along a direction the
        shapes choose, which no single coefficient of the components records --
        so it is refused rather than guessed.
        """
        self._require_specified()
        if not self.is_extreme_value:
            raise NotImplementedError(
                f"tail dependence for {self.name} has no closed form unless both "
                "components are extreme-value; estimate it from a sample instead"
            )
        return TailDependence(lower=0.0, upper=float(2.0 * (1.0 - self.pickands(0.5))))

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> Copula:
        """Not available: the shapes and both components are not identified by tau."""
        raise NotImplementedError(
            "a Khoudraji copula has more parameters than Kendall's tau can pin "
            "down; fit it, or calibrate a component and choose the shapes"
        )

    def describe(self) -> str:
        shapes = ", ".join(f"{v:.4g}" for v in self.shapes)
        return (
            f"Khoudraji copula, dim {self._dim}, shapes=({shapes})\n"
            f"  component 1: {self.copula1.describe()}\n"
            f"  component 2: {self.copula2.describe()}"
        )

    def __repr__(self) -> str:
        shapes = ", ".join(f"{v:.4g}" for v in self.shapes)
        return f"<Khoudraji({self.copula1!r}, {self.copula2!r}, shapes=({shapes}))>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KhoudrajiCopula):
            return NotImplemented
        return (
            self.copula1 == other.copula1
            and self.copula2 == other.copula2
            and np.array_equal(self.shapes, other.shapes)
        )

    def __hash__(self) -> int:
        return hash(("Khoudraji", self.copula1, self.copula2, self.shapes.tobytes()))


def _pickands_of(copula: Copula, t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pickands function of any extreme-value component, including the two
    families that do not advertise themselves as one."""
    from rcopula.core.archimedean import GumbelCopula
    from rcopula.core.extreme_value import gumbel_pickands
    from rcopula.core.other import IndependenceCopula

    if isinstance(copula, IndependenceCopula):
        return np.ones_like(t)
    if isinstance(copula, GumbelCopula):
        return gumbel_pickands(t, copula.theta)
    if isinstance(copula, ExtremeValueCopula):
        return np.asarray(copula.A(t))
    raise NotImplementedError(f"{copula.name} is not an extreme-value copula")
