r"""Mixtures of copulas.

A convex combination of copulas is a copula:

.. math::  C(\mathbf u) = \sum_{i} w_i\, C_i(\mathbf u), \qquad
           w_i \ge 0,\ \sum_i w_i = 1.

Nothing needs checking -- the defining properties are all preserved by convex
combination -- which makes this the cheapest way to build a family that no
single parametric form can express. The usual reason is **both tails at once**:
Clayton binds below and Gumbel above, and a mixture of the two has genuine
asymmetric dependence in *both* corners, which neither component and no
elliptical copula can produce.

Three quantities are exactly linear in the copula and so are exact weighted
averages of the components', with no integration at all:

* **Spearman's rho**, because :math:`\rho = 12\int\int C - 3`;
* **both tail-dependence coefficients**, because each is a limit of
  :math:`C(u,u)/u` or its survival counterpart;
* **Blomqvist's beta**, which depends on :math:`C` only at the centre.

**Kendall's tau is not**, since :math:`\tau = 4\int\int C\,dC - 1` is quadratic
in :math:`C` -- a mixture of two copulas with the same tau generally has a
different one. It is computed by quadrature.

Weights are carried on an unconstrained log-odds scale internally, as R does,
so that an optimiser sees a box rather than a simplex; :attr:`MixtureCopula.weights`
converts back.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer,
    section 3.2.4 -- convex sums of copulas.
Hu, L. (2006). Dependence patterns across financial markets: a mixed copula
    approach. *Applied Financial Economics* 16(10), 717-729.
    The both-tails mixture, and why a single family cannot do it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.base import Copula, TailDependence
from rcopula.core.measures import tau_by_partials

__all__ = ["MixtureCopula"]

#: Largest magnitude allowed on the log-odds scale. Beyond this a weight is zero
#: or one to within double precision, and the component is either absent or
#: alone -- so the bound costs nothing and keeps the optimiser in a finite box.
_MAX_LOG_ODDS = 30.0


class MixtureCopula(Copula):
    """Blend several copulas with fixed proportions, e.g. 40% Clayton and 60% Gumbel.

    A mixture behaves like "pick component ``i`` with probability ``w_i``, then
    draw from it". Its CDF and density are the weighted averages of the
    components'. Use it when no single family has the shape you need -- most
    often to get dependence in *both* the lower and the upper tail at once,
    which neither Clayton nor Gumbel (nor any elliptical copula) can give
    alone.

    The weights are parameters too, so :func:`~rcopula.fit.fit` can estimate
    them together with the components' parameters.

    Parameters
    ----------
    copulas : sequence of Copula
        Two or more components, all of the same dimension ``d``. Their
        parameters may be ``nan`` (to be fitted).
    weights : array_like of float, shape (k,), or None, default None
        Mixing weights for the ``k`` components, non-negative and summing to
        one. ``None`` gives equal weights ``1/k``.
    free : array_like of bool, shape (n_params,), or None, default None
        Keyword-only. Which parameters are estimated when fitting; ``None``
        means all of them.

    Attributes
    ----------
    copulas : list of Copula
        The components, in the order given.
    weights : numpy.ndarray of float64, shape (k,)
        The mixing weights (see :attr:`weights`).
    n_components : int
        Number of components ``k``.
    name : str
        ``"Mixture(<name 1>, <name 2>, ...)"``.
    param_names : tuple of str
        ``"c1.theta"``-style names for each component's parameters, then
        ``"logodds1", ..., "logodds{k-1}"``.

    Raises
    ------
    ValueError
        If fewer than two components are given, the components differ in
        dimension, the number of weights differs from the number of
        components, or the weights are negative or do not sum to one.

    Notes
    -----
    Parameters are the components' parameters followed by ``k - 1`` log-odds
    against the last component. That parameterisation is what makes the object
    fittable: the weights live on a simplex, which a box-constrained optimiser
    cannot represent, while the log-odds are unconstrained.

    Examples
    --------
    Clayton binds in the lower tail, Gumbel in the upper, and the mixture has
    both -- which neither component, and no elliptical copula, can manage:

    >>> from rcopula import ClaytonCopula, GumbelCopula
    >>> from rcopula.structural import MixtureCopula
    >>> mix = MixtureCopula([ClaytonCopula(3.0), GumbelCopula(2.5)], [0.4, 0.6])
    >>> lam = mix.lambda_()
    >>> bool(lam.lower > 0.3 and lam.upper > 0.3)
    True

    Tail dependence is an exact weighted average, so it needs no integration:

    >>> a, b = ClaytonCopula(3.0).lambda_(), GumbelCopula(2.5).lambda_()
    >>> bool(abs(lam.lower - 0.4 * a.lower) < 1e-12)
    True
    >>> bool(abs(lam.upper - 0.6 * b.upper) < 1e-12)
    True

    So is Spearman's rho:

    >>> expected = 0.4 * ClaytonCopula(3.0).rho() + 0.6 * GumbelCopula(2.5).rho()
    >>> bool(abs(mix.rho() - expected) < 1e-12)
    True

    Kendall's tau is *not*, and the clearest case is half comonotonicity and
    half independence. Spearman's rho is exactly 0.5, as averaging says it must
    be; Kendall's tau is 0.416, not 0.5:

    >>> from rcopula import FrechetUpperCopula, IndependenceCopula
    >>> half = MixtureCopula([FrechetUpperCopula(2), IndependenceCopula(2)], [0.5, 0.5])
    >>> float(round(half.rho(), 10))
    0.5
    >>> float(round(half.tau(), 4))
    0.4159
    """

    name = "Mixture"

    def __init__(
        self,
        copulas: Sequence[Copula],
        weights: ArrayLike | None = None,
        *,
        free: ArrayLike | None = None,
    ) -> None:
        parts = list(copulas)
        if len(parts) < 2:
            raise ValueError(f"a mixture needs at least two components, got {len(parts)}")
        dims = {c.dim for c in parts}
        if len(dims) != 1:
            raise ValueError(f"components must share a dimension, got {sorted(dims)}")

        w = (
            np.full(len(parts), 1.0 / len(parts))
            if weights is None
            else np.asarray(weights, dtype=np.float64).ravel()
        )
        if w.size != len(parts):
            raise ValueError(f"got {w.size} weights for {len(parts)} components")
        if np.any(w < 0.0):
            raise ValueError(f"weights must be non-negative, got {w}")
        if not np.isclose(w.sum(), 1.0):
            raise ValueError(f"weights must sum to 1, got {w.sum()}")

        self.copulas = parts
        self._sizes = [c.params.size for c in parts]
        self._n_component_params = int(sum(self._sizes))
        self.name = "Mixture(" + ", ".join(c.name for c in parts) + ")"
        self.param_names = tuple(
            f"c{i + 1}.{nm}" for i, c in enumerate(parts) for nm in c.param_names
        ) + tuple(f"logodds{i + 1}" for i in range(len(parts) - 1))

        super().__init__(
            np.concatenate([*(c.params for c in parts), _to_log_odds(w)]),
            parts[0].dim,
            free=free,
        )

    # -- plumbing ------------------------------------------------------

    @property
    def n_components(self) -> int:
        """Number of copulas being mixed.

        Returns
        -------
        int
            ``len(self.copulas)``, at least 2.
        """
        return len(self.copulas)

    @property
    def weights(self) -> NDArray[np.float64]:
        """The share of each component in the mixture.

        Recovered from the internal log-odds parameters, so it always reflects
        the current (e.g. fitted) values.

        Returns
        -------
        numpy.ndarray of float64, shape (k,)
            Non-negative weights summing to one, in component order.

        Examples
        --------
        >>> from rcopula import ClaytonCopula, GumbelCopula
        >>> from rcopula.structural import MixtureCopula
        >>> mix = MixtureCopula([ClaytonCopula(3.0), GumbelCopula(2.5)], [0.4, 0.6])
        >>> mix.weights.round(12).tolist()
        [0.4, 0.6]
        """
        return _from_log_odds(self._params[self._n_component_params :])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Allowed range of each parameter: the components' bounds, then the log-odds.

        Returns
        -------
        list of tuple of (float, float)
            One ``(lower, upper)`` pair per parameter. Each log-odds is bounded
            to ``[-30, 30]``, beyond which a weight is 0 or 1 to double
            precision.
        """
        return [
            *(b for c in self.copulas for b in c.param_bounds),
            *([(-_MAX_LOG_ODDS, _MAX_LOG_ODDS)] * (self.n_components - 1)),
        ]

    def _split(self, params: NDArray[np.float64]) -> tuple[list[Copula], NDArray[np.float64]]:
        out, at = [], 0
        for cop, size in zip(self.copulas, self._sizes, strict=True):
            out.append(cop.with_params(params[at : at + size]) if size else cop)
            at += size
        return out, _from_log_odds(params[at:])

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> MixtureCopula:
        parts, w = self._split(np.atleast_1d(np.asarray(params, dtype=np.float64)))
        return MixtureCopula(parts, w, free=free)

    def _validate_params(self) -> None:
        super()._validate_params()
        if np.isnan(self._params).any():
            return
        for cop in self._split(self._params)[0]:
            cop._validate_params()

    # -- evaluation ----------------------------------------------------

    def _cdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        parts, w = self._split(params)
        return np.asarray(sum(wi * c.cdf(u) for wi, c in zip(w, parts, strict=True)))

    def _logpdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Log of the mixed density, accumulated by ``logaddexp``.

        Summing the densities directly would underflow wherever one component is
        sharply peaked and the others are not -- which is exactly the situation a
        both-tails mixture is built to create.
        """
        parts, w = self._split(params)
        total = np.full(u.shape[0], -np.inf)
        for wi, cop in zip(w, parts, strict=True):
            if wi <= 0.0:
                continue
            total = np.logaddexp(total, np.log(wi) + cop.logpdf(u))
        return total

    def _rvs(
        self, size: int, params: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Pick a component, then draw from it -- the mixture's own definition."""
        parts, w = self._split(params)
        counts = rng.multinomial(size, w)
        blocks = [
            cop.rvs(int(n), random_state=rng) for cop, n in zip(parts, counts, strict=True) if n > 0
        ]
        out = np.vstack(blocks) if blocks else np.empty((0, self._dim))
        # The blocks arrive grouped by component; shuffle so that consecutive
        # draws are exchangeable, as a sample from the mixture should be.
        return np.asarray(rng.permutation(out, axis=0))

    # -- dependence ----------------------------------------------------

    def tau(self) -> float:
        """Return Kendall's tau of the mixture, a rank correlation between -1 and 1.

        Computed by quadrature -- it is **not** the weighted average of the
        components' taus. Tau is quadratic in the copula, so mixing two families with the same tau
        generally changes it. Averaging the components' values is the natural
        guess and it is wrong; see the class docstring.

        Returns
        -------
        float
            Kendall's tau.

        Raises
        ------
        ValueError
            If any parameter is unspecified (``nan``).
        NotImplementedError
            If ``dim != 2``.
        """
        self._require_specified()
        if self._dim != 2:
            raise NotImplementedError("Kendall's tau is implemented for dim=2")
        return tau_by_partials(self)

    def rho(self) -> float:
        r"""Return Spearman's rho of the mixture: exactly the weighted average of the components'.

        :math:`\rho = 12\int\int C - 3` is affine in :math:`C`, so mixing the
        copulas mixes their rhos. No integration needed.

        Returns
        -------
        float
            ``sum(w_i * rho_i)``.

        Raises
        ------
        ValueError
            If any parameter is unspecified (``nan``).
        """
        self._require_specified()
        values = [c.rho() for c in self.copulas]
        return float(np.dot(self.weights, np.atleast_1d(values).ravel()[: self.n_components]))

    def lambda_(self) -> TailDependence:
        r"""Return the chance of joint extremes in each tail, averaged over the components.

        Each coefficient is a limit of a quantity affine in :math:`C`, so it
        mixes. This is what lets a Clayton-Gumbel mixture have dependence in
        *both* corners while each component has it in only one.

        Returns
        -------
        TailDependence
            Named tuple ``(lower, upper)`` of floats in ``[0, 1]``.

        Raises
        ------
        ValueError
            If any parameter is unspecified (``nan``).
        """
        self._require_specified()
        w = self.weights
        pairs = [c.lambda_() for c in self.copulas]
        return TailDependence(
            lower=float(np.dot(w, [p.lower for p in pairs])),
            upper=float(np.dot(w, [p.upper for p in pairs])),
        )

    def beta(self) -> float:
        """Return Blomqvist's beta (medial correlation): the weighted average of the components'.

        Blomqvist's beta depends on ``C`` only at the centre, ``C(1/2, 1/2)``,
        so it mixes linearly.

        Returns
        -------
        float
            ``sum(w_i * beta_i)``.

        Raises
        ------
        ValueError
            If any parameter is unspecified (``nan``).
        """
        self._require_specified()
        return float(np.dot(self.weights, [c.beta() for c in self.copulas]))

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> Copula:
        """Not available: a mixture has more parameters than one tau value can identify.

        Raises
        ------
        NotImplementedError
            Always. Fit the mixture to data instead, or calibrate each
            component separately and choose the weights yourself.
        """
        raise NotImplementedError(
            "a mixture has more parameters than Kendall's tau can pin down; fit "
            "it, or calibrate the components and choose the weights"
        )

    def describe(self) -> str:
        """Return a multi-line, human-readable summary: one line per component with its weight.

        Returns
        -------
        str
            A header line followed by ``"  <weight> x <component description>"``
            rows.
        """
        rows = "\n".join(
            f"  {w:.4f} x {c.describe()}" for w, c in zip(self.weights, self.copulas, strict=True)
        )
        return f"Mixture copula, dim {self._dim}, {self.n_components} components\n{rows}"

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{w:.3g}*{c.name}" for w, c in zip(self.weights, self.copulas, strict=True)
        )
        return f"<Mixture copula, dim {self._dim}: {parts}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MixtureCopula):
            return NotImplemented
        return self.copulas == other.copulas and np.allclose(self.weights, other.weights)

    def __hash__(self) -> int:
        return hash(("Mixture", tuple(self.copulas), self.weights.tobytes()))


def _to_log_odds(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """Simplex to ``k - 1`` unconstrained values, against the last component."""
    w = np.clip(weights, 1e-300, None)
    return np.clip(np.log(w[:-1]) - np.log(w[-1]), -_MAX_LOG_ODDS, _MAX_LOG_ODDS)


def _from_log_odds(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of :func:`_to_log_odds`; a softmax with the last entry pinned."""
    full = np.append(np.clip(values, -_MAX_LOG_ODDS, _MAX_LOG_ODDS), 0.0)
    shifted = np.exp(full - full.max())
    return np.asarray(shifted / shifted.sum())
