"""The :class:`Copula` abstract base class and shared machinery.

Design notes
------------

**Copulas are immutable.** ``with_params`` returns a new instance rather than
mutating in place. R's ``setTheta`` mutates; that makes fitted objects aliasing
hazards and is not worth replicating. Allocating a small object per likelihood
evaluation is negligible next to the density evaluation itself, and families
expose parameter-explicit private hooks (``_logpdf``, ``_cdf``) so optimisers
never need to construct anything at all in their inner loop.

**Parameters may be partially fixed.** R threads a ``fixParam`` /
``isFree`` / ``nParam(freeOnly=)`` system through estimation; the same concept
lives here as a boolean ``free`` mask, honoured by ``fit``.

**Verbs follow scipy.** ``pdf`` / ``logpdf`` / ``cdf`` / ``rvs``, with
``random_state`` accepting a ``numpy.random.Generator``.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
    Definitions, the Frechet-Hoeffding bounds, and the d-increasing (C-volume)
    property implemented in :meth:`Copula.prob`.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Copula", "TailDependence"]


class TailDependence(NamedTuple):
    """Lower and upper tail-dependence coefficients.

    ``lower`` is :math:`\\lambda_L = \\lim_{u \\downarrow 0} C(u,u)/u` and
    ``upper`` is :math:`\\lambda_U = \\lim_{u \\uparrow 1} (1 - 2u + C(u,u))/(1-u)`.
    Both lie in ``[0, 1]``. A non-zero value means joint extremes occur with
    probability that does *not* vanish relative to marginal extremes — the
    property a Gaussian copula lacks and a t or Clayton copula has.
    """

    lower: float
    upper: float


class Copula(ABC):
    """Abstract base class for all copulas.

    Subclasses must define :attr:`param_names`, :attr:`param_bounds`, and
    implement :meth:`_logpdf`, :meth:`_cdf` and :meth:`_rvs`.
    """

    #: Human-readable family name, e.g. ``"Clayton"``.
    name: str = "Copula"

    #: Names of the parameters, in the order they appear in :attr:`params`.
    param_names: tuple[str, ...] = ()

    def __init__(
        self,
        params: ArrayLike,
        dim: int = 2,
        *,
        free: ArrayLike | None = None,
    ) -> None:
        self._dim = int(dim)
        if self._dim < 2:
            raise ValueError(f"dim must be at least 2, got {self._dim}")

        self._params = np.atleast_1d(np.asarray(params, dtype=np.float64)).copy()
        self._params.flags.writeable = False

        if free is None:
            free_arr = np.ones(self._params.shape, dtype=bool)
        else:
            free_arr = np.broadcast_to(np.asarray(free, dtype=bool), self._params.shape).copy()
        free_arr.flags.writeable = False
        self._free = free_arr

        self._validate_params()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Dimension ``d`` of the copula."""
        return self._dim

    #: Alias for :attr:`dim`, matching ``statsmodels``' spelling.
    @property
    def k_dim(self) -> int:
        return self._dim

    @property
    def params(self) -> NDArray[np.float64]:
        """Parameter vector (read-only)."""
        return self._params

    @property
    def free(self) -> NDArray[np.bool_]:
        """Boolean mask of which parameters are free to be estimated."""
        return self._free

    @property
    def n_params(self) -> int:
        """Number of free parameters."""
        return int(self._free.sum())

    @property
    @abstractmethod
    def param_bounds(self) -> list[tuple[float, float]]:
        """Open interval ``(lower, upper)`` for each parameter.

        Bounds may depend on :attr:`dim` — Clayton admits negative dependence in
        ``d = 2`` but not beyond, for instance.
        """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @abstractmethod
    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> Copula:
        """Build a new instance of this family with the given parameters and mask.

        Subclasses implement this rather than :meth:`with_params` /
        :meth:`fix_params` so that copies go through the real constructor.
        Building via ``object.__new__`` and patching attributes afterwards is
        fragile: validation runs before the family-specific attributes exist.
        """

    def with_params(self, params: ArrayLike) -> Copula:
        """Return a copy of this copula with new parameter values."""
        return self._reconstruct(params, self._free)

    def fix_params(self, free: ArrayLike) -> Copula:
        """Return a copy with the given free/fixed mask.

        Mirrors R's ``fixParam`` / ``fixedParam<-``. A ``False`` entry holds that
        parameter at its current value during estimation.
        """
        return self._reconstruct(self._params, free)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        if self._params.shape != (len(self.param_names),):
            raise ValueError(
                f"{self.name} copula expects {len(self.param_names)} parameter(s) "
                f"{self.param_names}, got {self._params.shape[0]}"
            )
        for value, nm, (lo, hi) in zip(
            self._params, self.param_names, self.param_bounds, strict=True
        ):
            if np.isnan(value):
                continue  # NaN marks "to be estimated", as in R's `claytonCopula()`
            if not (lo <= value <= hi):
                raise ValueError(
                    f"{self.name} copula: parameter {nm}={value!r} outside "
                    f"admissible range [{lo}, {hi}] for dim={self._dim}"
                )

    def _validate_u(self, u: ArrayLike) -> NDArray[np.float64]:
        """Coerce input to an ``(n, d)`` float array and check the unit cube."""
        arr = np.asarray(u, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"u must be 1- or 2-dimensional, got ndim={arr.ndim}")
        if arr.shape[1] != self._dim:
            raise ValueError(f"u has {arr.shape[1]} column(s) but the copula has dim={self._dim}")
        return arr

    def _require_specified(self) -> None:
        if np.isnan(self._params).any():
            raise ValueError(
                f"{self.name} copula has unspecified parameters "
                f"{dict(zip(self.param_names, self._params, strict=True))}; "
                "fit it or supply values before evaluation"
            )

    # ------------------------------------------------------------------
    # Abstract numerical core (parameter-explicit, for fast fitting)
    # ------------------------------------------------------------------

    @abstractmethod
    def _logpdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Log density on the open unit cube. ``u`` is ``(n, d)``, validated."""

    @abstractmethod
    def _cdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Distribution function. ``u`` is ``(n, d)``, validated."""

    @abstractmethod
    def _rvs(
        self, size: int, params: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Draw ``size`` observations. Returns ``(size, d)``."""

    # ------------------------------------------------------------------
    # Public evaluation
    # ------------------------------------------------------------------

    def logpdf(self, u: ArrayLike) -> NDArray[np.float64]:
        """Log of the copula density.

        Points outside :math:`[0,1]^d` have zero density, hence ``-inf`` log
        density — matching R, which treats out-of-range coordinates as boundary
        values even when another coordinate is NaN.
        """
        self._require_specified()
        arr = self._validate_u(u)

        out = np.full(arr.shape[0], -np.inf)
        inside = np.all((arr > 0.0) & (arr < 1.0), axis=1)
        if inside.any():
            out[inside] = self._logpdf(arr[inside], self._params)

        # NaN inputs propagate rather than silently reading as boundary.
        out[np.isnan(arr).any(axis=1) & ~np.any((arr < 0) | (arr > 1), axis=1)] = np.nan
        return out

    def pdf(self, u: ArrayLike) -> NDArray[np.float64]:
        """Copula density."""
        return np.exp(self.logpdf(u))

    def cdf(self, u: ArrayLike) -> NDArray[np.float64]:
        """Copula distribution function ``C(u)``."""
        self._require_specified()
        arr = np.clip(self._validate_u(u), 0.0, 1.0)

        out = np.empty(arr.shape[0])
        # Any coordinate at 0 forces C = 0; all coordinates at 1 gives C = 1.
        zero = np.any(arr <= 0.0, axis=1)
        out[zero] = 0.0
        rest = ~zero
        if rest.any():
            # A coordinate at exactly 1 is a legitimate query -- C(u, 1, ..., 1)
            # = u is the defining margin property -- but generators reach it
            # through infinite intermediates (psi^-1(1) = 0 via log(1 - 1)).
            # The limits are correct, so silence the boundary warnings here
            # rather than making every family special-case them.
            with np.errstate(divide="ignore", invalid="ignore"):
                out[rest] = self._cdf(arr[rest], self._params)
        return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)

    def rvs(
        self,
        size: int = 1,
        random_state: np.random.Generator | int | None = None,
    ) -> NDArray[np.float64]:
        """Draw pseudo-random observations from the copula.

        Parameters
        ----------
        size : int
            Number of observations.
        random_state : Generator, int or None
            Seed or generator. Accepting a ``Generator`` is the modern idiom;
            an ``int`` is promoted via ``np.random.default_rng``.

        Returns
        -------
        ndarray
            An ``(size, d)`` array of values in ``(0, 1)``.
        """
        self._require_specified()
        if size < 0:
            raise ValueError(f"size must be non-negative, got {size}")
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        return self._rvs(int(size), self._params, rng)

    # ------------------------------------------------------------------
    # Probability of a hypercube (the d-increasing / C-volume property)
    # ------------------------------------------------------------------

    def prob(self, lower: ArrayLike, upper: ArrayLike) -> float:
        """``P(lower_j < U_j <= upper_j for all j)`` — the C-volume of a box.

        Computed by the inclusion-exclusion sum over the ``2**d`` vertices,

        .. math::
            \\sum_{v \\in \\{0,1\\}^d} (-1)^{\\sum_j v_j}\\, C(x_v)

        where ``x_v`` takes ``lower_j`` when ``v_j = 1`` and ``upper_j`` otherwise.
        Being non-negative for every box is exactly what makes ``C`` a copula.

        Examples
        --------
        For the independence copula this is just the product of the side lengths:

        >>> import numpy as np
        >>> from rcopula import IndependenceCopula
        >>> c = IndependenceCopula(dim=2)
        >>> float(np.round(c.prob([0.25, 0.5], [1 / 3, 1.0]), 12))
        0.041666666667
        """
        lo = np.asarray(lower, dtype=np.float64).ravel()
        hi = np.asarray(upper, dtype=np.float64).ravel()
        if lo.shape != (self._dim,) or hi.shape != (self._dim,):
            raise ValueError(f"lower and upper must both have length dim={self._dim}")
        if np.any(lo > hi):
            raise ValueError("lower must be elementwise <= upper")

        # 2**d corners is fine for the dimensions copulas are used in; guard
        # against someone asking for d = 30 and waiting forever.
        if self._dim > 20:
            raise ValueError(
                f"prob() enumerates 2**d corners and is impractical for dim={self._dim}; "
                "estimate it by simulation instead"
            )

        corners = np.array(list(itertools.product(*zip(hi, lo, strict=True))))
        signs = np.array([(-1.0) ** sum(v) for v in itertools.product((0, 1), repeat=self._dim)])
        return float(np.sum(signs * self.cdf(corners)))

    # ------------------------------------------------------------------
    # Dependence measures
    # ------------------------------------------------------------------

    @abstractmethod
    def tau(self) -> float:
        """Population Kendall's tau."""

    @abstractmethod
    def rho(self) -> float:
        """Population Spearman's rho."""

    def beta(self) -> float:
        """Population Blomqvist's beta, ``2**d * C(1/2, ..., 1/2) - 1`` rescaled.

        For ``d = 2`` this is ``4 * C(1/2, 1/2) - 1``. Blomqvist's beta depends on
        the copula only at the centre point, which makes it cheap but insensitive
        to the tails.
        """
        d = self._dim
        centre = float(self.cdf(np.full((1, d), 0.5))[0])
        survival = float(self.prob(np.full(d, 0.5), np.ones(d)))
        return (2.0 ** (d - 1) * (centre + survival) - 1.0) / (2.0 ** (d - 1) - 1.0)

    @abstractmethod
    def lambda_(self) -> TailDependence:
        """Lower and upper tail-dependence coefficients."""

    # ------------------------------------------------------------------
    # Calibration from a target dependence (R's iTau / iRho)
    # ------------------------------------------------------------------

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> Copula:
        """Construct a copula calibrated to a target Kendall's tau.

        The Pythonic spelling of R's ``iTau``. A classmethod rather than an
        instance method because it *constructs* rather than mutates.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement calibration from Kendall's tau"
        )

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> Copula:
        """Construct a copula calibrated to a target Spearman's rho (R's ``iRho``)."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement calibration from Spearman's rho"
        )

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """One-line description, in the spirit of R's ``describeCop``."""
        if len(self.param_names) == 0:
            return f"{self.name} copula, dim {self._dim}"
        shown = ", ".join(
            f"{nm}={val:.6g}" + ("" if free else " (fixed)")
            for nm, val, free in zip(self.param_names, self._params, self._free, strict=True)
        )
        return f"{self.name} copula, dim {self._dim}, {shown}"

    def __repr__(self) -> str:
        return f"<{self.describe()}>"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return (
            self._dim == other._dim
            and np.array_equal(self._params, other._params, equal_nan=True)
            and np.array_equal(self._free, other._free)
        )

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._dim, self._params.tobytes()))
