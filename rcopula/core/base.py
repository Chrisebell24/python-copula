"""The :class:`Copula` abstract base class and shared machinery.

Every copula family in ``rcopula`` inherits from :class:`Copula`, so the
methods documented there (density, distribution function, sampling, rank
correlations, tail dependence, calibration) work the same way for all of them.

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
    """How likely two variables are to hit extremes together, in each tail.

    The pair returned by :meth:`Copula.lambda_`. ``lower`` measures the chance
    that both variables are very small at the same time (joint crashes);
    ``upper`` measures the chance that both are very large at the same time
    (joint booms). It behaves like an ordinary tuple, so
    ``lower, upper = cop.lambda_()`` works.

    Attributes
    ----------
    lower : float
        Lower tail-dependence coefficient, in ``[0, 1]``. Zero means joint
        extreme lows become vanishingly rare relative to single extreme lows.
    upper : float
        Upper tail-dependence coefficient, in ``[0, 1]``. Zero means joint
        extreme highs become vanishingly rare relative to single extreme highs.

    Notes
    -----
    ``lower`` is :math:`\\lambda_L = \\lim_{u \\downarrow 0} C(u,u)/u` and
    ``upper`` is :math:`\\lambda_U = \\lim_{u \\uparrow 1} (1 - 2u + C(u,u))/(1-u)`.
    Both lie in ``[0, 1]``. A non-zero value means joint extremes occur with
    probability that does *not* vanish relative to marginal extremes — the
    property a Gaussian copula lacks and a t or Clayton copula has.

    Examples
    --------
    >>> from rcopula.core.base import TailDependence
    >>> td = TailDependence(lower=0.5, upper=0.0)
    >>> td.lower, td.upper
    (0.5, 0.0)
    """

    lower: float
    upper: float


class Copula(ABC):
    """Common interface shared by every copula family in the package.

    A copula describes *only* how variables move together, separately from
    what each variable looks like on its own. It works on values in the unit
    interval ``[0, 1]`` (probabilities / ranks), one column per variable.
    You do not create a ``Copula`` directly; you create a concrete family such
    as :class:`~rcopula.ClaytonCopula` or :class:`~rcopula.GaussianCopula`,
    which all share the methods documented here (``pdf``, ``cdf``, ``rvs``,
    ``tau``, ``from_tau`` and so on).

    Copulas are immutable: methods such as :meth:`with_params` return a new
    object instead of changing the existing one.

    Parameters
    ----------
    params : array_like of float, shape (n_params,)
        Parameter values, in the order given by :attr:`param_names`. A value of
        ``nan`` means "not yet known, to be estimated by fitting"; such a
        copula can be fitted but not evaluated.
    dim : int, default 2
        Number of variables (columns) the copula joins. Must be at least 2.
    free : array_like of bool, shape (n_params,), or None, default None
        Keyword-only. Which parameters are estimated when fitting (``True``)
        and which are held at their current value (``False``). ``None`` means
        all parameters are free. A scalar is broadcast to every parameter.

    Attributes
    ----------
    name : str
        Human-readable family name, e.g. ``"Clayton"``.
    param_names : tuple of str
        Names of the parameters, in the order they appear in :attr:`params`.
    dim : int
        Number of variables.
    params : numpy.ndarray of float, shape (n_params,)
        Current parameter values (read-only).
    free : numpy.ndarray of bool, shape (n_params,)
        Mask of parameters that are estimated when fitting.

    Raises
    ------
    ValueError
        If ``dim < 2``, if the number of parameters is wrong for the family,
        or if a (non-``nan``) parameter lies outside :attr:`param_bounds`.

    Notes
    -----
    Subclasses must define :attr:`param_names`, :attr:`param_bounds`, and
    implement :meth:`_logpdf`, :meth:`_cdf` and :meth:`_rvs`.

    Examples
    --------
    >>> from rcopula import ClaytonCopula
    >>> c = ClaytonCopula(2.0, dim=2)
    >>> c.dim, c.param_names
    (2, ('theta',))
    >>> c.rvs(3, random_state=0).shape
    (3, 2)
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
        """Number of variables the copula joins (its dimension ``d``).

        Returns
        -------
        int
            The dimension, at least 2.
        """
        return self._dim

    #: Alias for :attr:`dim`, matching ``statsmodels``' spelling.
    @property
    def k_dim(self) -> int:
        """Same as :attr:`dim`; the name ``statsmodels`` uses.

        Returns
        -------
        int
            The dimension, at least 2.
        """
        return self._dim

    @property
    def params(self) -> NDArray[np.float64]:
        """Current parameter values, in the order of :attr:`param_names`.

        The array is read-only; use :meth:`with_params` to get a copula with
        different values.

        Returns
        -------
        numpy.ndarray of float, shape (n_params,)
            Parameter vector. ``nan`` entries are still to be estimated.
        """
        return self._params

    @property
    def free(self) -> NDArray[np.bool_]:
        """Which parameters are estimated when fitting and which are held fixed.

        Returns
        -------
        numpy.ndarray of bool, shape (n_params,)
            ``True`` for a free (estimated) parameter, ``False`` for a fixed one.
            Read-only; use :meth:`fix_params` to change it.
        """
        return self._free

    @property
    def n_params(self) -> int:
        """How many parameters fitting will estimate (fixed ones are not counted).

        Returns
        -------
        int
            Number of ``True`` entries in :attr:`free`.
        """
        return int(self._free.sum())

    @property
    @abstractmethod
    def param_bounds(self) -> list[tuple[float, float]]:
        """Allowed range of each parameter, as ``(lower, upper)`` pairs.

        Returns
        -------
        list of tuple of (float, float), length n_params
            One ``(lower, upper)`` pair per parameter, in the order of
            :attr:`param_names`. An unbounded side is ``inf`` or ``-inf``.

        Notes
        -----
        Open interval ``(lower, upper)`` for each parameter.

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
        """Return a copy of this copula with new parameter values.

        The original copula is left unchanged. The free/fixed mask is kept.

        Parameters
        ----------
        params : array_like of float, shape (n_params,)
            New parameter values, in the order of :attr:`param_names`. A scalar
            is accepted for one-parameter families.

        Returns
        -------
        Copula
            A new copula of the same family and dimension.

        Raises
        ------
        ValueError
            If the number of values is wrong or a value is out of range.

        Examples
        --------
        >>> from rcopula import ClaytonCopula
        >>> ClaytonCopula(2.0).with_params(3.0).params
        array([3.])
        """
        return self._reconstruct(params, self._free)

    def fix_params(self, free: ArrayLike) -> Copula:
        """Return a copy in which some parameters are held fixed during fitting.

        Use this when you know some parameters already (for example the degrees
        of freedom of a t copula) and only want to estimate the rest.

        Parameters
        ----------
        free : array_like of bool, shape (n_params,)
            ``True`` for parameters to estimate, ``False`` for parameters to
            hold at their current value. A scalar is broadcast.

        Returns
        -------
        Copula
            A new copula with the same parameter values and the new mask.

        Notes
        -----
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
        """Natural logarithm of the copula density at each point.

        Prefer this over ``log(pdf(u))`` when summing over many points (as in a
        likelihood): it avoids underflow when densities are tiny.

        Parameters
        ----------
        u : array_like of float, shape (n, d) or (d,)
            Points in the unit cube, one row per point and one column per
            variable. A 1-D array is treated as a single point.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Log density for each row. ``-inf`` for points outside the open unit
            cube, ``nan`` for rows containing ``nan``.

        Raises
        ------
        ValueError
            If any parameter is still ``nan`` (unfitted), or if ``u`` does not
            have ``d`` columns.

        Notes
        -----
        Points outside :math:`[0,1]^d` have zero density, hence ``-inf`` log
        density — matching R, which treats out-of-range coordinates as boundary
        values even when another coordinate is NaN.

        Examples
        --------
        >>> from rcopula import IndependenceCopula
        >>> IndependenceCopula(dim=2).logpdf([[0.3, 0.7], [0.5, 1.5]])
        array([  0., -inf])
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
        """Copula density at each point: how concentrated the probability is there.

        Values above 1 mean points like this are more common than they would be
        if the variables were independent; values below 1 mean less common.

        Parameters
        ----------
        u : array_like of float, shape (n, d) or (d,)
            Points in the unit cube, one row per point and one column per
            variable. A 1-D array is treated as a single point.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Density for each row; ``0`` outside the open unit cube.

        Raises
        ------
        ValueError
            If any parameter is still ``nan`` (unfitted), or if ``u`` does not
            have ``d`` columns.

        See Also
        --------
        logpdf : The same quantity on the log scale, safer for likelihoods.

        Examples
        --------
        >>> from rcopula import IndependenceCopula
        >>> IndependenceCopula(dim=2).pdf([0.3, 0.7])
        array([1.])
        """
        return np.exp(self.logpdf(u))

    def cdf(self, u: ArrayLike) -> NDArray[np.float64]:
        """Probability that every variable is at or below the given point, ``C(u)``.

        For ``d = 2`` and a point ``(a, b)`` this is
        ``P(U_1 <= a and U_2 <= b)``, the copula's cumulative distribution
        function.

        Parameters
        ----------
        u : array_like of float, shape (n, d) or (d,)
            Points, one row per point and one column per variable. Values
            outside ``[0, 1]`` are clipped into it. A 1-D array is treated as a
            single point.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Probabilities in ``[0, 1]``, one per row.

        Raises
        ------
        ValueError
            If any parameter is still ``nan`` (unfitted), or if ``u`` does not
            have ``d`` columns.

        Examples
        --------
        >>> from rcopula import IndependenceCopula
        >>> IndependenceCopula(dim=2).cdf([[0.5, 0.5], [0.2, 1.0]])
        array([0.25, 0.2 ])
        """
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
        """Generate random samples that follow this copula's dependence.

        Each row is one simulated observation; each column is uniformly
        distributed on ``(0, 1)`` on its own, but the columns are dependent in
        the way the copula describes. To get samples on real-world scales,
        transform each column with a quantile function, or use
        :class:`~rcopula.CopulaDistribution`.

        Parameters
        ----------
        size : int, default 1
            Number of observations (rows) to draw. Must be non-negative.
        random_state : int, numpy.random.Generator or None, default None
            Seed or generator. Accepting a ``Generator`` is the modern idiom;
            an ``int`` is promoted via ``np.random.default_rng``. ``None`` gives
            fresh, unreproducible randomness.

        Returns
        -------
        numpy.ndarray of float, shape (size, d)
            Samples with values in ``(0, 1)``.

        Raises
        ------
        ValueError
            If any parameter is still ``nan`` (unfitted), or ``size < 0``.

        Examples
        --------
        >>> from rcopula import ClaytonCopula
        >>> u = ClaytonCopula(2.0, dim=3).rvs(4, random_state=0)
        >>> u.shape
        (4, 3)
        >>> bool(((u > 0) & (u < 1)).all())
        True
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
        """Probability that every variable falls inside a given box.

        Gives ``P(lower_j < U_j <= upper_j for all j)`` — the C-volume of a box.
        Useful for questions like "how likely are both variables to land in
        their bottom 10% at the same time?".

        Parameters
        ----------
        lower : array_like of float, shape (d,)
            Lower corner of the box, one value per variable, in ``[0, 1]``.
        upper : array_like of float, shape (d,)
            Upper corner of the box, elementwise ``>= lower``, in ``[0, 1]``.

        Returns
        -------
        float
            Probability of the box, in ``[0, 1]`` up to rounding error.

        Raises
        ------
        ValueError
            If ``lower`` or ``upper`` does not have length ``d``, if
            ``lower > upper`` anywhere, if ``d > 20`` (too many corners), or if
            the copula is unfitted.

        Notes
        -----
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
        """Strength of dependence as Kendall's tau, a rank correlation in ``[-1, 1]``.

        Kendall's tau is the probability that two random observations are
        ordered the same way in both variables, minus the probability they are
        ordered oppositely. ``0`` means no (rank) association, ``1`` means the
        variables always move together, ``-1`` always oppositely. This is the
        value implied by the copula itself (the population value), not an
        estimate from data.

        Returns
        -------
        float
            Kendall's tau in ``[-1, 1]``. For ``d > 2`` families define it on
            a pair of variables; see the family's docstring.
        """

    @abstractmethod
    def rho(self) -> float:
        """Strength of dependence as Spearman's rho, a rank correlation in ``[-1, 1]``.

        Spearman's rho is the ordinary correlation of the variables' ranks.
        ``0`` means no (rank) association, ``1`` perfect positive, ``-1``
        perfect negative. This is the population value implied by the copula.

        Returns
        -------
        float
            Spearman's rho in ``[-1, 1]``. For ``d > 2`` families define it on
            a pair of variables; see the family's docstring.
        """

    def beta(self) -> float:
        """Strength of dependence as Blomqvist's beta, based on medians.

        Blomqvist's beta compares how often all variables sit on the same side
        of their medians (all below or all above) against what independence
        would give. ``0`` means independence at the centre; positive values
        mean the variables tend to be above or below their medians together.

        Returns
        -------
        float
            Blomqvist's beta. In ``[-1, 1]`` for ``d = 2``.

        Raises
        ------
        ValueError
            If the copula is unfitted (any parameter ``nan``).

        Notes
        -----
        Population Blomqvist's beta, ``2**d * C(1/2, ..., 1/2) - 1`` rescaled.

        For ``d = 2`` this is ``4 * C(1/2, 1/2) - 1``. Blomqvist's beta depends on
        the copula only at the centre point, which makes it cheap but insensitive
        to the tails.

        Examples
        --------
        >>> from rcopula import IndependenceCopula
        >>> IndependenceCopula(dim=2).beta()
        0.0
        """
        d = self._dim
        centre = float(self.cdf(np.full((1, d), 0.5))[0])
        survival = float(self.prob(np.full(d, 0.5), np.ones(d)))
        return (2.0 ** (d - 1) * (centre + survival) - 1.0) / (2.0 ** (d - 1) - 1.0)

    @abstractmethod
    def lambda_(self) -> TailDependence:
        """How strongly the variables tend to be extreme together, in each tail.

        The lower coefficient is about joint extreme lows (e.g. simultaneous
        crashes), the upper one about joint extreme highs. Both are in
        ``[0, 1]``; ``0`` means joint extremes become negligible far enough out
        in that tail. A Gaussian copula has ``0`` in both tails; Clayton has
        lower but not upper tail dependence; Gumbel the reverse.

        Returns
        -------
        TailDependence
            Named tuple ``(lower, upper)`` of floats in ``[0, 1]``.
        """

    # ------------------------------------------------------------------
    # Calibration from a target dependence (R's iTau / iRho)
    # ------------------------------------------------------------------

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> Copula:
        """Create a copula of this family whose Kendall's tau equals a target value.

        Handy when you know (or have estimated) the rank correlation you want
        and need the matching parameter, e.g. ``ClaytonCopula.from_tau(0.5)``.

        Parameters
        ----------
        tau : float
            Target Kendall's tau. Must lie in the range the family can reach
            (many families allow only ``0 <= tau < 1``).
        dim : int, default 2
            Number of variables of the copula to build.
        **kwargs : Any
            Extra family-specific constructor arguments (e.g. ``df`` for a t
            copula), where the family supports them.

        Returns
        -------
        Copula
            A new copula of the calling class.

        Raises
        ------
        NotImplementedError
            If the family does not support calibration from Kendall's tau.
        ValueError
            If ``tau`` is outside the range the family can attain (raised by
            the families that implement this).

        Notes
        -----
        The Pythonic spelling of R's ``iTau``. A classmethod rather than an
        instance method because it *constructs* rather than mutates.

        Examples
        --------
        >>> from rcopula import ClaytonCopula
        >>> c = ClaytonCopula.from_tau(0.5)
        >>> round(float(c.params[0]), 10), round(c.tau(), 10)
        (2.0, 0.5)
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement calibration from Kendall's tau"
        )

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> Copula:
        """Create a copula of this family whose Spearman's rho equals a target value.

        The same idea as :meth:`from_tau`, but matching Spearman's rank
        correlation instead (R's ``iRho``).

        Parameters
        ----------
        rho : float
            Target Spearman's rho, within the range the family can reach.
        dim : int, default 2
            Number of variables of the copula to build.
        **kwargs : Any
            Extra family-specific constructor arguments, where supported.

        Returns
        -------
        Copula
            A new copula of the calling class.

        Raises
        ------
        NotImplementedError
            If the family does not support calibration from Spearman's rho.
        ValueError
            If ``rho`` is outside the attainable range (raised by the families
            that implement this).
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement calibration from Spearman's rho"
        )

    def calibrated(self, measure: str, value: float) -> Copula:
        """Return a copy of this family re-tuned to a target ``tau`` or ``rho``.

        Like :meth:`from_tau` / :meth:`from_rho`, but called on an existing
        copula so its family and dimension are reused.

        Parameters
        ----------
        measure : {"tau", "rho"}
            Which rank correlation to match: Kendall's tau or Spearman's rho.
        value : float
            Target value of that measure.

        Returns
        -------
        Copula
            A new copula of the same family and dimension.

        Raises
        ------
        ValueError
            If ``measure`` is not ``"tau"`` or ``"rho"``, or ``value`` is out of
            the family's range.
        NotImplementedError
            If the family does not support that calibration.

        Notes
        -----
        The instance-level counterpart of :meth:`from_tau` / :meth:`from_rho`,
        and what :func:`~rcopula.fit.fit` calls for ``method="itau"`` and
        ``"irho"``. Families whose construction needs more than
        ``(value, dim)`` -- a rotation has to know *what* it is rotating --
        override this instead of the classmethods.

        Examples
        --------
        >>> from rcopula import ClaytonCopula
        >>> c = ClaytonCopula(1.0, dim=2).calibrated("tau", 0.5)
        >>> round(float(c.params[0]), 10)
        2.0
        """
        if measure not in ("tau", "rho"):
            raise ValueError(f"measure must be 'tau' or 'rho', got {measure!r}")
        factory = type(self).from_tau if measure == "tau" else type(self).from_rho
        return factory(value, dim=self._dim)

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------

    def describe(self) -> str:
        """Short human-readable summary of the family, dimension and parameters.

        In the spirit of R's ``describeCop``. Fixed parameters are marked
        ``(fixed)``.

        Returns
        -------
        str
            One-line description.

        Examples
        --------
        >>> from rcopula import ClaytonCopula
        >>> ClaytonCopula(2.0).describe()
        'Clayton copula, dim 2, theta=2'
        """
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
