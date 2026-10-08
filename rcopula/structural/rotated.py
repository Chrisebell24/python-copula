r"""Rotated and survival copulas.

Every Archimedean family is *asymmetric* in a fixed direction: Clayton binds in
the lower tail, Gumbel in the upper, and neither can be told to do the opposite.
Reflecting one or more coordinates fixes that, and costs nothing -- the reflected
object is still a copula, with the same number of parameters.

Reflecting coordinate :math:`j` replaces :math:`U_j` by :math:`1 - U_j`. Two
cases have names:

* **All** coordinates reflected gives the **survival copula**
  :math:`\hat C`, the copula of :math:`(1-U_1,\dots,1-U_d)`. In two dimensions
  this is the 180-degree rotation. It swaps the tails: survival Clayton has
  *upper* tail dependence, which is the usual way to get an
  upper-tail-dependent alternative to Gumbel with a different shape.
* **One** coordinate reflected, in two dimensions, gives the 90-degree (or
  270-degree) rotation, which turns positive dependence into negative. Clayton
  and Gumbel admit no negative dependence at all; rotating them is the standard
  way to model it, and it is what vine libraries mean by "rotation 90".

The CDF follows from inclusion-exclusion over the reflected coordinates,

.. math::
    C_S(\mathbf u) = \sum_{T \subseteq S} (-1)^{|T|}\, C(\mathbf x^T),
    \qquad
    x^T_j = \begin{cases} u_j & j \notin S\\ 1-u_j & j \in T\\ 1 & j \in S\setminus T,\end{cases}

while the density is simply the base density at the reflected point -- the
Jacobian of a reflection is 1.

.. warning::

   Rotation is why "Clayton for losses" is usually backwards. Aggregate loss
   risk is driven by the **upper** tail, and plain Clayton has none: at
   :math:`\tau = 0.5` the 99% expected shortfall of a two-line loss portfolio
   comes out *below* the Gaussian answer. It is the **survival** Clayton that
   belongs there.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer,
    section 2.6 -- the survival copula and the reflection identities.
Brechmann, E. C. and Schepsmeier, U. (2013). Modeling dependence with C- and
    D-vine copulas: the R package CDVine.
    *Journal of Statistical Software* 52(3), 1-27.
    The 0/90/180/270 rotation convention used by vine libraries.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.base import Copula, TailDependence
from rcopula.core.measures import rho_by_quadrature, tau_by_quadrature
from rcopula.core.other import FrechetLowerCopula

__all__ = ["RotatedCopula", "survival"]

#: Degree-to-flip map for the bivariate rotation convention used by vine
#: libraries. 90 reflects the *second* coordinate, 270 the first; the choice is
#: a convention, and this one matches ``VineCopula``/``pyvinecopulib``.
_DEGREES: dict[int, tuple[bool, bool]] = {
    0: (False, False),
    90: (False, True),
    180: (True, True),
    270: (True, False),
}


class RotatedCopula(Copula):
    """Turn an existing copula around so its strong dependence sits in a different corner.

    Reflecting coordinate ``j`` means replacing each value ``u_j`` by
    ``1 - u_j`` (a small value becomes a large one). Use this when a family has
    the right *shape* but the wrong orientation: for example Clayton ties
    variables together when they are all small (lower tail), and reflecting
    every coordinate gives a copula that ties them together when they are all
    large (upper tail) -- usually what you want for joint losses. In two
    dimensions, reflecting one coordinate turns positive dependence into
    negative, which Clayton and Gumbel cannot otherwise express.

    The result has exactly the base copula's parameters, so it can be fitted,
    evaluated and sampled like any other copula.

    Parameters
    ----------
    base : Copula
        The copula being reflected, with any dimension ``d >= 2``. Its
        parameters may be ``nan`` (to be fitted).
    flip : bool, array_like of bool with shape (d,), or int, default True
        Which coordinates to reflect. A single ``bool`` applies to all of them,
        so ``flip=True`` (the default) is the survival copula. A sequence of
        ``d`` booleans picks coordinates individually. In two dimensions an
        ``int`` in ``{0, 90, 180, 270}`` selects a rotation by that many
        degrees (90 reflects the second coordinate, 270 the first, 180 both).
    free : array_like of bool, shape (n_params,), or None, default None
        Keyword-only. Which parameters are estimated when fitting; ``None``
        keeps the base copula's mask.

    Attributes
    ----------
    base : Copula
        The copula being reflected. Never itself a :class:`RotatedCopula` --
        nested rotations are collapsed on construction.
    flip : numpy.ndarray of bool, shape (d,)
        Read-only mask of reflected coordinates.
    name : str
        ``"Rotated <base name>"``.
    param_names : tuple of str
        The base copula's parameter names.

    Raises
    ------
    ValueError
        If ``flip`` is a sequence whose length is not ``d``, if it is an
        ``int`` but ``d != 2``, or if the ``int`` is not one of
        ``0, 90, 180, 270``.

    Notes
    -----
    Rotations compose: ``RotatedCopula(RotatedCopula(c, S1), S2)`` collapses to a
    single rotation by the symmetric difference of ``S1`` and ``S2``, so the
    class is closed under composition and a double rotation returns to the base.

    That symmetric difference is worth reading carefully. The operation is a
    coordinate **reflection**, which is an involution, so composition follows the
    Klein four-group rather than the cyclic one the degree labels suggest:
    applying 90 twice gives **0**, not 180, and reaching 180 takes a 90 and a
    270. The names match the vine convention and denote the right copulas; only
    their composition differs from what "rotate twice" would imply. (For a
    non-exchangeable base a true geometric rotation would also transpose the
    coordinates; for the exchangeable families this wraps, the two coincide.)

    Parameters are the base copula's, so :func:`~rcopula.fit.fit`,
    :func:`~rcopula.select_copula` and the goodness-of-fit machinery all work
    unchanged.

    Examples
    --------
    The survival Clayton has upper tail dependence where Clayton has lower:

    >>> from rcopula import ClaytonCopula
    >>> from rcopula.structural import RotatedCopula
    >>> base = ClaytonCopula(3.0)
    >>> base.lambda_()
    TailDependence(lower=0.7937005259840998, upper=0.0)
    >>> RotatedCopula(base, True).lambda_()
    TailDependence(lower=0.0, upper=0.7937005259840998)

    A 90-degree rotation turns positive dependence into negative:

    >>> cop = RotatedCopula(ClaytonCopula(3.0), 90)
    >>> float(round(cop.tau(), 12))
    -0.6
    >>> float(round(base.tau(), 12))
    0.6

    Rotating twice returns the original:

    >>> RotatedCopula(RotatedCopula(base, True), True).flip.tolist()
    [False, False]
    """

    name = "Rotated"

    #: The copula being reflected. Never itself a :class:`RotatedCopula` -- the
    #: constructor collapses nested rotations.
    base: Copula

    def __init__(
        self,
        base: Copula,
        flip: ArrayLike | int | bool = True,
        *,
        free: ArrayLike | None = None,
    ) -> None:
        mask = _as_flip(flip, base.dim)

        # Collapse nested rotations so the class is closed under composition and
        # tail-dependence bookkeeping never has to recurse.
        if isinstance(base, RotatedCopula):
            mask = mask ^ base.flip
            base = base.base

        self.base = base
        self._flip = mask
        self._flip.flags.writeable = False
        self.name = f"Rotated {base.name}"
        self.param_names = base.param_names

        super().__init__(base.params, base.dim, free=base.free if free is None else free)

    # -- plumbing ------------------------------------------------------

    @property
    def flip(self) -> NDArray[np.bool_]:
        """Which coordinates are reflected.

        Returns
        -------
        numpy.ndarray of bool, shape (d,)
            Read-only mask; ``True`` where ``u_j`` is replaced by ``1 - u_j``.
        """
        return self._flip

    @property
    def degrees(self) -> int:
        """The rotation expressed in degrees, as vine libraries label it (two dimensions only).

        Returns
        -------
        int
            One of ``0``, ``90``, ``180``, ``270``. 90 means the second
            coordinate is reflected, 270 the first, 180 both.

        Raises
        ------
        ValueError
            If the copula is not bivariate.

        Examples
        --------
        >>> from rcopula import ClaytonCopula
        >>> from rcopula.structural import RotatedCopula
        >>> RotatedCopula(ClaytonCopula(2.0), [True, False]).degrees
        270
        """
        if self.dim != 2:
            raise ValueError(f"the degree convention is bivariate; this copula has dim={self.dim}")
        return next(d for d, f in _DEGREES.items() if np.array_equal(np.array(f), self._flip))

    @property
    def is_survival(self) -> bool:
        """Whether this is the survival copula, i.e. every coordinate is reflected.

        Returns
        -------
        bool
            ``True`` when every coordinate is reflected.
        """
        return bool(self._flip.all())

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Allowed range of each parameter -- the same as the base copula's.

        Returns
        -------
        list of tuple of (float, float)
            One ``(lower, upper)`` pair per parameter.
        """
        return self.base.param_bounds

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> Copula:
        return RotatedCopula(self.base.with_params(params), self._flip, free=free)

    def _validate_params(self) -> None:
        # Delegate: the base already knows its own admissible ranges, and
        # reflection does not change them.
        self.base._validate_params()

    def _reflect(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        out = u.copy()
        out[:, self._flip] = 1.0 - out[:, self._flip]
        return out

    # -- evaluation ----------------------------------------------------

    def _logpdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        # A reflection has |Jacobian| = 1, so the density is just relabelled.
        return self.base._logpdf(self._reflect(u), params)

    def _cdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        idx = np.flatnonzero(self._flip)
        if idx.size == 0:
            return self.base._cdf(u, params)

        total = np.zeros(u.shape[0])
        # Inclusion-exclusion over subsets of the reflected coordinates: each
        # "U_j >= 1 - u_j" event is written as 1 minus its complement.
        for size in range(idx.size + 1):
            for subset in itertools.combinations(idx, size):
                point = np.ones_like(u)
                point[:, ~self._flip] = u[:, ~self._flip]
                for j in subset:
                    point[:, j] = 1.0 - u[:, j]
                total += (-1.0) ** size * self._base_cdf(point, params)
        return total

    def _base_cdf(
        self, point: NDArray[np.float64], params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """``base._cdf`` with the zero-coordinate guard :meth:`Copula.cdf` applies.

        Reflection turns ``u_j = 1`` into ``0``, and ``_cdf`` implementations are
        entitled to assume they never see that -- the public :meth:`Copula.cdf`
        short-circuits it before they are called. Skipping those rows here keeps
        that contract intact; without it Clayton's log-space CDF meets
        ``log(0)`` and returns ``nan``, which showed up as the *rotated* copula
        losing its uniform margins.
        """
        out = np.zeros(point.shape[0])
        inside = ~np.any(point <= 0.0, axis=1)
        if inside.any():
            out[inside] = self.base._cdf(point[inside], params)
        return out

    def _rvs(
        self, size: int, params: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return self._reflect(self.base.with_params(params)._rvs(size, params, rng))

    # -- dependence ----------------------------------------------------

    def _sign(self) -> float:
        """Reflecting an odd number of coordinates reverses concordance."""
        return -1.0 if int(self._flip.sum()) % 2 else 1.0

    def tau(self) -> float:
        """Return Kendall's tau, a rank correlation between -1 and 1, after reflection.

        Exact in ``d = 2``; the base's value when ``d > 2`` and all or none of
        the coordinates are reflected.

        In two dimensions concordance simply flips sign with each reflection.
        Beyond two dimensions the scalar tau of an exchangeable family survives
        only when *all* or *no* coordinates are reflected; a partial reflection
        makes the pairwise taus differ, so there is no single number to report.

        Returns
        -------
        float
            Kendall's tau of the rotated copula.

        Raises
        ------
        NotImplementedError
            For a partial reflection with ``d > 2``.
        """
        if self.dim == 2:
            return self._sign() * self.base.tau()
        if self._flip.all() or not self._flip.any():
            return self.base.tau()
        raise NotImplementedError(
            f"a partial reflection of a {self.dim}-dimensional copula has different "
            "tau for different pairs; there is no scalar value to return"
        )

    def rho(self) -> float:
        """Return Spearman's rho, a rank correlation between -1 and 1, after reflection.

        Same conventions as :meth:`tau`: in two dimensions the sign flips with
        each reflected coordinate.

        Returns
        -------
        float
            Spearman's rho of the rotated copula.

        Raises
        ------
        NotImplementedError
            For a partial reflection with ``d > 2``.
        """
        if self.dim == 2:
            return self._sign() * self.base.rho()
        if self._flip.all() or not self._flip.any():
            return self.base.rho()
        raise NotImplementedError(
            f"a partial reflection of a {self.dim}-dimensional copula has different "
            "rho for different pairs; there is no scalar value to return"
        )

    def lambda_(self) -> TailDependence:
        r"""Return how strongly the variables move together in extremes, after reflection.

        The lower (upper) tail-dependence coefficient is the limiting chance
        that one variable is extremely small (large) given that another is.

        Reflecting **every** coordinate swaps the two coefficients: the survival
        copula's lower tail is the base's upper tail. Reflecting **none** leaves
        them alone.

        A partial reflection is different in kind. In two dimensions the
        rotated copula's lower-tail coefficient is the base's *cross*
        coefficient :math:`\lim_{u\to0} P(U_1 > 1-u,\ U_2 \le u)/u`, which
        measures mass at an off-diagonal corner. That is zero for any positively
        quadrant dependent copula, hence for every parametric family in this
        package. The one exception is the Frechet lower bound :math:`W`, whose
        entire mass sits on that anti-diagonal -- and reflecting it gives
        :math:`M`, which is comonotone -- so it is handled explicitly.

        Returns
        -------
        TailDependence
            Named tuple ``(lower, upper)`` of floats in ``[0, 1]``.
        """
        lower, upper = self.base.lambda_()
        if not self._flip.any():
            return TailDependence(lower=lower, upper=upper)
        if self._flip.all():
            return TailDependence(lower=upper, upper=lower)
        cross = 1.0 if isinstance(self.base, FrechetLowerCopula) else 0.0
        return TailDependence(lower=cross, upper=cross)

    # -- calibration ---------------------------------------------------

    def calibrated(self, measure: str, value: float) -> Copula:
        """Return a new rotated copula whose tau or rho equals a target value.

        The wrapped family is calibrated so that the *rotated* copula hits the
        target (the sign is flipped for you where the rotation reverses
        concordance). The rotation itself is kept.

        This is why the instance-level hook exists: a classmethod cannot know
        which family is being rotated. It is what makes ``method="itau"`` and
        ``"irho"`` work on rotated copulas, and what supplies the starting value
        for ``"mpl"``.

        Parameters
        ----------
        measure : {"tau", "rho"}
            Which rank correlation to match.
        value : float
            Target value of that measure for the rotated copula.

        Returns
        -------
        Copula
            A :class:`RotatedCopula` with the same ``flip`` and new parameters.

        Raises
        ------
        ValueError
            If ``measure`` is not ``"tau"`` or ``"rho"``, or if the base family
            cannot reach the (sign-adjusted) value.

        Examples
        --------
        >>> from rcopula import ClaytonCopula
        >>> from rcopula.structural import RotatedCopula
        >>> template = RotatedCopula(ClaytonCopula(1.0), 90)
        >>> float(round(template.calibrated("tau", -0.4).tau(), 12))
        -0.4
        """
        if measure not in ("tau", "rho"):
            raise ValueError(f"measure must be 'tau' or 'rho', got {measure!r}")
        sign = self._sign() if self.dim == 2 else 1.0
        return RotatedCopula(self.base.calibrated(measure, sign * value), self._flip)

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> Copula:
        """Build a rotated copula with a given Kendall's tau.

        The base family is calibrated so that the *rotated* copula has this tau;
        for a 90- or 270-degree rotation that means a negative tau gives a valid
        positive-dependence base.

        Parameters
        ----------
        tau : float
            Target Kendall's tau of the rotated copula, in ``(-1, 1)``.
        dim : int, default 2
            Number of variables.
        **kwargs
            ``base`` (required): the copula *class* to rotate, e.g.
            ``ClaytonCopula``. ``flip`` (bool, array_like of bool or int,
            default ``True``): as in the constructor. Anything else is passed
            to ``base.from_tau``.

        Returns
        -------
        Copula
            A :class:`RotatedCopula`.

        Raises
        ------
        TypeError
            If ``base`` is not given.
        ValueError
            If ``flip`` is invalid or the base family cannot reach the value.

        Examples
        --------
        >>> from rcopula import ClaytonCopula
        >>> from rcopula.structural import RotatedCopula
        >>> cop = RotatedCopula.from_tau(-0.5, base=ClaytonCopula, flip=90)
        >>> float(round(cop.tau(), 12))
        -0.5
        """
        base_cls = kwargs.pop("base", None)
        flip = kwargs.pop("flip", True)
        if base_cls is None:
            raise TypeError("RotatedCopula.from_tau requires a base= copula class")
        mask = _as_flip(flip, dim)
        sign = -1.0 if int(mask.sum()) % 2 and dim == 2 else 1.0
        return cls(base_cls.from_tau(sign * tau, dim=dim, **kwargs), mask)

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> Copula:
        """Build a rotated copula with a given Spearman's rho.

        Exactly like :meth:`from_tau`, but matching Spearman's rho.

        Parameters
        ----------
        rho : float
            Target Spearman's rho of the rotated copula, in ``(-1, 1)``.
        dim : int, default 2
            Number of variables.
        **kwargs
            ``base`` (required): the copula *class* to rotate. ``flip``
            (default ``True``): as in the constructor. Anything else is passed
            to ``base.from_rho``.

        Returns
        -------
        Copula
            A :class:`RotatedCopula`.

        Raises
        ------
        TypeError
            If ``base`` is not given.
        ValueError
            If ``flip`` is invalid or the base family cannot reach the value.
        """
        base_cls = kwargs.pop("base", None)
        flip = kwargs.pop("flip", True)
        if base_cls is None:
            raise TypeError("RotatedCopula.from_rho requires a base= copula class")
        mask = _as_flip(flip, dim)
        sign = -1.0 if int(mask.sum()) % 2 and dim == 2 else 1.0
        return cls(base_cls.from_rho(sign * rho, dim=dim, **kwargs), mask)

    # -- presentation --------------------------------------------------

    def describe(self) -> str:
        """Return a one-line, human-readable description of the rotated copula.

        Returns
        -------
        str
            For example ``"90-degree rotated Clayton ..."`` in two dimensions,
            or ``"survival ..."`` / ``"reflected on [0, 2] ..."`` otherwise,
            followed by the base copula's own description.
        """
        if self.dim == 2:
            label = f"{self.degrees}-degree rotated"
        elif self.is_survival:
            label = "survival"
        else:
            label = f"reflected on {np.flatnonzero(self._flip).tolist()}"
        return f"{label} {self.base.describe()}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RotatedCopula):
            return NotImplemented
        return self.base == other.base and np.array_equal(self._flip, other._flip)

    def __hash__(self) -> int:
        return hash(("RotatedCopula", self.base, self._flip.tobytes()))


def _as_flip(flip: ArrayLike | int | bool, dim: int) -> NDArray[np.bool_]:
    """Normalise the ``flip`` argument to a boolean mask of length ``dim``."""
    if isinstance(flip, bool | np.bool_):
        return np.full(dim, bool(flip))
    if isinstance(flip, int | np.integer):
        if dim != 2:
            raise ValueError(
                f"the degree convention is bivariate; use a boolean mask for dim={dim}"
            )
        if int(flip) not in _DEGREES:
            raise ValueError(f"rotation must be one of {sorted(_DEGREES)}, got {flip}")
        return np.array(_DEGREES[int(flip)])
    mask = np.asarray(flip, dtype=bool).ravel()
    if mask.size != dim:
        raise ValueError(f"flip has length {mask.size} but the copula has dim={dim}")
    return mask


def survival(copula: Copula) -> RotatedCopula:
    r"""Flip a copula so its lower-tail behaviour becomes upper-tail behaviour.

    Returns the survival copula :math:`\hat C`: the copula with every
    coordinate reflected (``u -> 1 - u``), and lower and upper tails swapped.
    It is shorthand for ``RotatedCopula(copula, True)``.

    :math:`\hat C(\mathbf u)` is the copula of :math:`(1-U_1,\dots,1-U_d)`, and
    it is what you want whenever the risk lives in the **upper** tail: joint
    large losses, joint large claims, joint large flows.

    Parameters
    ----------
    copula : Copula
        Any copula, of any dimension.

    Returns
    -------
    RotatedCopula
        The survival copula, with the same parameters as ``copula``. Applying
        :func:`survival` twice gives back the original copula's behaviour.

    Examples
    --------
    Survival Clayton is the upper-tail-dependent version of Clayton. On a
    portfolio of two exponential losses at identical Kendall's tau, plain
    Clayton comes out *below* the Gaussian answer and survival Clayton above --
    a 29% difference in expected shortfall between two copulas with the same
    rank correlation:

    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, GaussianCopula
    >>> from rcopula.structural import survival
    >>> from rcopula.risk import expected_shortfall
    >>> def es(cop):
    ...     losses = -np.log(1 - cop.rvs(200_000, random_state=0)).sum(axis=1)
    ...     return expected_shortfall(losses, 0.99)
    >>> plain = ClaytonCopula.from_tau(0.5)
    >>> bool(es(plain) < es(GaussianCopula.from_tau(0.5)) < es(survival(plain)))
    True

    Kendall's tau is unchanged, so no rank-correlation-based summary would show
    the difference:

    >>> bool(abs(survival(plain).tau() - plain.tau()) < 1e-12)
    True
    """
    return RotatedCopula(copula, True)


def _quadrature_check(copula: Copula) -> tuple[float, float]:  # pragma: no cover
    """Independent tau/rho for a rotated copula, used by the tests."""
    return tau_by_quadrature(copula), rho_by_quadrature(copula)
