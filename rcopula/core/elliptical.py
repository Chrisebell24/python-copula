r"""Elliptical copulas: Gaussian and Student-t.

An elliptical copula is what you get by taking an elliptical distribution and
throwing its margins away:

.. math::

    C(\mathbf{u}) = F_{\Sigma}\bigl(F^{-1}(u_1), \dots, F^{-1}(u_d)\bigr)

with :math:`F_\Sigma` the joint and :math:`F` the common margin. The two that
matter in practice are the Gaussian copula (no tail dependence at all) and the
Student-t copula (symmetric tail dependence in *both* tails, controlled by the
degrees of freedom). The gap between them is the single most consequential
modelling choice in quantitative risk: they can be calibrated to the same
Kendall's tau and still disagree by an order of magnitude about the probability
of a joint crash.

Correlation structures follow R's ``dispstr``:

========= ================== =================================================
dispstr   Free parameters    Structure
========= ================== =================================================
``ex``    1                  Exchangeable: every pair shares one rho
``ar1``   1                  Autoregressive: ``Sigma[i,j] = rho ** |i-j|``
``toep``  ``d - 1``          Toeplitz: constant along each diagonal
``un``    ``d(d-1)/2``       Unstructured: every pair free
========= ================== =================================================

References
----------
McNeil, A. J., Frey, R. and Embrechts, P. (2015). *Quantitative Risk
    Management*, 2nd ed. Princeton, Chapter 7, for elliptical copulas, the
    ``tau = (2/pi) arcsin(rho)`` identity and the t tail-dependence formula.
Demarta, S. and McNeil, A. J. (2005). The t copula and related copulas.
    *International Statistical Review* 73(1), 111-129.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, toeplitz
from scipy.optimize import brentq
from scipy.special import gammaln, ndtr, ndtri
from scipy.stats import t as student_t

from rcopula.core.base import Copula, TailDependence
from rcopula.special.mvtnorm import mvn_cdf, mvt_cdf

__all__ = ["EllipticalCopula", "GaussianCopula", "P2p", "StudentCopula", "p2P"]

DISPSTRS = ("ex", "ar1", "toep", "un")


def _lower_indices(dim: int) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Lower-triangle indices in R's ``lower.tri`` order: column by column.

    ``np.tril_indices`` runs row by row instead. The two orders agree up to
    ``dim=3`` and differ from ``dim=4``, which is why the difference can hide.
    """
    cols, rows = np.triu_indices(dim, 1)
    return rows, cols


def p2P(param: ArrayLike, dim: int) -> NDArray[np.float64]:
    """Turn a flat list of pairwise correlations into a full correlation matrix.

    Use this to go from the parameter vector of an unstructured
    (``dispstr="un"``) Gaussian or t copula to the ``d x d`` matrix it
    describes. It is R's ``p2P``; :func:`P2p` is the inverse.

    Parameters
    ----------
    param : array_like of float, shape (d * (d - 1) / 2,)
        The below-diagonal entries, one per pair of variables. Any shape is
        accepted and flattened; only the total count matters. Values are copied
        in as given -- nothing checks that they lie in ``[-1, 1]`` or that the
        result is positive definite.
    dim : int
        Size ``d`` of the matrix to build.

    Returns
    -------
    numpy.ndarray of float, shape (dim, dim)
        Symmetric matrix with ones on the diagonal.

    Raises
    ------
    ValueError
        If ``param`` does not hold exactly ``dim * (dim - 1) / 2`` values.

    Notes
    -----
    Entries fill the lower triangle column by column, as R does, so for
    ``dim=4`` the parameter vector is
    ``(rho_12, rho_13, rho_14, rho_23, rho_24, rho_34)``. This is *not* the
    row-by-row order of ``numpy.tril_indices``; the two agree only up to
    ``dim=3``.

    Examples
    --------
    >>> from rcopula.core.elliptical import p2P
    >>> p2P([0.6, 0.3, 0.2], 3)
    array([[1. , 0.6, 0.3],
           [0.6, 1. , 0.2],
           [0.3, 0.2, 1. ]])
    >>> p2P([12, 13, 14, 23, 24, 34], 4)[[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]]
    array([12., 13., 14., 23., 24., 34.])
    """
    param = np.asarray(param, dtype=np.float64).ravel()
    expected = dim * (dim - 1) // 2
    if param.size != expected:
        raise ValueError(f"need {expected} parameters for dim={dim}, got {param.size}")
    out = np.eye(dim)
    idx = _lower_indices(dim)
    out[idx] = param
    out.T[idx] = param
    return out


def P2p(matrix: ArrayLike) -> NDArray[np.float64]:
    """Flatten a correlation matrix into the list of its pairwise correlations.

    The inverse of :func:`p2P`: it reads the below-diagonal entries of a
    ``d x d`` matrix, column by column (R's order), into a vector. It is R's
    ``P2p``.

    Parameters
    ----------
    matrix : array_like of float, shape (d, d)
        A square matrix, normally a correlation matrix. Only the strictly lower
        triangle is read; nothing checks symmetry.

    Returns
    -------
    numpy.ndarray of float, shape (d * (d - 1) / 2,)
        ``(m[1, 0], m[2, 0], ..., m[d-1, 0], m[2, 1], ...)``.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.core.elliptical import P2p, p2P
    >>> P2p(p2P([0.6, 0.3, 0.2], 3))
    array([0.6, 0.3, 0.2])
    """
    m = np.asarray(matrix, dtype=np.float64)
    return m[_lower_indices(m.shape[0])]


def _n_corr_params(dispstr: str, dim: int) -> int:
    if dispstr in ("ex", "ar1"):
        return 1
    if dispstr == "toep":
        return dim - 1
    if dispstr == "un":
        return dim * (dim - 1) // 2
    raise ValueError(f"dispstr must be one of {DISPSTRS}, got {dispstr!r}")


def _build_sigma(rho: NDArray[np.float64], dispstr: str, dim: int) -> NDArray[np.float64]:
    """Assemble the correlation matrix implied by a dispersion structure."""
    if dispstr == "ex":
        out = np.full((dim, dim), float(rho[0]))
        np.fill_diagonal(out, 1.0)
        return out
    if dispstr == "ar1":
        return toeplitz(float(rho[0]) ** np.arange(dim))
    if dispstr == "toep":
        return toeplitz(np.concatenate([[1.0], rho]))
    return p2P(rho, dim)


def _equicorrelation(value: float, dim: int, kwargs: dict[str, Any]) -> NDArray[np.float64]:
    """Correlation parameters giving *every* pair the correlation ``value``.

    Used by ``from_tau`` / ``from_rho``: a single target pins down a single
    correlation, so for the multi-parameter structures (``"toep"`` with
    ``d - 1`` values, ``"un"`` with ``d (d - 1) / 2``) it is repeated --
    the exchangeable matrix, expressed in that structure. ``"ex"`` and
    ``"ar1"`` take the one value as it is (for ``"ar1"`` that is the lag-1
    correlation; pairs further apart get its powers).
    """
    dispstr = str(kwargs.get("dispstr", "ex"))
    return np.full(_n_corr_params(dispstr, int(dim)), float(value))


class EllipticalCopula(Copula):
    """Common base for copulas built from a correlation matrix (Gaussian and t).

    You normally use :class:`GaussianCopula` or :class:`StudentCopula`
    directly; this class holds what they share -- the correlation structure,
    the dependence measures and calibration from a target tau or rho. It is
    not meant to be instantiated itself.

    Parameters
    ----------
    params : float or array_like of float, default nan
        The correlation parameter(s), each in ``[-1, 1]``. How many are
        needed depends on ``dispstr`` (see below). ``nan`` -- the default --
        means "to be estimated", so ``GaussianCopula()`` is a family ready for
        :func:`~rcopula.fit.fit`; a single ``nan`` expands to as many as
        ``dispstr`` requires.
    dim : int, default 2
        Number of variables ``d``; at least 2.
    dispstr : {"ex", "ar1", "toep", "un"}, default "ex"
        How the correlation matrix is built from ``params`` (R's
        ``dispstr``):

        * ``"ex"`` (exchangeable): one value shared by every pair. Must be at
          least ``-1 / (d - 1)``.
        * ``"ar1"`` (autoregressive): one value ``r``; pair ``(i, j)`` gets
          ``r ** |i - j|``.
        * ``"toep"`` (Toeplitz): ``d - 1`` values, one per distance
          ``|i - j|``.
        * ``"un"`` (unstructured): ``d * (d - 1) / 2`` values, one per pair,
          in the order used by :func:`p2P`.
    free : array_like of bool or None, default None
        Keyword-only. Which parameters :func:`~rcopula.fit.fit` may change;
        ``False`` holds that parameter at its given value. ``None`` frees all.

    Attributes
    ----------
    dispstr : str
        The correlation structure, as passed.
    param_names : tuple of str
        ``("rho",)`` for ``"ex"``/``"ar1"``, ``("rho.1", ..., "rho.{d-1}")``
        for ``"toep"``, and ``("rho.21", "rho.31", ...)`` for ``"un"``.

    Raises
    ------
    ValueError
        If ``dispstr`` is not one of the four codes, ``params`` has the wrong
        length or is out of range, or the implied correlation matrix is not
        positive definite.
    """

    #: Number of parameters beyond the correlation block (t adds ``df``).
    _n_extra: int = 0

    def __init__(
        self,
        params: ArrayLike = np.nan,
        dim: int = 2,
        dispstr: str = "ex",
        *,
        free: ArrayLike | None = None,
    ) -> None:
        # Defaulting to NaN means `GaussianCopula()` reads as "this family, to
        # be estimated", matching `ClaytonCopula()` and R's `normalCopula()`.
        if dispstr not in DISPSTRS:
            raise ValueError(f"dispstr must be one of {DISPSTRS}, got {dispstr!r}")
        self.dispstr = dispstr
        self._n_corr = _n_corr_params(dispstr, int(dim))
        self.param_names = self._make_param_names(dispstr, int(dim))

        given = np.atleast_1d(np.asarray(params, dtype=np.float64))
        if given.size == 1 and np.isnan(given[0]) and self._n_corr > 1:
            given = np.full(self._n_corr, np.nan)
        super().__init__(given, dim, free=free)

    def _make_param_names(self, dispstr: str, dim: int) -> tuple[str, ...]:
        n = _n_corr_params(dispstr, dim)
        if dispstr in ("ex", "ar1"):
            names: tuple[str, ...] = ("rho",)
        elif dispstr == "toep":
            names = tuple(f"rho.{i + 1}" for i in range(n))
        else:
            i, j = _lower_indices(dim)
            names = tuple(f"rho.{b + 1}{a + 1}" for a, b in zip(i, j, strict=True))
        return names

    # -- correlation ---------------------------------------------------

    @property
    def rho_params(self) -> NDArray[np.float64]:
        """Just the correlation parameters, without the t copula's ``df``.

        Returns
        -------
        numpy.ndarray of float, shape (n_corr,)
            The first ``n_corr`` entries of :attr:`params`, where ``n_corr``
            depends on :attr:`dispstr` (1 for ``"ex"``/``"ar1"``, ``d - 1`` for
            ``"toep"``, ``d * (d - 1) / 2`` for ``"un"``). Read-only view.
        """
        return self._params[: self._n_corr]

    def sigma(self) -> NDArray[np.float64]:
        """Return the full correlation matrix that the parameters describe.

        Expands the (possibly structured) correlation parameters into the
        ``d x d`` matrix -- R's ``getSigma``. Useful for inspecting a fitted
        model or passing the matrix to other code.

        Returns
        -------
        numpy.ndarray of float, shape (d, d)
            Symmetric correlation matrix with ones on the diagonal. Contains
            ``nan`` if the parameters are still unspecified.

        Examples
        --------
        >>> from rcopula import GaussianCopula
        >>> GaussianCopula(0.5, dim=3).sigma()
        array([[1. , 0.5, 0.5],
               [0.5, 1. , 0.5],
               [0.5, 0.5, 1. ]])
        >>> GaussianCopula(0.5, dim=3, dispstr="ar1").sigma()
        array([[1.  , 0.5 , 0.25],
               [0.5 , 1.  , 0.5 ],
               [0.25, 0.5 , 1.  ]])
        """
        return _build_sigma(self.rho_params, self.dispstr, self._dim)

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Allowed range ``(lower, upper)`` for each correlation parameter.

        Returns
        -------
        list of tuple of (float, float), length n_corr
            ``(-1 / (d - 1), 1)`` for the exchangeable structure and
            ``(-1, 1)`` otherwise. Positive definiteness of the whole matrix is
            checked separately when the copula is built.
        """
        # The exchangeable structure needs rho >= -1/(d-1) to stay positive
        # definite; the others are only box-bounded here, with definiteness
        # enforced separately.
        lower = -1.0 / (self._dim - 1) if self.dispstr == "ex" else -1.0
        return [(lower, 1.0)] * self._n_corr

    def _validate_params(self) -> None:
        super()._validate_params()
        if np.isnan(self._params).any():
            return
        sigma = self.sigma()
        eig = np.linalg.eigvalsh(sigma)
        if eig.min() <= 0:
            raise ValueError(
                f"{self.name} copula: the implied correlation matrix is not "
                f"positive definite (smallest eigenvalue {eig.min():.3g}). "
                "Check the parameters, or project with nearest-correlation."
            )

    # -- numerical core ------------------------------------------------

    def _quantile(self, u: NDArray[np.float64], params: NDArray[np.float64]):
        raise NotImplementedError

    def _rvs_latent(
        self, size: int, params: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        raise NotImplementedError

    def _rvs(self, size, params, rng):
        x = self._rvs_latent(size, params, rng)
        return self._marginal_cdf(x, params)

    def _marginal_cdf(self, x, params):
        raise NotImplementedError

    # -- dependence measures -------------------------------------------

    def tau(self) -> Any:
        r"""Kendall's tau: a rank correlation between -1 and 1 implied by the copula.

        Kendall's tau is the probability that two random draws are ordered the
        same way in both variables minus the probability they are ordered
        oppositely. Here it follows exactly from the correlation,
        :math:`\tau = (2/\pi)\arcsin(\rho)`.

        Returns
        -------
        float or numpy.ndarray of float, shape (d * (d - 1) / 2,)
            A float when ``dim=2`` or ``dispstr="ex"`` (every pair shares one
            value); otherwise one value per pair, in :func:`P2p` order
            (matching R).

        Raises
        ------
        ValueError
            If the parameters are still unspecified (``nan``).

        Notes
        -----
        The identity holds for
        *every* elliptical copula, Gaussian and t alike — which is exactly why
        tau alone cannot distinguish them.
        """
        self._require_specified()
        vals = 2.0 / np.pi * np.arcsin(P2p(self.sigma()))
        return (
            float(vals[0])
            if self._n_corr == 1 and self._dim == 2
            else (float(vals[0]) if self.dispstr == "ex" else vals)
        )

    def rho(self) -> Any:
        r"""Spearman's rho: the correlation of the variables' ranks implied by the copula.

        Spearman's rho is the ordinary correlation of the uniform-scale
        variables, between -1 and 1. For the Gaussian copula it is
        :math:`(6/\pi)\arcsin(\rho/2)`; :class:`StudentCopula` overrides this
        because the formula does not hold for the t copula.

        Returns
        -------
        float or numpy.ndarray of float, shape (d * (d - 1) / 2,)
            A float when ``dim=2`` or ``dispstr="ex"``; otherwise one value per
            pair, in :func:`P2p` order.

        Raises
        ------
        ValueError
            If the parameters are still unspecified (``nan``).
        """
        self._require_specified()
        vals = 6.0 / np.pi * np.arcsin(P2p(self.sigma()) / 2.0)
        return float(vals[0]) if self.dispstr == "ex" or self._dim == 2 else vals

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> EllipticalCopula:
        r"""Build a copula whose Kendall's tau equals the value you ask for.

        Handy for setting the strength of dependence on an interpretable rank
        scale rather than as a raw correlation. Uses the exact inversion
        :math:`\rho = \sin(\pi\tau/2)` (R's ``iTau``).

        Parameters
        ----------
        tau : float
            Target Kendall's tau, strictly between -1 and 1.
        dim : int, default 2
            Number of variables.
        **kwargs
            Passed to the constructor, e.g. ``dispstr`` or (for the t copula)
            ``df``. One target fixes one correlation, so with ``"toep"`` or
            ``"un"`` every pair gets that same correlation (the exchangeable
            matrix written in that structure, with the right number of
            parameters); with ``"ar1"`` it is the lag-1 correlation.

        Returns
        -------
        EllipticalCopula
            A new copula of the calling class.

        Raises
        ------
        ValueError
            If ``tau`` is not in ``(-1, 1)``, or the resulting correlation is
            not admissible for ``dim`` (e.g. too negative for an equicorrelated
            matrix, which needs at least ``-1 / (d - 1)``).

        Examples
        --------
        >>> from rcopula import GaussianCopula
        >>> c = GaussianCopula.from_tau(0.5, dim=3, dispstr="un")
        >>> c.rho_params.round(6).tolist()
        [0.707107, 0.707107, 0.707107]
        """
        if not -1.0 < tau < 1.0:
            raise ValueError(f"tau must lie in (-1, 1), got {tau}")
        return cls(_equicorrelation(np.sin(np.pi * tau / 2.0), dim, kwargs), dim, **kwargs)

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> EllipticalCopula:
        r"""Build a copula whose Spearman's rho equals the value you ask for.

        Uses the Gaussian-copula inversion
        :math:`\rho_P = 2\sin(\pi\rho_S/6)` (R's ``iRho``).
        :class:`StudentCopula` overrides this with a numerical inversion,
        because the formula is not valid for the t copula.

        Parameters
        ----------
        rho : float
            Target Spearman's rho, strictly between -1 and 1.
        dim : int, default 2
            Number of variables.
        **kwargs
            Passed to the constructor, e.g. ``dispstr``. As for
            :meth:`from_tau`, ``"toep"`` and ``"un"`` get the same correlation
            for every pair.

        Returns
        -------
        EllipticalCopula
            A new copula of the calling class.

        Raises
        ------
        ValueError
            If ``rho`` is not in ``(-1, 1)``, or the resulting correlation is
            not admissible for ``dim``.
        """
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {rho}")
        value = 2.0 * np.sin(np.pi * rho / 6.0)
        return cls(_equicorrelation(value, dim, kwargs), dim, **kwargs)


class GaussianCopula(EllipticalCopula):
    r"""The Gaussian (normal) copula: the dependence structure of a multivariate normal.

    It links variables through a correlation matrix, exactly as a multivariate
    normal does, but lets each variable keep whatever marginal distribution
    you choose. It is the standard default and is easy to fit and interpret,
    but it assumes extreme values in different variables are essentially
    unrelated (see below); use :class:`StudentCopula` when joint extremes
    matter.

    Parameters
    ----------
    params : float or array_like of float, default nan
        Correlation parameter(s) in ``[-1, 1]``; how many depends on
        ``dispstr``. ``nan`` means "to be estimated".
    dim : int, default 2
        Number of variables ``d``; at least 2.
    dispstr : {"ex", "ar1", "toep", "un"}, default "ex"
        Correlation structure: exchangeable (1 parameter), autoregressive
        (1), Toeplitz (``d - 1``) or unstructured (``d * (d - 1) / 2``). See
        :class:`EllipticalCopula`.
    free : array_like of bool or None, default None
        Keyword-only mask of parameters :func:`~rcopula.fit.fit` may change.

    Attributes
    ----------
    dispstr : str
        The correlation structure.
    params : numpy.ndarray of float
        The correlation parameters (read-only).

    Raises
    ------
    ValueError
        For an unknown ``dispstr``, the wrong number of parameters, values out
        of range, or a correlation matrix that is not positive definite.

    Notes
    -----
    :math:`C(\mathbf{u}) = \Phi_\Sigma(\Phi^{-1}(u_1), \dots, \Phi^{-1}(u_d))`.

    **No tail dependence in either tail**, for any correlation short of 1. That
    is its defining weakness: it says joint extremes are asymptotically
    independent, which is empirically false for financial returns and is the
    reason its use for CDO pricing aged so badly.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import GaussianCopula
    >>> g = GaussianCopula(0.5, dim=2)
    >>> float(round(g.tau(), 12))
    0.333333333333
    >>> g.lambda_()
    TailDependence(lower=0.0, upper=0.0)
    >>> float(round(g.cdf([[0.5, 0.5]])[0], 12))
    0.333333333333

    Calibration round-trips:

    >>> float(round(GaussianCopula.from_tau(1 / 3).params[0], 12))
    0.5
    """

    name = "Gaussian"

    def _quantile(self, u, params):
        return ndtri(u)

    def _marginal_cdf(self, x, params):
        return ndtr(x)

    def _logpdf(self, u, params):
        x = ndtri(u)
        sigma = _build_sigma(params[: self._n_corr], self.dispstr, self._dim)
        chol = cho_factor(sigma, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
        quad = np.einsum("ij,ij->i", x, cho_solve(chol, x.T).T)
        # The (2 pi) factors of the joint and the margins cancel exactly.
        return -0.5 * log_det - 0.5 * quad + 0.5 * np.einsum("ij,ij->i", x, x)

    def _cdf(self, u, params):
        sigma = _build_sigma(params[: self._n_corr], self.dispstr, self._dim)
        return mvn_cdf(ndtri(u), sigma)

    def _rvs_latent(self, size, params, rng):
        sigma = _build_sigma(params[: self._n_corr], self.dispstr, self._dim)
        chol = np.linalg.cholesky(sigma)
        return rng.standard_normal((size, self._dim)) @ chol.T

    def _reconstruct(self, params, free):
        return GaussianCopula(params, self._dim, self.dispstr, free=free)

    def lambda_(self) -> TailDependence:
        """Tail dependence: always zero in both tails for the Gaussian copula.

        Tail dependence is the chance that one variable is extreme given that
        another is equally extreme, in the limit. The Gaussian copula has none
        for any correlation below 1 -- its defining limitation.

        Returns
        -------
        TailDependence
            ``TailDependence(lower=0.0, upper=0.0)``.

        Raises
        ------
        ValueError
            If the parameters are still unspecified (``nan``).
        """
        self._require_specified()
        return TailDependence(lower=0.0, upper=0.0)


class StudentCopula(EllipticalCopula):
    r"""The Student-t copula: like the Gaussian, but extremes tend to happen together.

    It uses a correlation matrix like :class:`GaussianCopula` plus a
    degrees-of-freedom parameter ``df`` that controls how often variables hit
    extremes at the same time: small ``df`` means strong joint extremes, large
    ``df`` approaches the Gaussian. It is the usual choice for financial
    returns, where crashes are shared.

    Parameters
    ----------
    params : float or array_like of float, default nan
        Correlation parameter(s) in ``[-1, 1]``, as for
        :class:`GaussianCopula`. May optionally include ``df`` as one extra
        final element, in which case the ``df`` keyword is ignored. ``nan``
        means "to be estimated".
    dim : int, default 2
        Number of variables ``d``; at least 2.
    dispstr : {"ex", "ar1", "toep", "un"}, default "ex"
        Correlation structure; see :class:`EllipticalCopula`.
    df : float, default 4.0
        Keyword-only. Degrees of freedom, any positive real (at least 0.01);
        need not be an integer.
    df_fixed : bool, default False
        Keyword-only. If ``True``, :func:`~rcopula.fit.fit` keeps ``df`` at its
        given value and estimates only the correlations. Ignored when ``free``
        is given.
    free : array_like of bool or None, default None
        Keyword-only mask of parameters (correlations then ``df``) that
        :func:`~rcopula.fit.fit` may change.

    Attributes
    ----------
    df : float
        Degrees of freedom.
    df_fixed : bool
        Whether ``df`` was requested fixed.
    dispstr : str
        The correlation structure.
    params : numpy.ndarray of float
        Correlation parameters followed by ``df`` (read-only).

    Raises
    ------
    ValueError
        For an unknown ``dispstr``, the wrong number of parameters, values out
        of range, or a correlation matrix that is not positive definite.

    Notes
    -----
    :math:`C(\mathbf{u}) = t_{\nu,\Sigma}(t_\nu^{-1}(u_1), \dots)`.

    Symmetric tail dependence in both tails,

    .. math::

        \lambda_L = \lambda_U
          = 2\, t_{\nu+1}\!\left(-\sqrt{\tfrac{(\nu+1)(1-\rho)}{1+\rho}}\right),

    which is strictly positive for every finite ``df`` — even at
    :math:`\rho = 0`. As ``df`` grows the copula converges to the Gaussian and
    the tail dependence vanishes.

    Non-integer degrees of freedom are supported (R's ``pmvt`` refuses them,
    though ``fitCopula`` happily produces them).

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import StudentCopula
    >>> t = StudentCopula(0.5, df=4, dim=2)
    >>> float(round(t.tau(), 12))        # identical to the Gaussian's
    0.333333333333
    >>> float(round(t.lambda_().lower, 10))
    0.2531699951

    Tail dependence survives zero correlation, unlike the Gaussian:

    >>> float(round(StudentCopula(0.0, df=3).lambda_().upper, 10))
    0.1161165235

    and vanishes as the degrees of freedom grow:

    >>> bool(StudentCopula(0.5, df=1e6).lambda_().upper < 1e-4)
    True
    """

    name = "Student"
    _n_extra = 1

    def __init__(
        self,
        params: ArrayLike = np.nan,
        dim: int = 2,
        dispstr: str = "ex",
        *,
        df: float = 4.0,
        df_fixed: bool = False,
        free: ArrayLike | None = None,
    ) -> None:
        rho = np.atleast_1d(np.asarray(params, dtype=np.float64))
        n_corr = _n_corr_params(dispstr, int(dim))
        # A bare `StudentCopula(dim=3, dispstr="un")` means "all correlations to
        # be estimated", so a single NaN expands to one per pair.
        if rho.size == 1 and np.isnan(rho[0]) and n_corr > 1:
            rho = np.full(n_corr, np.nan)
        # `df` may be supplied either as the last element of `params` or via the
        # keyword; the keyword wins when the vector is only the correlations.
        full = rho if rho.size == n_corr + 1 else np.append(rho, float(df))
        self.df_fixed = bool(df_fixed)
        super().__init__(full, dim, dispstr, free=free)
        if free is None and df_fixed:
            mask = np.ones(full.shape, dtype=bool)
            mask[-1] = False
            self._free = mask
            self._free.flags.writeable = False

    def _make_param_names(self, dispstr, dim):
        return (*super()._make_param_names(dispstr, dim), "df")

    @property
    def df(self) -> float:
        """Degrees of freedom: smaller means more joint extremes.

        Returns
        -------
        float
            The last entry of :attr:`params`; ``nan`` if not yet estimated.
        """
        return float(self._params[-1])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Allowed range ``(lower, upper)`` for each parameter.

        Returns
        -------
        list of tuple of (float, float), length n_corr + 1
            The correlation bounds of :class:`EllipticalCopula`, followed by
            ``(0.01, inf)`` for ``df``.
        """
        return [*super().param_bounds, (1e-2, np.inf)]

    def _quantile(self, u, params):
        return student_t.ppf(u, df=float(params[-1]))

    def _marginal_cdf(self, x, params):
        return student_t.cdf(x, df=float(params[-1]))

    def _logpdf(self, u, params):
        nu = float(params[-1])
        d = self._dim
        x = student_t.ppf(u, df=nu)
        sigma = _build_sigma(params[: self._n_corr], self.dispstr, d)
        chol = cho_factor(sigma, lower=True)
        log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
        quad = np.einsum("ij,ij->i", x, cho_solve(chol, x.T).T)

        log_joint = (
            gammaln((nu + d) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * log_det
            - (nu + d) / 2.0 * np.log1p(quad / nu)
        )
        log_margins = np.sum(student_t.logpdf(x, df=nu), axis=1)
        return log_joint - log_margins

    def _cdf(self, u, params):
        nu = float(params[-1])
        sigma = _build_sigma(params[: self._n_corr], self.dispstr, self._dim)
        return mvt_cdf(student_t.ppf(u, df=nu), sigma, nu)

    def _rvs_latent(self, size, params, rng):
        nu = float(params[-1])
        sigma = _build_sigma(params[: self._n_corr], self.dispstr, self._dim)
        chol = np.linalg.cholesky(sigma)
        z = rng.standard_normal((size, self._dim)) @ chol.T
        # X = Z / sqrt(W / nu) with W ~ chi2_nu.
        w = rng.chisquare(nu, size)[:, None]
        return z / np.sqrt(w / nu)

    def _reconstruct(self, params, free):
        return StudentCopula(params, self._dim, self.dispstr, df_fixed=self.df_fixed, free=free)

    def lambda_(self) -> TailDependence:
        r"""Tail dependence: how likely joint extremes are, equal in both tails.

        Tail dependence is the chance that one variable is extreme given that
        another is equally extreme, in the limit. For the t copula it is the
        same in the lower and upper tail and positive for every finite ``df``.

        Returns
        -------
        TailDependence
            ``lower == upper``, each in ``[0, 1]``, from
            :math:`2\, t_{\nu+1}(-\sqrt{(\nu+1)(1-\rho)/(1+\rho)})`.

        Raises
        ------
        ValueError
            If the parameters are still unspecified (``nan``), or if
            ``dim > 2`` and the pairs do not all share one correlation (see
            Notes).

        Notes
        -----
        Tail dependence is a pairwise quantity. It is a single number when
        ``dim == 2`` or when every pair has the same correlation (always so
        for ``dispstr="ex"``). Otherwise each pair has its own value and no
        single answer is honest, so this raises rather than silently
        reporting one pair; take the pair you want with
        :func:`~rcopula.marginal_copula`, e.g.
        ``marginal_copula(cop, [0, 2]).lambda_()``.

        Examples
        --------
        >>> from rcopula import StudentCopula, marginal_copula
        >>> cop = StudentCopula([0.2, 0.5, 0.7], dim=3, dispstr="un", df=4)
        >>> cop.lambda_()
        Traceback (most recent call last):
            ...
        ValueError: tail dependence differs between pairs of this 3-dimensional t copula...
        >>> pair = marginal_copula(cop, [1, 2])
        >>> bool(pair.lambda_().upper > 0)
        True
        """
        self._require_specified()
        nu = self.df
        pairs = P2p(self.sigma())
        if np.ptp(pairs) != 0.0:
            raise ValueError(
                f"tail dependence differs between pairs of this {self._dim}-dimensional "
                f"t copula (dispstr={self.dispstr!r}), so there is no single value. "
                "Take the bivariate margin of the pair you want, e.g. "
                "rcopula.marginal_copula(cop, [0, 1]).lambda_()"
            )
        rho = float(pairs[0])
        value = 2.0 * student_t.cdf(-np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)), df=nu + 1.0)
        return TailDependence(lower=float(value), upper=float(value))

    def rho(self) -> Any:
        r"""Spearman's rho: the correlation of the variables' ranks, computed numerically.

        Spearman's rho is the ordinary correlation of the uniform-scale
        variables, between -1 and 1. The t copula has no closed form for it, so
        it is computed by numerical integration (accurate to about twelve
        digits).

        Returns
        -------
        float or numpy.ndarray of float, shape (d * (d - 1) / 2,)
            A float when ``dim=2`` or ``dispstr="ex"``; otherwise one value per
            pair, in :func:`P2p` order.

        Raises
        ------
        ValueError
            If the parameters are still unspecified (``nan``).

        Notes
        -----
        Kendall's tau is :math:`\frac{2}{\pi}\arcsin\rho` for *every* elliptical
        copula, so the t copula inherits the Gaussian expression. **Spearman's
        rho does not.** The relation
        :math:`\rho_S = \frac{6}{\pi}\arcsin(\rho/2)` is specific to the
        Gaussian copula, and applying it to a t copula is simply wrong: at
        :math:`\rho = 0.5` with 4 degrees of freedom it gives 0.4826 where the
        true value is 0.4690.

        R declines the question and returns ``NA``. There is no closed form, but
        :math:`\rho_S = 12\int\int C - 3` is a two-dimensional integral of a
        smooth function, and tanh-sinh quadrature settles it to twelve digits.
        The result is cached, since the integral costs about 80 ms.

        Examples
        --------
        Below the Gaussian value at the same correlation, and converging up to
        it as the degrees of freedom grow:

        >>> from rcopula import GaussianCopula, StudentCopula
        >>> float(round(StudentCopula(0.5, df=4).rho(), 6))
        0.46902
        >>> float(round(GaussianCopula(0.5).rho(), 6))
        0.482584
        >>> bool(abs(StudentCopula(0.5, df=1e6).rho() - GaussianCopula(0.5).rho()) < 1e-5)
        True
        """
        self._require_specified()
        values = np.array([_student_rho(float(r), self.df) for r in P2p(self.sigma())])
        return float(values[0]) if self.dispstr == "ex" or self._dim == 2 else values

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> StudentCopula:
        r"""Build a t copula whose Kendall's tau equals the value you ask for.

        Uses :math:`\rho = \sin(\pi\tau/2)`, which holds for every elliptical
        copula; ``df`` does not affect tau, so set it through ``kwargs``.

        Parameters
        ----------
        tau : float
            Target Kendall's tau, strictly between -1 and 1.
        dim : int, default 2
            Number of variables.
        **kwargs
            Passed to the constructor, e.g. ``df=3`` or ``dispstr="ar1"``.
            With ``"toep"`` or ``"un"`` every pair gets the same correlation,
            as in :meth:`EllipticalCopula.from_tau`.

        Returns
        -------
        StudentCopula

        Raises
        ------
        ValueError
            If ``tau`` is not in ``(-1, 1)``, or the correlation is not
            admissible for ``dim``.

        Examples
        --------
        >>> from rcopula import StudentCopula
        >>> cop = StudentCopula.from_tau(0.5, df=3)
        >>> float(round(cop.tau(), 12)), cop.df
        (0.5, 3.0)
        """
        if not -1.0 < tau < 1.0:
            raise ValueError(f"tau must lie in (-1, 1), got {tau}")
        return cls(_equicorrelation(np.sin(np.pi * tau / 2.0), dim, kwargs), dim, **kwargs)

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> StudentCopula:
        r"""Build a t copula whose Spearman's rho equals the value you ask for.

        Because the t copula's Spearman's rho has no closed form, the
        correlation is found numerically (Brent's method on :meth:`rho`), so
        this takes around a second.

        Parameters
        ----------
        rho : float
            Target Spearman's rho, strictly between -1 and 1.
        dim : int, default 2
            Number of variables.
        **kwargs
            Passed to the constructor. ``df`` (default 4.0) is also used in the
            inversion, since the answer depends on it. With ``"toep"`` or
            ``"un"`` every pair gets the same correlation, as in
            :meth:`EllipticalCopula.from_tau`.

        Returns
        -------
        StudentCopula

        Raises
        ------
        ValueError
            If ``rho`` is not in ``(-1, 1)``, is too close to ``+-1`` to be
            reached by any representable correlation, or the correlation found
            is not admissible for ``dim``.

        Notes
        -----
        The Gaussian relation :math:`\rho_P = 2\sin(\pi\rho_S/6)` -- which R
        uses, and which this method used to use -- is not valid for a t copula,
        so the inversion is done against the quadrature value instead.
        Spearman's rho is strictly increasing in the correlation, so the root is
        bracketed by the whole half-range: ``[0, 1)`` for a positive target and
        ``(-1, 0]`` for a negative one. Brent's method then needs about ten
        quadratures, against the eighty bisection steps used before (which also
        searched only ``+-0.2`` around the Gaussian value and silently returned
        the edge of that window when the root lay outside it).

        Examples
        --------
        >>> from rcopula import StudentCopula
        >>> cop = StudentCopula.from_rho(0.6, df=3)
        >>> bool(abs(cop.rho() - 0.6) < 1e-8)
        True
        """
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {rho}")
        df = float(kwargs.get("df", 4.0))
        if rho == 0.0:
            return cls(_equicorrelation(0.0, dim, kwargs), dim, **kwargs)

        edge = 1.0 - 1e-10
        end = edge if rho > 0 else -edge
        reached = _student_rho(end, df)
        if abs(rho) >= abs(reached):
            raise ValueError(
                f"Spearman's rho = {rho} is too close to {'+' if rho > 0 else '-'}1 for a "
                f"t copula with df={df}: the most extreme value reachable is {reached!r}"
            )
        lo, hi = (0.0, end) if rho > 0 else (end, 0.0)
        correlation = float(
            brentq(lambda r: _student_rho(r, df) - rho, lo, hi, xtol=1e-13, rtol=8.9e-16)
        )
        return cls(_equicorrelation(correlation, dim, kwargs), dim, **kwargs)


#: Quadrature level for the Student-t Spearman rho. The bivariate t CDF is
#: smooth, so the tanh-sinh rule is converged to twelve digits well before this;
#: 40 nodes costs ~80 ms and buys a large margin.
_STUDENT_RHO_LEVEL = 40


@lru_cache(maxsize=256)
def _student_rho(correlation: float, df: float) -> float:
    """Spearman's rho of a bivariate t copula, cached.

    Separate from the method so the cache is keyed on the two scalars that
    actually determine the answer, rather than on a copula object.
    """
    from rcopula.core.measures import rho_by_quadrature

    if correlation == 0.0:
        return 0.0
    return rho_by_quadrature(StudentCopula(correlation, df=df), level=_STUDENT_RHO_LEVEL)
