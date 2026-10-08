r"""Factor copulas: a few common factors drive hundreds of variables.

An unstructured Gaussian or Student-t copula on ``d`` variables has
``d(d-1)/2`` correlations -- 319,600 for 800 stocks, far more than a few years
of data can pin down. A **factor copula** assumes instead that a handful of
hidden factors drive all the co-movement. Each variable's latent score is

.. math::

    Z_i = a_i M + b_i G_{g(i)} + \sqrt{1 - a_i^2 - b_i^2}\,\varepsilon_i ,

where :math:`M` is one common ("market") factor, :math:`G_g` is the factor of
variable :math:`i`'s group :math:`g(i)` (a sector, a region, a rating bucket),
and the :math:`\varepsilon_i` are independent noise. All of them are standard
normal, so :math:`Z` is multivariate normal with correlation matrix

.. math::

    R = A A^\top + \operatorname{diag}(1 - a_i^2 - b_i^2),

where row :math:`i` of :math:`A` is :math:`a_i` in the market column and
:math:`b_i` in its group's column. That is ``2d`` parameters instead of
``d(d-1)/2``. The **Gaussian** factor copula is the copula of :math:`Z`. The
**Student-t** factor copula divides every :math:`Z_i` by one shared
:math:`\sqrt{W/\nu}` with :math:`W \sim \chi^2_\nu` -- a common "panic"
variable that makes every pair crash (and rally) together -- which adds one
degrees-of-freedom parameter and gives every pair tail dependence.

Both are exactly a Gaussian or Student-t copula with the structured matrix
:math:`R`, so their density is known in closed form. The point of the structure
is speed: :math:`R^{-1}` and :math:`\det R` follow from the Woodbury identity
and the matrix-determinant lemma using only a :math:`k \times k` matrix
(:math:`k` = number of factors), so the density costs :math:`O(dk)` per
observation and no :math:`d \times d` matrix is ever inverted. 800 variables
are as easy as 8.

============================  ================================================
:class:`FactorCopula`         The model: density, sampler, distribution
                              function, pairwise dependence measures.
:func:`fit_factor`            Estimate the loadings (and ``df``) from data.
============================  ================================================

References
----------
Krupskii, P. and Joe, H. (2013). Factor copula models for multivariate data.
    *Journal of Multivariate Analysis* 120, 85-101.
Hull, J. and White, A. (2004). Valuation of a CDO and an n-th to default CDS
    without Monte Carlo simulation. *Journal of Derivatives* 12(2), 8-23.
    The conditional-independence quadrature used by :meth:`FactorCopula.cdf`.
Lindskog, F., McNeil, A. and Schmock, U. (2003). Kendall's tau for elliptical
    distributions. In *Credit Risk*, Physica-Verlag, 149-156.
    The identity :math:`\tau = (2/\pi)\arcsin r`, used both ways here.
Demarta, S. and McNeil, A. J. (2005). The t copula and related copulas.
    *International Statistical Review* 73(1), 111-129.
    Tail dependence of the t copula and the profile likelihood for ``df``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate, linalg, optimize, special

from rcopula.core.base import Copula, TailDependence
from rcopula.core.elliptical import GaussianCopula, P2p, StudentCopula
from rcopula.dependence import cor_kendall
from rcopula.fit.api import _as_pseudo_obs

if TYPE_CHECKING:
    from rcopula.fit.results import CopulaFitResult

__all__ = ["DEFAULT_DF_GRID", "FactorCopula", "fit_factor"]

#: The copula families a :class:`FactorCopula` can take.
FAMILIES = ("gaussian", "student")

#: Degrees of freedom tried by :func:`fit_factor` before refining the best one.
DEFAULT_DF_GRID: tuple[float, ...] = (2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0)

# Roughly 4 million latent values per block: the density works
# through rows in blocks this size so that 50,000 x 800 needs ~100 MB, not 1 GB.
_BLOCK_ELEMENTS = 1 << 22
# The sampler's blocks are larger (16.8 million values, ~400 MB of working
# memory at peak), since its draws depend on the block size: 20,000 days of
# 800 stocks come out of one block, exactly as if drawn all at once.
_RVS_BLOCK_ELEMENTS = 1 << 24

# Quadrature sizes for the distribution function (see FactorCopula.cdf).
_N_HERMITE = 48
_N_LAGUERRE = 40

# Distinct correlations above which a Student-t rho_matrix interpolates.
_N_RHO_NODES = 15


class FactorCopula(Copula):
    """A copula in which one market factor (plus optional group factors) drives every variable.

    Each variable ``i`` has a hidden score
    ``Z_i = a_i * M + b_i * G[g(i)] + sqrt(1 - a_i**2 - b_i**2) * e_i``: a
    loading ``a_i`` on a market factor ``M`` shared by all variables, a loading
    ``b_i`` on the factor of its group ``g(i)`` (for example its sector), and
    noise of its own. Given the factors, the variables are independent. That
    needs ``2d`` numbers instead of the ``d(d-1)/2`` correlations of a full
    matrix, which is what makes 800 stocks tractable. Use :func:`fit_factor`
    to estimate one from data.

    With ``family="student"`` all scores are divided by one shared random
    scale (a chi-square "panic" variable with ``df`` degrees of freedom), so
    every pair of variables tends to crash -- and rally -- together.

    Parameters
    ----------
    market : array_like of float, shape (d,)
        Market loadings ``a_i``, each in ``[-1, 1]``. Must have ``d >= 2``
        entries. ``nan`` means "to be estimated" (see :meth:`unfitted`).
    group_loadings : array_like of float, shape (d,), or None, default None
        Group loadings ``b_i``, each in ``[-1, 1]``. Give it together with
        ``groups``, or leave both as ``None`` for a one-factor model.
    groups : array_like, shape (d,), or None, default None
        Group label of each variable (integers or strings). Labels are
        re-coded to ``0, 1, ..., n_groups - 1`` in sorted order; read the
        codes back from :attr:`groups`.
    family : {"gaussian", "student"} or None, default None
        Keyword-only. ``None`` means ``"student"`` if ``df`` is given and
        ``"gaussian"`` otherwise.
    df : float or None, default None
        Keyword-only. Degrees of freedom of the Student-t family, at least
        ``0.01``; ``None`` gives ``4.0`` when ``family="student"``. Not
        allowed for the Gaussian family.
    free : array_like of bool, shape (n_params,), or None, default None
        Keyword-only. Free/fixed mask, as for every
        :class:`~rcopula.core.base.Copula`. :func:`rcopula.fit` honours a
        pinned ``df``; loadings cannot be pinned.

    Attributes
    ----------
    family : str
        ``"gaussian"`` or ``"student"``.
    market : numpy.ndarray of float, shape (d,)
        Market loadings (read-only view of :attr:`params`).
    group_loadings : numpy.ndarray of float, shape (d,), or None
        Group loadings, or ``None`` for a one-factor model.
    groups : numpy.ndarray of int, shape (d,), or None
        Group code of each variable, in ``0..n_groups - 1``.
    group_labels : numpy.ndarray, shape (n_groups,), or None
        The original label behind each code.
    n_groups : int
        Number of group factors (``0`` for a one-factor model).
    n_factors : int
        ``1 + n_groups``.
    df : float
        Degrees of freedom; ``inf`` for the Gaussian family.
    idiosyncratic : numpy.ndarray of float, shape (d,)
        Each variable's own-noise variance, ``1 - a_i**2 - b_i**2``.
    params : numpy.ndarray of float, shape (n_params,)
        ``market``, then ``group_loadings`` (if any), then ``df`` (Student-t
        only). :attr:`n_params` therefore counts every loading plus ``df``.
    dim : int
        Number of variables ``d``.

    Raises
    ------
    ValueError
        If ``d < 2``; if exactly one of ``group_loadings`` and ``groups`` is
        given or either has the wrong length; if ``family`` is unknown or
        ``df`` is given with ``family="gaussian"``; if a loading is outside
        ``[-1, 1]``; or if some ``a_i**2 + b_i**2 >= 1`` (no noise left).

    Notes
    -----
    The copula is exactly a :class:`~rcopula.GaussianCopula` or
    :class:`~rcopula.StudentCopula` whose correlation matrix is
    ``R = A A' + diag(1 - a**2 - b**2)`` (:meth:`sigma`); :meth:`to_elliptical`
    builds that dense equivalent. The density uses the Woodbury identity and
    the matrix-determinant lemma, so it costs ``O(d k)`` per observation for
    ``k = 1 + n_groups`` factors and never forms a ``d x d`` inverse: 1,000
    days of 800 stocks evaluate in a fraction of a second. The sampler is
    ``O(d)`` per draw.

    Pairwise measures follow from ``R``: Kendall's tau is
    ``(2/pi) arcsin(r_ij)`` for both families, Spearman's rho is
    ``(6/pi) arcsin(r_ij/2)`` for the Gaussian (and computed numerically for
    the Student-t), and the Student-t tail dependence is
    ``2 t_{df+1}(-sqrt((df+1)(1-r_ij)/(1+r_ij)))``. :meth:`tau`, :meth:`rho`
    and :meth:`lambda_` follow the elliptical classes' conventions; the
    ``*_matrix`` methods return full ``d x d`` matrices.

    Examples
    --------
    Six stocks in two sectors, with a Student-t panic factor:

    >>> import numpy as np
    >>> import rcopula as rc
    >>> cop = rc.FactorCopula(
    ...     [0.6, 0.5, 0.55, 0.6, 0.5, 0.45],
    ...     group_loadings=[0.4, 0.3, 0.35, 0.3, 0.4, 0.45],
    ...     groups=[0, 0, 0, 1, 1, 1],
    ...     df=4.0,
    ... )
    >>> cop.dim, cop.n_factors, cop.n_params
    (6, 3, 13)
    >>> R = cop.sigma()
    >>> round(float(R[0, 1]), 4), round(float(R[0, 3]), 4)   # same vs different sector
    (0.42, 0.36)
    >>> u = cop.rvs(1000, random_state=0)
    >>> u.shape
    (1000, 6)

    It is exactly the Student-t copula with that matrix:

    >>> dense = cop.to_elliptical()
    >>> bool(np.allclose(cop.logpdf(u[:5]), dense.logpdf(u[:5]), atol=1e-9))
    True
    >>> bool(cop.lambda_matrix()[0, 1] > 0)               # crashes are shared
    True
    """

    name = "Factor"

    def __init__(
        self,
        market: ArrayLike,
        group_loadings: ArrayLike | None = None,
        groups: ArrayLike | None = None,
        *,
        family: str | None = None,
        df: float | None = None,
        free: ArrayLike | None = None,
    ) -> None:
        a = np.atleast_1d(np.asarray(market, dtype=np.float64))
        if a.ndim != 1:
            raise ValueError(f"market must be 1-dimensional, got shape {a.shape}")
        d = a.size
        if (group_loadings is None) != (groups is None):
            raise ValueError("give group_loadings and groups together, or neither")

        codes: NDArray[np.intp] | None = None
        labels: NDArray[Any] | None = None
        b: NDArray[np.float64] | None = None
        if groups is not None:
            raw = np.asarray(groups)
            if raw.shape != (d,):
                raise ValueError(f"groups must have shape ({d},), got {raw.shape}")
            labels, inverse = np.unique(raw, return_inverse=True)
            codes = np.asarray(inverse, dtype=np.intp).reshape(d)
            b = np.atleast_1d(np.asarray(group_loadings, dtype=np.float64))
            if b.shape != (d,):
                raise ValueError(f"group_loadings must have shape ({d},), got {b.shape}")

        fam = family if family is not None else ("student" if df is not None else "gaussian")
        if fam not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {fam!r}")
        if fam == "gaussian" and df is not None:
            raise ValueError("df applies only to family='student'")

        self._family = fam
        self._d = d
        self._codes = codes
        self._labels = labels
        self._n_groups = 0 if labels is None else int(labels.size)
        names = [f"market_{i}" for i in range(d)]
        values = [a]
        if b is not None:
            names += [f"group_{i}" for i in range(d)]
            values.append(b)
        if fam == "student":
            names.append("df")
            values.append(np.array([4.0 if df is None else float(df)]))
        self.param_names = tuple(names)
        super().__init__(np.concatenate(values), d, free=free)

    @classmethod
    def unfitted(
        cls,
        dim: int,
        groups: ArrayLike | None = None,
        family: str = "student",
    ) -> FactorCopula:
        """Create a factor copula whose loadings are still to be estimated.

        Every loading (and ``df``) is ``nan``. Pass it to
        :func:`rcopula.fit` or :meth:`rcopula.garch.CopulaGarch.fit`, which
        estimate it with :func:`fit_factor`; it cannot be evaluated before.

        Parameters
        ----------
        dim : int
            Number of variables ``d``, at least 2.
        groups : array_like, shape (d,), or None, default None
            Group label of each variable; ``None`` for one market factor only.
        family : {"gaussian", "student"}, default "student"
            Copula family.

        Returns
        -------
        FactorCopula
            A template with ``nan`` parameters.

        Raises
        ------
        ValueError
            If ``dim < 2``, ``groups`` has the wrong length or ``family`` is
            unknown.

        Examples
        --------
        >>> import rcopula as rc
        >>> template = rc.FactorCopula.unfitted(4, groups=[0, 0, 1, 1])
        >>> template.n_params
        9
        """
        nan = np.full(int(dim), np.nan)
        return cls(
            nan,
            None if groups is None else nan.copy(),
            groups,
            family=family,
            df=np.nan if family == "student" else None,
        )

    # -- plumbing ------------------------------------------------------

    @property
    def family(self) -> str:
        """The copula family: ``"gaussian"`` or ``"student"``.

        Returns
        -------
        str
            Family name.
        """
        return self._family

    @property
    def market(self) -> NDArray[np.float64]:
        """Market loadings ``a_i``, one per variable.

        Returns
        -------
        numpy.ndarray of float, shape (d,)
            Read-only view of the first ``d`` parameters.
        """
        return self._params[: self._d]

    @property
    def group_loadings(self) -> NDArray[np.float64] | None:
        """Group loadings ``b_i``, one per variable, or ``None`` without groups.

        Returns
        -------
        numpy.ndarray of float, shape (d,), or None
            Read-only view of parameters ``d .. 2d - 1``.
        """
        return None if self._codes is None else self._params[self._d : 2 * self._d]

    @property
    def groups(self) -> NDArray[np.intp] | None:
        """Group code of each variable, in ``0 .. n_groups - 1``, or ``None``.

        Returns
        -------
        numpy.ndarray of int, shape (d,), or None
            A copy of the codes; :attr:`group_labels` maps codes to labels.
        """
        return None if self._codes is None else self._codes.copy()

    @property
    def group_labels(self) -> NDArray[Any] | None:
        """The original group label behind each code, or ``None``.

        Returns
        -------
        numpy.ndarray, shape (n_groups,), or None
            Sorted distinct labels; label ``group_labels[c]`` has code ``c``.
        """
        return None if self._labels is None else self._labels.copy()

    @property
    def n_groups(self) -> int:
        """Number of group factors; ``0`` for a one-factor model.

        Returns
        -------
        int
            Count of distinct group labels.
        """
        return self._n_groups

    @property
    def n_factors(self) -> int:
        """Total number of common factors: the market plus one per group.

        Returns
        -------
        int
            ``1 + n_groups``.
        """
        return 1 + self._n_groups

    @property
    def df(self) -> float:
        """Degrees of freedom of the Student-t family; ``inf`` for the Gaussian.

        Returns
        -------
        float
            ``df`` (``nan`` while unfitted), or ``inf``.
        """
        return float(self._params[-1]) if self._family == "student" else float("inf")

    @property
    def idiosyncratic(self) -> NDArray[np.float64]:
        """Each variable's own-noise variance, ``1 - a_i**2 - b_i**2``.

        Returns
        -------
        numpy.ndarray of float, shape (d,)
            Values in ``(0, 1]``; small means the variable is almost fully
            explained by the factors.
        """
        a, b, _ = self._split(self._params)
        return np.asarray(1.0 - a**2 - b**2, dtype=np.float64)

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Allowed range of each parameter, as ``(lower, upper)`` pairs.

        Returns
        -------
        list of tuple of (float, float), length n_params
            ``(-1, 1)`` for every loading and ``(0.01, inf)`` for ``df``. The
            joint condition ``a_i**2 + b_i**2 < 1`` is checked separately.
        """
        bounds = [(-1.0, 1.0)] * (self._d * (2 if self._codes is not None else 1))
        if self._family == "student":
            bounds.append((1e-2, np.inf))
        return bounds

    def _validate_params(self) -> None:
        super()._validate_params()
        a, b, _ = self._split(self._params)
        communality = a**2 + b**2
        bad = np.flatnonzero(communality >= 1.0)  # nan compares False: skipped
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"variable {i} has market loading {a[i]:.6g} and group loading {b[i]:.6g}: "
                "a_i**2 + b_i**2 must be below 1, leaving it some noise of its own"
            )

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> FactorCopula:
        p = np.atleast_1d(np.asarray(params, dtype=np.float64))
        d = self._d
        grouped = self._codes is not None
        return FactorCopula(
            p[:d],
            p[d : 2 * d] if grouped else None,
            self._group_array() if grouped else None,
            family=self._family,
            df=float(p[-1]) if self._family == "student" else None,
            free=free,
        )

    def _group_array(self) -> NDArray[Any]:
        assert self._labels is not None and self._codes is not None
        return self._labels[self._codes]

    def _split(
        self, params: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Market loadings, group loadings (zeros without groups) and df."""
        d = self._d
        a = params[:d]
        b = params[d : 2 * d] if self._codes is not None else np.zeros(d)
        nu = float(params[-1]) if self._family == "student" else float("inf")
        return a, b, nu

    def _loading_matrix(
        self, a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The ``d x k`` matrix ``A``: market column, then one column per group."""
        if self._codes is None:
            return a[:, None].copy()
        out = np.zeros((self._d, 1 + self._n_groups))
        out[:, 0] = a
        out[np.arange(self._d), 1 + self._codes] = b
        return out

    def _structure(
        self, a: NDArray[np.float64], b: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
        """Everything the density needs from ``R``, without forming ``R``.

        With ``R = D + A A'`` (``D`` diagonal), Woodbury gives
        ``x' R^{-1} x = x' D^{-1} x - y' (I + A' D^{-1} A)^{-1} y`` for
        ``y = A' D^{-1} x``, and the determinant lemma gives
        ``log det R = log det D + log det (I + A' D^{-1} A)``. Only the
        ``k x k`` Cholesky factor of ``I + A' D^{-1} A`` is ever factorised.
        """
        noise = 1.0 - a**2 - b**2
        loadings = self._loading_matrix(a, b)
        scaled = loadings / noise[:, None]  # D^{-1} A
        capacitance = np.eye(loadings.shape[1]) + loadings.T @ scaled
        chol = linalg.cholesky(capacitance, lower=True)
        logdet = float(np.sum(np.log(noise)) + 2.0 * np.sum(np.log(np.diag(chol))))
        return noise, scaled, chol, logdet

    @staticmethod
    def _quadratic(
        x: NDArray[np.float64],
        noise: NDArray[np.float64],
        scaled: NDArray[np.float64],
        chol: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Row-wise ``x' R^{-1} x`` via Woodbury, ``O(n d k)``."""
        y = linalg.solve_triangular(chol, (x @ scaled).T, lower=True)
        return np.asarray(np.sum(x**2 / noise, axis=1) - np.sum(y**2, axis=0), dtype=np.float64)

    def _block_rows(self) -> int:
        return max(1, _BLOCK_ELEMENTS // self._d)

    # -- density -------------------------------------------------------

    def _logpdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        a, b, nu = self._split(params)
        noise, scaled, chol, logdet = self._structure(a, b)
        out = np.empty(u.shape[0])
        step = self._block_rows()
        for start in range(0, u.shape[0], step):
            x = _latent(u[start : start + step], nu)
            out[start : start + step] = _log_density(x, noise, scaled, chol, logdet, nu)
        return out

    def loglik(self, data: ArrayLike) -> float:
        """Sum the log-density over all rows: how well the model explains the data.

        Higher is better. Useful for comparing fitted models on the same data
        (for example a Gaussian against a Student-t factor copula on held-out
        days), or for building an AIC/BIC with :attr:`n_params`.

        Parameters
        ----------
        data : array_like of float, shape (n, d)
            Observations. If every value already lies strictly inside
            ``(0, 1)`` they are used as they are; otherwise each column is
            converted to pseudo-observations (ranks scaled into ``(0, 1)``).

        Returns
        -------
        float
            The total log-likelihood, ``sum(log c(u_t))``.

        Raises
        ------
        ValueError
            If the copula is unfitted or ``data`` does not have ``d`` columns.

        Examples
        --------
        >>> import rcopula as rc
        >>> cop = rc.FactorCopula([0.7, 0.6, 0.5])
        >>> u = cop.rvs(500, random_state=1)
        >>> bool(cop.loglik(u) > 0)
        True
        """
        return float(np.sum(self.logpdf(_as_pseudo_obs(data, "average"))))

    # -- distribution function -----------------------------------------

    def _cdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        a, b, nu = self._split(params)
        noise_sd = np.sqrt(1.0 - a**2 - b**2)
        hermite_x, hermite_w = special.roots_hermitenorm(_N_HERMITE)
        log_hw = np.log(hermite_w / hermite_w.sum())
        if np.isinf(nu):
            x = special.ndtri(u)
            scales, log_sw = np.ones(1), np.zeros(1)
        else:
            x = _stdtrit(nu, u)
            # W/2 ~ Gamma(nu/2): generalised Gauss-Laguerre in t = W/2.
            t_nodes, t_weights = special.roots_genlaguerre(_N_LAGUERRE, nu / 2.0 - 1.0)
            scales = np.sqrt(2.0 * t_nodes / nu)
            log_sw = np.log(t_weights / t_weights.sum())
        onehot = None
        if self._codes is not None:
            onehot = np.zeros((self._d, self._n_groups))
            onehot[np.arange(self._d), self._codes] = 1.0
        out = np.empty(u.shape[0])
        for row in range(u.shape[0]):
            per_scale = np.array(
                [
                    _log_conditional_cdf(x[row] * s, a, b, noise_sd, onehot, hermite_x, log_hw)
                    for s in scales
                ]
            )
            out[row] = np.exp(special.logsumexp(per_scale + log_sw))
        return out

    def cdf(self, u: ArrayLike) -> NDArray[np.float64]:
        r"""Probability that every variable is at or below the given point, ``C(u)``.

        Computed by conditioning on the factors: given the market factor and
        the group factors the variables are independent, so ``C(u)`` is a
        product of normal probabilities averaged over the factors.

        Parameters
        ----------
        u : array_like of float, shape (n, d) or (d,)
            Points, one row per point; values outside ``[0, 1]`` are clipped.

        Returns
        -------
        numpy.ndarray of float, shape (n,)
            Probabilities in ``[0, 1]``.

        Raises
        ------
        ValueError
            If the copula is unfitted or ``u`` does not have ``d`` columns.

        Notes
        -----
        The integral over the market factor and over each group factor uses
        48-node Gauss-Hermite quadrature, nested (group factors are
        independent given the market, so the cost is ``O(48^2 d)`` per point,
        not exponential in the number of groups). The Student-t family adds a
        40-node generalised Gauss-Laguerre rule over the chi-square scale,
        multiplying the cost by 40. Everything is summed in logs, so tiny
        joint probabilities do not underflow. Agreement with the dense
        :class:`~rcopula.GaussianCopula` / :class:`~rcopula.StudentCopula`
        distribution functions is typically better than ``1e-4``.

        Examples
        --------
        >>> import rcopula as rc
        >>> cop = rc.FactorCopula([0.6, 0.6])
        >>> dense = cop.to_elliptical()          # Gaussian copula, r = 0.36
        >>> bool(abs(cop.cdf([0.3, 0.4])[0] - dense.cdf([0.3, 0.4])[0]) < 1e-4)
        True
        """
        return super().cdf(u)

    # -- sampling ------------------------------------------------------

    def _rvs(
        self, size: int, params: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        a, b, nu = self._split(params)
        idio = np.sqrt(1.0 - a**2 - b**2)
        out = np.empty((size, self._d))
        step = max(1, _RVS_BLOCK_ELEMENTS // self._d)
        for start in range(0, size, step):
            n = min(step, size - start)
            # Draw order is fixed -- market, groups, noise, then the panic
            # scale -- so a seeded run is reproducible across versions. The
            # arithmetic is done in place to keep one block's temporaries to
            # two arrays.
            z = a * rng.standard_normal((n, 1))
            if self._codes is not None:
                g = rng.standard_normal((n, self._n_groups))[:, self._codes]
                g *= b
                z += g
                del g
            e = rng.standard_normal((n, self._d))
            e *= idio
            z += e
            del e
            if np.isinf(nu):
                special.ndtr(z, out=out[start : start + n])
            else:
                w = rng.chisquare(nu, (n, 1)) / nu  # one shared panic draw per row
                z /= np.sqrt(w)
                special.stdtr(nu, z, out=out[start : start + n])
        return out

    # -- dependence ----------------------------------------------------

    def sigma(self) -> NDArray[np.float64]:
        """Return the implied ``d x d`` correlation matrix ``R = A A' + diag(noise)``.

        Entry ``[i, j]`` is ``a_i a_j`` plus ``b_i b_j`` when ``i`` and ``j``
        share a group; the diagonal is 1. This is the correlation matrix of
        the latent scores (before the Student-t scaling), matching
        :meth:`rcopula.GaussianCopula.sigma`.

        Returns
        -------
        numpy.ndarray of float, shape (d, d)
            Symmetric positive-definite correlation matrix.

        Raises
        ------
        ValueError
            If the copula is unfitted.

        Examples
        --------
        >>> import rcopula as rc
        >>> rc.FactorCopula([0.8, 0.5]).sigma()
        array([[1. , 0.4],
               [0.4, 1. ]])
        """
        self._require_specified()
        a, b, _ = self._split(self._params)
        loadings = self._loading_matrix(a, b)
        out = loadings @ loadings.T
        np.fill_diagonal(out, 1.0)
        return np.asarray(out, dtype=np.float64)

    def to_elliptical(self) -> GaussianCopula | StudentCopula:
        """Build the equivalent dense Gaussian or Student-t copula.

        The factor copula *is* an elliptical copula with correlation matrix
        :meth:`sigma`; this returns that copula with an unstructured
        (``dispstr="un"``) matrix. Handy for checks and for functions that
        need an elliptical copula, but it has ``d(d-1)/2`` parameters and a
        ``O(d^3)`` density, so prefer the factor form for large ``d``.

        Returns
        -------
        GaussianCopula or StudentCopula
            ``StudentCopula`` (with the same ``df``) for the Student-t family.

        Raises
        ------
        ValueError
            If the copula is unfitted.

        Examples
        --------
        >>> import rcopula as rc
        >>> rc.FactorCopula([0.8, 0.5], df=5.0).to_elliptical().describe()
        'Student copula, dim 2, rho.12=0.4, df=5'
        """
        pairs = P2p(self.sigma())
        if self._family == "student":
            return StudentCopula(pairs, dim=self._d, dispstr="un", df=self.df)
        return GaussianCopula(pairs, dim=self._d, dispstr="un")

    def tau_matrix(self) -> NDArray[np.float64]:
        """Kendall's tau of every pair, as a ``d x d`` matrix.

        Uses ``tau_ij = (2/pi) arcsin(r_ij)``, exact for both families.

        Returns
        -------
        numpy.ndarray of float, shape (d, d)
            Symmetric, ones on the diagonal.

        Raises
        ------
        ValueError
            If the copula is unfitted.

        Examples
        --------
        >>> import rcopula as rc
        >>> float(round(rc.FactorCopula([0.8, 0.5]).tau_matrix()[0, 1], 6))
        0.26198
        """
        return np.asarray(2.0 / np.pi * np.arcsin(np.clip(self.sigma(), -1.0, 1.0)))

    def rho_matrix(self) -> NDArray[np.float64]:
        """Spearman's rho of every pair, as a ``d x d`` matrix.

        Returns
        -------
        numpy.ndarray of float, shape (d, d)
            Symmetric, ones on the diagonal.

        Raises
        ------
        ValueError
            If the copula is unfitted.

        Notes
        -----
        Gaussian: ``(6/pi) arcsin(r_ij / 2)``, exact. Student-t: there is no
        closed form, so each distinct correlation is integrated numerically
        as in :meth:`rcopula.StudentCopula.rho`; with more than 15 distinct
        values, a polynomial through 15 integrated Chebyshev points spanning
        them is used instead (error about ``1e-8``), which keeps 800
        variables to several seconds rather than hours.

        Examples
        --------
        >>> import rcopula as rc
        >>> float(round(rc.FactorCopula([0.8, 0.5]).rho_matrix()[0, 1], 6))
        0.384565
        """
        r = self.sigma()
        if self._family == "gaussian":
            return np.asarray(6.0 / np.pi * np.arcsin(r / 2.0))
        out = np.ones_like(r)
        off = ~np.eye(self._d, dtype=bool)
        out[off] = _student_rho_values(r[off], self.df)
        return out

    def lambda_matrix(self) -> NDArray[np.float64]:
        """Tail dependence of every pair, as a ``d x d`` matrix (same in both tails).

        Returns
        -------
        numpy.ndarray of float, shape (d, d)
            Zero off the diagonal for the Gaussian family;
            ``2 t_{df+1}(-sqrt((df+1)(1-r)/(1+r)))`` for the Student-t.
            Ones on the diagonal.

        Raises
        ------
        ValueError
            If the copula is unfitted.

        Examples
        --------
        >>> import rcopula as rc
        >>> cop = rc.FactorCopula([0.8, 0.5], df=4.0)
        >>> lam = cop.lambda_matrix()[0, 1]
        >>> bool(abs(lam - cop.to_elliptical().lambda_().upper) < 1e-12)
        True
        """
        r = self.sigma()
        if self._family == "gaussian":
            return np.eye(self._d)
        nu = self.df
        with np.errstate(divide="ignore"):
            arg = -np.sqrt((nu + 1.0) * (1.0 - r) / (1.0 + r))
        return np.asarray(2.0 * special.stdtr(nu + 1.0, arg))

    def _pairs(self, matrix: NDArray[np.float64]) -> Any:
        values = P2p(matrix)
        return float(values[0]) if self._d == 2 else values

    def tau(self) -> Any:
        """Kendall's tau implied by the copula: one value per pair.

        Returns
        -------
        float or numpy.ndarray of float, shape (d * (d - 1) / 2,)
            A float when ``d == 2``; otherwise every pair in
            :func:`~rcopula.P2p` order (as :meth:`rcopula.GaussianCopula.tau`
            does). :meth:`tau_matrix` gives the same values as a matrix.

        Raises
        ------
        ValueError
            If the copula is unfitted.

        Examples
        --------
        >>> import rcopula as rc
        >>> round(rc.FactorCopula([0.8, 0.5]).tau(), 6)
        0.26198
        """
        return self._pairs(self.tau_matrix())

    def rho(self) -> Any:
        """Spearman's rho implied by the copula: one value per pair.

        Returns
        -------
        float or numpy.ndarray of float, shape (d * (d - 1) / 2,)
            A float when ``d == 2``; otherwise every pair in
            :func:`~rcopula.P2p` order. See :meth:`rho_matrix`.

        Raises
        ------
        ValueError
            If the copula is unfitted.
        """
        return self._pairs(self.rho_matrix())

    def lambda_(self) -> TailDependence:
        """Tail dependence, when every pair shares one value.

        Returns
        -------
        TailDependence
            ``(lower, upper)``, equal to each other; zero for the Gaussian.

        Raises
        ------
        ValueError
            If the copula is unfitted, or if it is a Student-t factor copula
            whose pairs have different tail dependence (the usual case for
            ``d > 2``); use :meth:`lambda_matrix` then.

        Examples
        --------
        >>> import rcopula as rc
        >>> rc.FactorCopula([0.8, 0.5]).lambda_()
        TailDependence(lower=0.0, upper=0.0)
        """
        values = P2p(self.lambda_matrix())
        if np.ptp(values) > 1e-14:
            raise ValueError(
                "tail dependence differs between pairs of this factor copula, so there "
                "is no single value; use lambda_matrix()"
            )
        value = float(values[0])
        return TailDependence(lower=value, upper=value)

    # -- presentation --------------------------------------------------

    def describe(self) -> str:
        """Short human-readable summary: family, size, factors and loadings.

        Returns
        -------
        str
            One line, e.g. ``"Factor copula (Student-t, df=4), dim 800,
            1 market + 10 group factors, market loadings 0.45..0.65 (mean
            0.55), group loadings 0.30..0.50 (mean 0.40), 1601 parameters"``.

        Examples
        --------
        >>> import rcopula as rc
        >>> rc.FactorCopula([0.6, 0.5, 0.4]).describe()  # doctest: +NORMALIZE_WHITESPACE
        'Factor copula (Gaussian), dim 3, 1 market factor,
         market loadings 0.4..0.6 (mean 0.5), 3 parameters'
        """

        def summary(label: str, values: NDArray[np.float64]) -> str:
            if np.isnan(values).any():
                return f"{label} loadings unfitted"
            return (
                f"{label} loadings {values.min():.3g}..{values.max():.3g} "
                f"(mean {values.mean():.3g})"
            )

        fam = "Gaussian" if self._family == "gaussian" else f"Student-t, df={self.df:.4g}"
        factors = (
            "1 market factor"
            if self._codes is None
            else f"1 market + {self._n_groups} group factors"
        )
        parts = [
            f"Factor copula ({fam})",
            f"dim {self._d}",
            factors,
            summary("market", self.market),
        ]
        group = self.group_loadings
        if group is not None:
            parts.append(summary("group", group))
        parts.append(f"{self.n_params} parameters")
        return ", ".join(parts)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FactorCopula):
            return NotImplemented
        same_groups = (self._codes is None and other._codes is None) or (
            self._codes is not None
            and other._codes is not None
            and np.array_equal(self._codes, other._codes)
        )
        return bool(
            self._family == other._family
            and same_groups
            and np.array_equal(self._params, other._params, equal_nan=True)
            and np.array_equal(self._free, other._free)
        )

    def __hash__(self) -> int:
        codes = b"" if self._codes is None else self._codes.tobytes()
        return hash((type(self).__name__, self._family, codes, self._params.tobytes()))


# --------------------------------------------------------------------------
# numerical helpers
# --------------------------------------------------------------------------


def _latent(u: NDArray[np.float64], nu: float) -> NDArray[np.float64]:
    """Normal (``nu = inf``) or Student-t quantiles of ``u``.

    The t quantile function is slow for a general ``nu`` (about 0.6 s for a
    million values), but pseudo-observations take only ``n`` distinct values
    however many columns there are, so it is evaluated once per distinct value.
    """
    if np.isinf(nu):
        return np.asarray(special.ndtri(u))
    distinct, inverse = np.unique(u, return_inverse=True)
    return np.asarray(_stdtrit(nu, distinct)[inverse].reshape(u.shape))


def _log_density(
    x: NDArray[np.float64],
    noise: NDArray[np.float64],
    scaled: NDArray[np.float64],
    chol: NDArray[np.float64],
    logdet: float,
    nu: float,
) -> NDArray[np.float64]:
    """Gaussian (``nu = inf``) or Student-t copula log-density at latent ``x``."""
    d = x.shape[1]
    q = FactorCopula._quadratic(x, noise, scaled, chol)
    if np.isinf(nu):
        return np.asarray(-0.5 * logdet - 0.5 * (q - np.sum(x**2, axis=1)))
    const = (
        special.gammaln((nu + d) / 2.0)
        + (d - 1) * special.gammaln(nu / 2.0)
        - d * special.gammaln((nu + 1.0) / 2.0)
    )
    return np.asarray(
        const
        - 0.5 * logdet
        - 0.5 * (nu + d) * np.log1p(q / nu)
        + 0.5 * (nu + 1.0) * np.sum(np.log1p(x**2 / nu), axis=1)
    )


def _stdtrit(nu: float, u: ArrayLike) -> NDArray[np.float64]:
    """Student-t quantile with exact endpoints: -inf at 0, +inf at 1.

    ``scipy.special.stdtrit`` returns nan at 0 and 1 in SciPy < 1.16 (the
    versions Python 3.10 installs), which turned ``C(u, 1, ..., 1)`` into 0
    instead of ``u``. ``ndtri`` has no such problem.
    """
    arr = np.asarray(u, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        out = np.asarray(special.stdtrit(nu, arr), dtype=np.float64)
    return np.where(arr >= 1.0, np.inf, np.where(arr <= 0.0, -np.inf, out))


def _log_conditional_cdf(
    x: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    noise_sd: NDArray[np.float64],
    onehot: NDArray[np.float64] | None,
    nodes: NDArray[np.float64],
    log_weights: NDArray[np.float64],
) -> float:
    """``log P(Z <= x)`` for the Gaussian factor model, by nested quadrature."""
    if onehot is None:
        arg = (x[None, :] - a[None, :] * nodes[:, None]) / noise_sd
        per_market = special.log_ndtr(arg).sum(axis=1)
    else:
        # arg[m, s, i] = (x_i - a_i M_m - b_i S_s) / sd_i
        centred = x[None, :] - a[None, :] * nodes[:, None]  # (Q, d)
        arg = (centred[:, None, :] - b[None, None, :] * nodes[None, :, None]) / noise_sd
        logs = special.log_ndtr(arg)  # (Q, Q, d)
        by_group = logs @ onehot  # (Q, Q, G): sum within each group
        inner = special.logsumexp(by_group + log_weights[None, :, None], axis=1)  # (Q, G)
        per_market = inner.sum(axis=1)
    return float(special.logsumexp(per_market + log_weights))


def _student_rho_values(r: NDArray[np.float64], df: float) -> NDArray[np.float64]:
    """Spearman's rho of the bivariate t copula at each correlation in ``r``."""
    from rcopula.core.elliptical import _student_rho

    distinct = np.unique(r)
    if distinct.size <= _N_RHO_NODES:
        table = np.array([_student_rho(float(v), float(df)) for v in distinct])
        return np.asarray(table[np.searchsorted(distinct, r)])
    # Polynomial interpolation through Chebyshev points spanning the values:
    # rho is a smooth function of r, so 15 points give about 1e-8.
    lo, hi = float(distinct[0]), float(distinct[-1])
    k = np.arange(_N_RHO_NODES)
    nodes = 0.5 * (lo + hi) - 0.5 * (hi - lo) * np.cos(np.pi * k / (_N_RHO_NODES - 1))
    values = np.array([_student_rho(float(v), float(df)) for v in nodes])
    return np.asarray(interpolate.BarycentricInterpolator(nodes, values)(r))


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------


def fit_factor(
    data: ArrayLike,
    groups: ArrayLike | None = None,
    family: str = "student",
    *,
    method: str = "itau",
    df: float | None = None,
    df_grid: Sequence[float] = DEFAULT_DF_GRID,
    max_iter: int = 2000,
    tol: float = 1e-7,
    max_communality: float = 0.98,
) -> FactorCopula:
    """Fit a factor copula to data: loadings by correlation matching, ``df`` by likelihood.

    Give it a table of observations (one column per variable) and, optionally,
    each variable's group (for example its sector). It returns a
    :class:`FactorCopula` with one market factor plus one factor per group.
    Built for many variables: 800 columns of 1,000 rows take a few seconds.

    Parameters
    ----------
    data : array_like or pandas.DataFrame of float, shape (n, d)
        Observations, one row per observation and one column per variable,
        with ``d >= 2``. If every value lies strictly inside ``(0, 1)`` the
        data are taken as pseudo-observations; otherwise each column is
        rank-transformed first.
    groups : array_like, shape (d,), or None, default None
        Group label of each column. ``None`` fits one market factor only.
    family : {"student", "gaussian"}, default "student"
        Copula family. The Student-t adds a degrees-of-freedom parameter and
        tail dependence.
    method : {"itau", "normal"}, default "itau"
        Which correlation matrix the loadings are matched to. ``"itau"``
        inverts Kendall's tau, ``r = sin(pi * tau / 2)``, which is consistent
        for both families (heavy joint tails do not bias it). ``"normal"``
        uses the Pearson correlation of normal scores, which is faster but
        consistent only for the Gaussian family.
    df : float or None, default None
        Keyword-only. Hold the Student-t degrees of freedom at this value
        instead of estimating them. Not allowed with ``family="gaussian"``.
    df_grid : sequence of float, default DEFAULT_DF_GRID
        Keyword-only. Degrees of freedom at which the profile log-likelihood
        is evaluated before the best one is refined. The estimate is confined
        to ``[min(df_grid), max(df_grid)]``.
    max_iter : int, default 2000
        Keyword-only. Maximum number of correlation-matching sweeps.
    tol : float, default 1e-7
        Keyword-only. The sweeps stop when no loading moves by more than this.
    max_communality : float, default 0.98
        Keyword-only. Upper limit on ``a_i**2 + b_i**2``, so every variable
        keeps at least ``1 - max_communality`` of noise of its own (the
        density needs it to be positive). Must lie in ``(0, 1)``.

    Returns
    -------
    FactorCopula
        The fitted copula. Its :meth:`~FactorCopula.loglik` on the fitting
        data is the maximised (profile) log-likelihood.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D with at least two columns and two rows, if
        ``groups`` has the wrong length, if ``family`` or ``method`` is
        unknown, if ``df`` is given for the Gaussian family, if
        ``max_communality`` is not in ``(0, 1)``, or if the correlation matrix
        has missing values (a constant column, for instance).

    Notes
    -----
    **Loadings.** The model implies ``r_ij = a_i a_j + b_i b_j [same group]``
    off the diagonal. The loadings minimise the squared distance between that
    and the target correlation matrix over all off-diagonal pairs, by
    alternating least squares: each sweep updates every market loading given
    the rest, then every group loading given the new market loadings (the
    iterated principal-factor idea, restricted to the factor pattern).
    Because each loading is fitted to ``d - 1`` correlations, the noise in the
    individual correlations largely averages out. Group loadings are kept
    non-negative (a group factor adds correlation within its group); a group
    with a single member gets loading 0, since it cannot be identified.

    **Degrees of freedom.** With the loadings fixed, the Student-t
    log-likelihood is evaluated at every ``df_grid`` value and the best one
    is refined by a bounded one-dimensional search on ``log(df)`` between its
    grid neighbours. Each evaluation is ``O(n d k)``.

    **Accuracy.** Simulated from a Student-t factor copula with ``df = 4``,
    50 variables in 5 groups, market loadings in ``[0.4, 0.7]`` and group
    loadings in ``[0.3, 0.5]``, 1,000 rows: mean absolute loading error about
    0.02 (market) and 0.03 (group), and ``df`` within about 0.5 of the truth.
    At 800 variables in 10 groups the loading errors shrink to about 0.015
    and the whole fit takes 5-10 seconds, most of it the 319,600 Kendall's
    taus (:func:`~rcopula.cor_kendall`).

    This is a two-stage, moment-based estimator, not full maximum
    likelihood, so there are no standard errors; bootstrap the rows if you
    need them.

    Examples
    --------
    >>> import numpy as np
    >>> import rcopula as rc
    >>> rng = np.random.default_rng(0)
    >>> groups = np.repeat([0, 1, 2], 6)
    >>> truth = rc.FactorCopula(
    ...     rng.uniform(0.4, 0.7, 18), rng.uniform(0.3, 0.5, 18), groups, df=4.0
    ... )
    >>> u = truth.rvs(2000, random_state=1)
    >>> fitted = rc.fit_factor(u, groups=groups)
    >>> bool(np.mean(np.abs(fitted.market - truth.market)) < 0.05)
    True
    >>> bool(2.5 < fitted.df < 6.5)
    True
    >>> fitted.n_params
    37
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if method not in ("itau", "normal"):
        raise ValueError(f"method must be 'itau' or 'normal', got {method!r}")
    if family == "gaussian" and df is not None:
        raise ValueError("df applies only to family='student'")
    if not 0.0 < max_communality < 1.0:
        raise ValueError(f"max_communality must lie in (0, 1), got {max_communality}")

    u = _as_pseudo_obs(data, "average")
    if u.ndim != 2 or u.shape[1] < 2 or u.shape[0] < 2:
        raise ValueError(f"data must be 2-D with at least 2 rows and 2 columns, got {u.shape}")
    d = u.shape[1]
    codes: NDArray[np.intp] | None = None
    if groups is not None:
        raw = np.asarray(groups)
        if raw.shape != (d,):
            raise ValueError(f"groups must have shape ({d},), got {raw.shape}")
        codes = np.asarray(np.unique(raw, return_inverse=True)[1], dtype=np.intp).reshape(d)

    if method == "itau":
        target = np.sin(np.pi / 2.0 * cor_kendall(u))
    else:
        target = np.corrcoef(special.ndtri(u), rowvar=False)
    if not np.all(np.isfinite(target)):
        raise ValueError("the correlation matrix has missing values (a constant column?)")

    a, b = _fit_loadings(target, codes, max_iter, tol, max_communality)
    group_loadings = None if groups is None else b

    if family == "gaussian":
        return FactorCopula(a, group_loadings, groups, family="gaussian")
    if df is not None:
        return FactorCopula(a, group_loadings, groups, df=float(df))

    template = FactorCopula(a, group_loadings, groups, df=4.0)
    noise, scaled, chol, logdet = template._structure(*template._split(template.params)[:2])
    step = template._block_rows()
    distinct, inverse = np.unique(u, return_inverse=True)
    inverse = inverse.reshape(u.shape)

    def loglik(nu: float) -> float:
        quantiles = _stdtrit(nu, distinct)
        return float(
            sum(
                np.sum(
                    _log_density(quantiles[inverse[s : s + step]], noise, scaled, chol, logdet, nu)
                )
                for s in range(0, u.shape[0], step)
            )
        )

    grid = np.unique(np.asarray(df_grid, dtype=np.float64))
    if grid.size == 0 or grid[0] <= 0.01:
        raise ValueError("df_grid must contain degrees of freedom above 0.01")
    profile = np.array([loglik(v) for v in grid])
    best = int(np.argmax(profile))
    nu_hat = float(grid[best])
    if grid.size > 1:
        lo = float(grid[max(best - 1, 0)])
        hi = float(grid[min(best + 1, grid.size - 1)])
        res = optimize.minimize_scalar(
            lambda t: -loglik(float(np.exp(t))),
            bounds=(np.log(lo), np.log(hi)),
            method="bounded",
            options={"xatol": 1e-3},
        )
        if -float(res.fun) > profile[best]:
            nu_hat = float(np.exp(res.x))
    return FactorCopula(a, group_loadings, groups, df=nu_hat)


def _rank_one(
    corr: NDArray[np.float64],
    mask: NDArray[np.bool_],
    start: NDArray[np.float64],
    max_iter: int,
    tol: float,
) -> NDArray[np.float64]:
    """Least-squares ``x`` with ``x_i x_j ~ corr_ij`` over the pairs in ``mask``.

    Each sweep moves every ``x_i`` halfway to its best value given the
    others, ``sum_j corr_ij x_j / sum_j x_j**2`` over ``j`` in ``mask[i]`` --
    a damped power iteration, ``O(d^2)`` per sweep. Without the damping the
    sweep oscillates when the mask leaves out whole blocks (the within-group
    pairs); with it, a few dozen sweeps reach ``1e-10``.
    """
    target = np.where(mask, corr, 0.0)
    x = start.copy()
    for _ in range(max_iter):
        denom = mask @ (x**2)
        best = np.divide(target @ x, denom, out=np.zeros_like(x), where=denom > 1e-12)
        new = 0.5 * (x + best)
        change = float(np.max(np.abs(new - x)))
        x = new
        if change < tol:
            break
    return x


def _fit_loadings(
    target: NDArray[np.float64],
    codes: NDArray[np.intp] | None,
    max_iter: int,
    tol: float,
    max_communality: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Least-squares fit of ``a a' + [same group] b b'`` to the off-diagonal of ``target``."""
    d = target.shape[0]
    off = ~np.eye(d, dtype=bool)
    corr = np.where(off, target, 0.0)
    same = np.zeros((d, d), dtype=bool) if codes is None else (codes[:, None] == codes[None, :])
    same &= off
    across = off & ~same
    # Pairs in different groups see only the market factor, so with three or
    # more groups they pin down the market loadings on their own. With one or
    # two groups they cannot (two groups leave a scale ambiguity between them),
    # so every pair is used and the market absorbs the average correlation.
    n_groups = 0 if codes is None else int(np.unique(codes).size)
    market_pairs = across if n_groups >= 3 else off

    # Start: if corr = a a' on the mask, the row means are a_i * mean(a).
    counts = np.maximum(market_pairs.sum(axis=1), 1)
    row_mean = np.where(market_pairs, corr, 0.0).sum(axis=1) / counts
    scale = np.sqrt(max(float(row_mean.mean()), 1e-4))
    a = _rank_one(corr, market_pairs, np.clip(row_mean / scale, -0.95, 0.95), max_iter, tol)
    if a.sum() < 0:
        a = -a

    b = np.zeros(d)
    if codes is not None:
        # What the market leaves over within each group is the group factor's.
        excess = corr - np.outer(a, a)
        for g in np.unique(codes):
            members = np.flatnonzero(codes == g)
            if members.size < 2:
                continue  # a lone member's group loading cannot be identified
            block = excess[np.ix_(members, members)]
            inner = same[np.ix_(members, members)]
            level = float(block[inner].mean())
            if level <= 0.0:
                continue  # the group adds no correlation beyond the market
            start = np.full(members.size, np.sqrt(level))
            b[members] = np.clip(_rank_one(block, inner, start, max_iter, tol), 0.0, None)

    norm = np.sqrt(a**2 + b**2)
    shrink = np.minimum(1.0, np.sqrt(max_communality) / np.maximum(norm, 1e-300))
    return a * shrink, b * shrink


def _fit_as_result(copula: FactorCopula, u: NDArray[np.float64], method: str) -> CopulaFitResult:
    """What :func:`rcopula.fit` returns for a :class:`FactorCopula`."""
    from rcopula.fit.results import CopulaFitResult

    if method not in ("itau", "mpl", "ml"):
        raise ValueError(
            f"a FactorCopula is fitted with fit_factor; method must be 'itau', 'mpl' or "
            f"'ml', got {method!r}"
        )
    free = np.asarray(copula.free, dtype=bool)
    n_loadings = len(copula.params) - (1 if copula.family == "student" else 0)
    if not free[:n_loadings].all():
        raise ValueError("FactorCopula loadings cannot be pinned with fix_params; only df can")
    pinned_df = copula.family == "student" and not free[-1]
    groups = None if copula.groups is None else copula._group_array()
    fitted = fit_factor(
        u,
        groups=groups,
        family=copula.family,
        df=copula.df if pinned_df else None,
    )
    pinned = fitted.fix_params(free)
    assert isinstance(pinned, FactorCopula)
    fitted = pinned
    return CopulaFitResult(
        copula=fitted,
        params=np.asarray(fitted.params)[free],
        param_names=tuple(np.asarray(fitted.param_names)[free].tolist()),
        loglik=fitted.loglik(u),
        n_obs=u.shape[0],
        method=method,
        cov_params=None,
        converged=True,
        message="loadings by Kendall's-tau correlation matching, df by profile likelihood "
        "(fit_factor); no standard errors",
    )
