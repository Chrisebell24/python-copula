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

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, toeplitz
from scipy.special import gammaln, ndtr, ndtri
from scipy.stats import t as student_t

from rcopula.core.base import Copula, TailDependence
from rcopula.special.mvtnorm import mvn_cdf, mvt_cdf

__all__ = ["EllipticalCopula", "GaussianCopula", "P2p", "StudentCopula", "p2P"]

DISPSTRS = ("ex", "ar1", "toep", "un")


def p2P(param: ArrayLike, dim: int) -> NDArray[np.float64]:
    """Build a correlation matrix from its lower triangle (R's ``p2P``).

    Entries fill the lower triangle column by column, so for ``dim=3`` the
    parameter vector is ``(rho_12, rho_13, rho_23)``.

    Examples
    --------
    >>> from rcopula.core.elliptical import p2P
    >>> p2P([0.6, 0.3, 0.2], 3)
    array([[1. , 0.6, 0.3],
           [0.6, 1. , 0.2],
           [0.3, 0.2, 1. ]])
    """
    param = np.asarray(param, dtype=np.float64).ravel()
    expected = dim * (dim - 1) // 2
    if param.size != expected:
        raise ValueError(f"need {expected} parameters for dim={dim}, got {param.size}")
    out = np.eye(dim)
    idx = np.tril_indices(dim, -1)
    out[idx] = param
    out.T[idx] = param
    return out


def P2p(matrix: ArrayLike) -> NDArray[np.float64]:
    """Extract the lower triangle of a correlation matrix (R's ``P2p``).

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.core.elliptical import P2p, p2P
    >>> P2p(p2P([0.6, 0.3, 0.2], 3))
    array([0.6, 0.3, 0.2])
    """
    m = np.asarray(matrix, dtype=np.float64)
    return m[np.tril_indices(m.shape[0], -1)]


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


class EllipticalCopula(Copula):
    """Shared machinery for the Gaussian and Student-t copulas."""

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
            i, j = np.tril_indices(dim, -1)
            names = tuple(f"rho.{b + 1}{a + 1}" for a, b in zip(i, j, strict=True))
        return names

    # -- correlation ---------------------------------------------------

    @property
    def rho_params(self) -> NDArray[np.float64]:
        """The correlation parameters, excluding any extra (e.g. ``df``)."""
        return self._params[: self._n_corr]

    def sigma(self) -> NDArray[np.float64]:
        """The implied ``d x d`` correlation matrix (R's ``getSigma``).

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
        r"""Kendall's tau, :math:`(2/\pi)\arcsin(\rho)`.

        Returns a float when there is a single correlation parameter, and the
        vector of pairwise values otherwise (matching R). The identity holds for
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
        r"""Spearman's rho, :math:`(6/\pi)\arcsin(\rho/2)`."""
        self._require_specified()
        vals = 6.0 / np.pi * np.arcsin(P2p(self.sigma()) / 2.0)
        return float(vals[0]) if self.dispstr == "ex" or self._dim == 2 else vals

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> EllipticalCopula:
        r"""Calibrate from Kendall's tau: :math:`\rho = \sin(\pi\tau/2)`."""
        if not -1.0 < tau < 1.0:
            raise ValueError(f"tau must lie in (-1, 1), got {tau}")
        return cls(np.sin(np.pi * tau / 2.0), dim, **kwargs)

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> EllipticalCopula:
        r"""Calibrate from Spearman's rho: :math:`\rho_P = 2\sin(\pi\rho_S/6)`."""
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {rho}")
        return cls(2.0 * np.sin(np.pi * rho / 6.0), dim, **kwargs)


class GaussianCopula(EllipticalCopula):
    r"""Gaussian (normal) copula.

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
        """Zero in both tails — the Gaussian copula's defining limitation."""
        self._require_specified()
        return TailDependence(lower=0.0, upper=0.0)


class StudentCopula(EllipticalCopula):
    r"""Student-t copula.

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
        """Degrees of freedom."""
        return float(self._params[-1])

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
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
        self._require_specified()
        nu = self.df
        rho = float(P2p(self.sigma())[0])
        value = 2.0 * student_t.cdf(-np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)), df=nu + 1.0)
        return TailDependence(lower=float(value), upper=float(value))

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> StudentCopula:
        if not -1.0 < tau < 1.0:
            raise ValueError(f"tau must lie in (-1, 1), got {tau}")
        return cls(np.sin(np.pi * tau / 2.0), dim, **kwargs)

    @classmethod
    def from_rho(cls, rho: float, dim: int = 2, **kwargs: Any) -> StudentCopula:
        """Calibrate from Spearman's rho.

        Uses the Gaussian relation :math:`\\rho_P = 2\\sin(\\pi\\rho_S/6)`, as R
        does: the t copula's Spearman rho has no closed form, and the Gaussian
        expression is an excellent approximation that becomes exact as
        ``df -> inf``.
        """
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {rho}")
        return cls(2.0 * np.sin(np.pi * rho / 6.0), dim, **kwargs)
