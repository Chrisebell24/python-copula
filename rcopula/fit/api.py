r"""Copula estimation.

Five methods, using R's exact option strings so code ports across unchanged:

========== ====================================================================
``"mpl"``  Maximum **pseudo**-likelihood. The default and the usual choice:
           maximise the copula likelihood at rank-based pseudo-observations,
           making no assumption about the margins.
``"ml"``   Same estimate, different standard errors. Use only when the data
           really are copula observations (margins known, not estimated).
``"itau"`` Invert Kendall's tau. Closed form for most families, robust, and a
           good starting value even when it is not the final answer.
``"irho"`` Invert Spearman's rho. Same idea, usually slightly less efficient.
``"itau.mpl"`` Mashal-Zeevi: correlations from inverted tau, degrees of
           freedom by pseudo-likelihood. For t copulas in higher dimensions,
           where a joint optimisation over both is badly conditioned.
========== ====================================================================

``"mpl"`` and ``"ml"`` produce the *same* point estimate and differ only in the
variance -- an easy thing to misread in R's documentation, and the reason
:func:`fit` reports the method it used in the summary.

References
----------
Genest, C., Ghoudi, K. and Rivest, L.-P. (1995). A semiparametric estimation
    procedure of dependence parameters in multivariate families of
    distributions. *Biometrika* 82(3), 543-552.
Mashal, R. and Zeevi, A. (2002). Beyond correlation: extreme co-movements
    between financial assets. Columbia University working paper.
Higham, N. J. (2002). Computing the nearest correlation matrix -- a problem
    from finance. *IMA Journal of Numerical Analysis* 22(3), 329-343.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from rcopula.core.base import Copula
from rcopula.core.elliptical import EllipticalCopula, P2p, StudentCopula, p2P
from rcopula.dependence import pseudo_obs
from rcopula.fit.results import CopulaFitResult
from rcopula.fit.variance import var_inversion_multi, var_itau, var_ml, var_mpl

__all__ = ["METHODS", "fit", "loglik_copula", "nearest_correlation"]

METHODS = ("mpl", "ml", "itau", "irho", "itau.mpl")


def nearest_correlation(
    matrix: ArrayLike, tol: float = 1e-10, max_iter: int = 200
) -> NDArray[np.float64]:
    """Nearest positive-definite correlation matrix, by alternating projections.

    Inverting pairwise dependence measures gives a symmetric matrix with unit
    diagonal, but nothing guarantees it is positive definite -- each entry is
    estimated separately. R applies the same repair (``Matrix::nearPD``) after
    ``itau`` and ``irho`` fits.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.fit import nearest_correlation
    >>> bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    >>> bool(np.linalg.eigvalsh(bad).min() < 0)
    True
    >>> fixed = nearest_correlation(bad)
    >>> bool(np.linalg.eigvalsh(fixed).min() > 0)
    True
    >>> bool(np.allclose(np.diag(fixed), 1.0))
    True
    """
    a = np.asarray(matrix, dtype=np.float64)
    a = 0.5 * (a + a.T)
    y = a.copy()
    delta = np.zeros_like(a)

    for _ in range(max_iter):
        r = y - delta
        # Project onto the positive-semidefinite cone.
        vals, vecs = np.linalg.eigh(r)
        x = (vecs * np.maximum(vals, tol)) @ vecs.T
        delta = x - r
        # Project onto the unit-diagonal set.
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        if np.linalg.eigvalsh(y).min() > tol and np.max(np.abs(x - y)) < tol:
            break

    np.fill_diagonal(y, 1.0)
    # A final nudge, in case the loop exited on the iteration cap.
    vals, vecs = np.linalg.eigh(y)
    if vals.min() <= 0:
        y = (vecs * np.maximum(vals, tol)) @ vecs.T
        d = np.sqrt(np.diag(y))
        y = y / np.outer(d, d)
    return y


def _as_pseudo_obs(data: ArrayLike, ties_method: str) -> NDArray[np.float64]:
    """Coerce input to pseudo-observations, transforming only if needed."""
    frame = data if isinstance(data, pd.DataFrame) else None
    arr = np.asarray(frame.to_numpy() if frame is not None else data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    inside = np.all((arr > 0.0) & (arr < 1.0))
    if inside:
        return arr
    # Values outside the open unit cube cannot be copula observations, so treat
    # the input as raw data. R instead errors; transforming is friendlier and
    # unambiguous, since the two cases cannot overlap.
    return np.asarray(pseudo_obs(arr, ties_method=ties_method), dtype=np.float64)


def loglik_copula(params: ArrayLike, u: ArrayLike, copula: Copula, error: str = "-inf") -> float:
    """Log-likelihood of a copula at given parameters (R's ``loglikCopula``).

    Parameters
    ----------
    params : array_like
        Parameter vector to evaluate at.
    u : array_like
        ``(n, d)`` observations in the unit cube.
    copula : Copula
        Supplies the family; its own parameter values are ignored.
    error : {"-inf", "raise"}
        What to do when the parameters are inadmissible.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, loglik_copula
    >>> u = ClaytonCopula(2.0).rvs(500, random_state=0)
    >>> at_truth = loglik_copula([2.0], u, ClaytonCopula())
    >>> at_wrong = loglik_copula([0.5], u, ClaytonCopula())
    >>> bool(at_truth > at_wrong)
    True

    Inadmissible parameters give ``-inf`` rather than an error, so optimisers
    can walk into them safely:

    >>> bool(np.isneginf(loglik_copula([-5.0], u, ClaytonCopula())))
    True
    """
    arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
    try:
        candidate = copula.with_params(params)
        value = float(np.sum(candidate.logpdf(arr)))
    except (ValueError, np.linalg.LinAlgError, NotImplementedError):
        if error == "raise":
            raise
        return -np.inf
    return value if np.isfinite(value) else -np.inf


# ======================================================================
# Moment estimators
# ======================================================================


def _pairwise_measure(u: NDArray[np.float64], measure: str) -> NDArray[np.float64]:
    """Vector of pairwise sample tau or rho, in lower-triangle order."""
    d = u.shape[1]
    fn = stats.kendalltau if measure == "tau" else stats.spearmanr
    out = []
    for j in range(d):
        for i in range(j + 1, d):
            out.append(float(fn(u[:, i], u[:, j]).statistic))
    return np.array(out)


def _fit_by_inversion(
    copula: Copula, u: NDArray[np.float64], measure: str, estimate_variance: bool
) -> CopulaFitResult:
    """Estimate by inverting Kendall's tau or Spearman's rho."""
    d = copula.dim
    ctor = type(copula)
    from_measure = ctor.from_tau if measure == "tau" else ctor.from_rho

    if isinstance(copula, EllipticalCopula) and copula.dispstr == "un" and d > 2:
        # Invert each pair separately, then repair the matrix.
        stat = _pairwise_measure(u, measure)
        rho = np.sin(np.pi * stat / 2.0) if measure == "tau" else 2.0 * np.sin(np.pi * stat / 6.0)
        sigma = nearest_correlation(p2P(rho, d))
        params = P2p(sigma)
        if isinstance(copula, StudentCopula):
            params = np.append(params, copula.df)
        fitted = copula.with_params(params)

        cov = None
        if estimate_variance:
            # Each correlation depends only on its own pairwise statistic, so
            # the Jacobian is diagonal: d(sin(pi t / 2))/dt for tau, and
            # d(2 sin(pi r / 6))/dr for rho.
            jac = np.diag(
                (np.pi / 2.0) * np.cos(np.pi * stat / 2.0)
                if measure == "tau"
                else (np.pi / 3.0) * np.cos(np.pi * stat / 6.0)
            )
            cov = var_inversion_multi(u, jac, measure=measure)
    else:
        # One-parameter case: average the pairwise statistics, then invert once.
        scalar_stat = float(np.mean(_pairwise_measure(u, measure)))
        try:
            fitted = from_measure(scalar_stat, dim=d)
        except (ValueError, NotImplementedError) as exc:
            raise ValueError(
                f"cannot invert {measure} = {scalar_stat:.4f} for the {copula.name} "
                f"family in dimension {d}: {exc}"
            ) from exc
        params = fitted.params

        cov = None
        if estimate_variance and d == 2 and params.size == 1:
            # Delta method needs g'(stat); differentiate the inverse map.
            h = 1e-5
            try:
                hi = float(np.atleast_1d(from_measure(scalar_stat + h, dim=d).params)[0])
                lo = float(np.atleast_1d(from_measure(scalar_stat - h, dim=d).params)[0])
                cov = var_itau(u, (hi - lo) / (2.0 * h), measure=measure)
            except (ValueError, NotImplementedError):
                cov = None

    loglik = _safe_loglik(fitted, u)
    return CopulaFitResult(
        copula=fitted,
        params=fitted.params,
        param_names=fitted.param_names,
        loglik=loglik,
        n_obs=u.shape[0],
        method=f"i{measure}",
        cov_params=cov,
        converged=True,
        message="closed-form inversion",
    )


def _safe_loglik(copula: Copula, u: NDArray[np.float64]) -> float:
    """Log-likelihood, tolerating families without a density."""
    try:
        return float(np.sum(copula.logpdf(u)))
    except NotImplementedError:
        return float("nan")


# ======================================================================
# Likelihood estimators
# ======================================================================


def _optimise(
    copula: Copula,
    u: NDArray[np.float64],
    start: NDArray[np.float64],
    optim_method: str | None,
) -> optimize.OptimizeResult:
    """Maximise the (pseudo-)likelihood over the free parameters."""
    bounds = copula.param_bounds
    free = copula.free
    fixed_values = copula.params

    def unpack(x: NDArray[np.float64]) -> NDArray[np.float64]:
        full = np.array(fixed_values, dtype=np.float64)
        full[free] = x
        return full

    def negative_loglik(x: NDArray[np.float64]) -> float:
        value = loglik_copula(unpack(x), u, copula)
        # Optimisers dislike -inf; a large finite penalty steers them back.
        return 1e10 if not np.isfinite(value) else -value

    # Nudge the bounds inward: most families are undefined at their endpoints.
    span = [
        (
            max(lo, -1e8) + 1e-6 if np.isfinite(lo) else -1e8,
            min(hi, 1e8) - 1e-6 if np.isfinite(hi) else 1e8,
        )
        for lo, hi in np.asarray(bounds)[free]
    ]
    x0 = np.clip(
        np.asarray(start, dtype=np.float64)[free], [s[0] for s in span], [s[1] for s in span]
    )

    method = optim_method or ("L-BFGS-B" if x0.size > 1 else "Nelder-Mead")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "Nelder-Mead":
            res = optimize.minimize(
                negative_loglik,
                x0,
                method="Nelder-Mead",
                bounds=span,
                options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 5000},
            )
        else:
            res = optimize.minimize(
                negative_loglik,
                x0,
                method=method,
                bounds=span,
                options={"maxiter": 5000},
            )
    res.x_full = unpack(res.x)
    return res


def _starting_value(copula: Copula, u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Default start: the inversion-of-tau estimate, as in R."""
    try:
        return _fit_by_inversion(copula, u, "tau", estimate_variance=False).params
    except (ValueError, NotImplementedError):
        # Fall back to the midpoint of each admissible interval.
        out = []
        for lo, hi in copula.param_bounds:
            lo_f = max(lo, -10.0)
            hi_f = min(hi, 10.0)
            out.append(0.5 * (lo_f + hi_f))
        return np.array(out)


# ======================================================================
# Public entry point
# ======================================================================


def fit(
    copula: Copula,
    data: ArrayLike,
    method: str = "mpl",
    *,
    start: ArrayLike | None = None,
    optim_method: str | None = None,
    estimate_variance: bool = True,
    ties_method: str = "average",
) -> CopulaFitResult:
    """Fit a copula to data.

    Parameters
    ----------
    copula : Copula
        Family to fit. Its parameter values are used only as a starting point;
        pass e.g. ``ClaytonCopula()`` with unspecified parameters.
    data : array_like
        ``(n, d)`` observations. Values already in ``(0, 1)`` are taken to be
        pseudo-observations; anything else is rank-transformed first.
    method : {"mpl", "ml", "itau", "irho", "itau.mpl"}
        Estimation method. See the module docstring.
    start : array_like, optional
        Starting parameters for the likelihood methods. Defaults to the
        inversion-of-tau estimate, as in R.
    optim_method : str, optional
        A ``scipy.optimize.minimize`` method name. Defaults to ``L-BFGS-B`` for
        multi-parameter problems and ``Nelder-Mead`` for one.
    estimate_variance : bool
        Whether to compute the asymptotic covariance matrix.
    ties_method : str
        Passed to :func:`~rcopula.dependence.pseudo_obs` when transforming.

    Returns
    -------
    CopulaFitResult

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, fit
    >>> u = ClaytonCopula(2.0).rvs(2000, random_state=0)
    >>> res = fit(ClaytonCopula(), u, method="mpl")
    >>> bool(abs(res.params[0] - 2.0) < 0.2)
    True
    >>> bool(res.bse[0] > 0)                     # standard errors, unlike elsewhere
    True

    ``"mpl"`` and ``"ml"`` agree on the estimate and differ on the uncertainty:

    >>> a = fit(ClaytonCopula(), u, method="mpl")
    >>> b = fit(ClaytonCopula(), u, method="ml")
    >>> bool(np.allclose(a.params, b.params, rtol=1e-6))
    True
    >>> bool(a.bse[0] != b.bse[0])
    True

    Inversion of Kendall's tau needs no optimiser at all:

    >>> res = fit(ClaytonCopula(), u, method="itau")
    >>> bool(abs(res.params[0] - 2.0) < 0.2)
    True
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    u = _as_pseudo_obs(data, ties_method)
    if u.shape[1] != copula.dim:
        raise ValueError(f"data has {u.shape[1]} column(s) but the copula has dim={copula.dim}")
    n = u.shape[0]

    if method in ("itau", "irho"):
        return _fit_by_inversion(copula, u, method[1:], estimate_variance)

    if method == "itau.mpl":
        return _fit_itau_mpl(copula, u, optim_method, estimate_variance)

    # -- mpl / ml ------------------------------------------------------
    x0 = np.asarray(start, dtype=np.float64) if start is not None else _starting_value(copula, u)
    res = _optimise(copula, u, x0, optim_method)
    fitted = copula.with_params(res.x_full)
    free = copula.free
    theta = res.x_full[free]

    cov = None
    if estimate_variance:

        def logpdf_at(uu: NDArray[np.float64], t: NDArray[np.float64]) -> NDArray[np.float64]:
            full = np.array(res.x_full, dtype=np.float64)
            full[free] = t
            try:
                return copula.with_params(full).logpdf(uu)
            except (ValueError, np.linalg.LinAlgError):
                return np.full(uu.shape[0], -np.inf)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cov = (
                var_mpl(logpdf_at, u, theta)
                if method == "mpl"
                else var_ml(lambda t: logpdf_at(u, t), theta, n)
            )

    return CopulaFitResult(
        copula=fitted,
        params=theta,
        param_names=tuple(np.array(copula.param_names)[free]),
        loglik=-float(res.fun),
        n_obs=n,
        method=method,
        cov_params=cov,
        converged=bool(res.success),
        message=str(res.message),
    )


def _fit_itau_mpl(
    copula: Copula,
    u: NDArray[np.float64],
    optim_method: str | None,
    estimate_variance: bool,
) -> CopulaFitResult:
    """Mashal-Zeevi two-stage estimator for the t copula.

    Correlations come from inverted Kendall's tau; only the degrees of freedom
    are then maximised over. Jointly optimising both is poorly conditioned in
    higher dimensions -- the likelihood is very flat in ``df`` once the
    correlations are even roughly right -- which is exactly the problem this
    avoids.
    """
    if not isinstance(copula, StudentCopula):
        raise ValueError(
            f"method='itau.mpl' applies to the Student-t copula only, got {copula.name}"
        )
    if copula.dispstr != "un":
        raise ValueError(
            f"method='itau.mpl' requires dispstr='un', as in R; got dispstr={copula.dispstr!r}"
        )

    d = copula.dim
    stat = _pairwise_measure(u, "tau")
    sigma = nearest_correlation(p2P(np.sin(np.pi * stat / 2.0), d))
    corr = P2p(sigma)

    def negative_loglik(x: NDArray[np.float64]) -> float:
        value = loglik_copula(np.append(corr, x[0]), u, copula)
        return 1e10 if not np.isfinite(value) else -value

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = optimize.minimize_scalar(
            lambda df: negative_loglik(np.array([df])),
            bounds=(0.2, 200.0),
            method="bounded",
            options={"xatol": 1e-8},
        )

    params = np.append(corr, res.x)
    fitted = copula.with_params(params)
    return CopulaFitResult(
        copula=fitted,
        params=params,
        param_names=fitted.param_names,
        loglik=-float(res.fun),
        n_obs=u.shape[0],
        method="itau.mpl",
        # R does not compute a variance here either: the two stages use
        # different information and combining them is not standard.
        cov_params=None,
        converged=bool(res.success),
        message="correlations by inverted tau; df by pseudo-likelihood",
    )
