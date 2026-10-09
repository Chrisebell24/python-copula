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
from scipy.stats import qmc

from rcopula.core.base import Copula
from rcopula.core.elliptical import EllipticalCopula, P2p, StudentCopula, p2P
from rcopula.core.extreme_value import TEVCopula
from rcopula.dependence import pseudo_obs
from rcopula.fit.results import CopulaFitResult
from rcopula.fit.variance import var_inversion_multi, var_itau, var_ml, var_mpl

__all__ = ["METHODS", "fit", "loglik_copula", "nearest_correlation"]

METHODS = ("mpl", "ml", "itau", "irho", "itau.mpl")


def nearest_correlation(
    matrix: ArrayLike, tol: float = 1e-10, max_iter: int = 200
) -> NDArray[np.float64]:
    """Repair a "correlation matrix" that is not a valid one, changing it as little as possible.

    Returns the nearest positive-definite matrix with ones on the diagonal,
    found by alternating projections (Higham, 2002). Use it when a matrix
    assembled entry by entry -- for example from pairwise Kendall's tau --
    cannot be used as a correlation matrix because it has a negative
    eigenvalue.

    Parameters
    ----------
    matrix : array_like of float, shape (d, d)
        Square matrix to repair. It is symmetrised first, as
        ``(matrix + matrix.T) / 2``.
    tol : float, default 1e-10
        Small positive number: the smallest eigenvalue allowed in the result,
        and the convergence tolerance.
    max_iter : int, default 200
        Maximum number of projection rounds (a positive integer).

    Returns
    -------
    ndarray of float, shape (d, d)
        Symmetric, positive-definite matrix with unit diagonal.

    Notes
    -----
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
    """Score how well a copula with the given parameters explains the data (higher is better).

    Returns the copula log-likelihood: the sum of the log-density over all
    rows of ``u``. This is R's ``loglikCopula``. Use it to compare parameter
    values by hand or to build your own optimiser; :func:`fit` uses it
    internally.

    Parameters
    ----------
    params : array_like of float, shape (p,)
        Full parameter vector to evaluate at, in the order of
        ``copula.param_names``.
    u : array_like of float, shape (n, d)
        Observations strictly inside the unit cube (pseudo-observations or
        copula samples). A 1-D input is treated as a single row.
    copula : Copula
        Supplies the family; its own parameter values are ignored.
    error : {"-inf", "raise"}, default "-inf"
        What to do when the parameters are inadmissible (outside the family's
        range, or giving a non-finite density): ``"-inf"`` returns
        ``-inf``; ``"raise"`` re-raises the underlying error.

    Returns
    -------
    float
        The log-likelihood, or ``-inf`` for inadmissible parameters when
        ``error="-inf"``.

    Raises
    ------
    ValueError, numpy.linalg.LinAlgError or NotImplementedError
        Only with ``error="raise"``, when the parameters are inadmissible or
        the family has no density.

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


def _names(copula: Copula, mask: NDArray[np.bool_]) -> tuple[str, ...]:
    """Parameter names selected by ``mask``, as plain ``str`` (never ``numpy.str_``)."""
    return tuple(str(name) for name, keep in zip(copula.param_names, mask, strict=True) if keep)


def _calibrate(copula: Copula, measure: str, value: float) -> Copula:
    """Re-tune ``copula`` to a target tau/rho, keeping everything inversion does not set.

    :meth:`Copula.calibrated` rebuilds the family from ``(value, dim)`` alone, so
    for a family with more than one parameter it would quietly reset the rest to
    the constructor defaults -- a t copula fitted at ``df=3`` came back at
    ``df=4``, and its ``"irho"`` inversion was computed at the wrong ``df``. The
    correlation structure and the degrees of freedom are passed through here.
    """
    if isinstance(copula, (EllipticalCopula, TEVCopula)):
        kwargs: dict[str, object] = {}
        if isinstance(copula, EllipticalCopula):
            kwargs["dispstr"] = copula.dispstr
        df = getattr(copula, "df", np.nan)
        if isinstance(copula, (StudentCopula, TEVCopula)) and np.isfinite(df):
            kwargs["df"] = float(df)
        factory = type(copula).from_tau if measure == "tau" else type(copula).from_rho
        return factory(value, dim=copula.dim, **kwargs)
    return copula.calibrated(measure, value)


def _determined_by_inversion(copula: Copula) -> NDArray[np.bool_]:
    """Which parameters an inversion of tau/rho actually estimates.

    The correlations of an elliptical copula, and otherwise the first (the
    dependence) parameter. Anything else -- a t copula's ``df``, say -- does not
    enter Kendall's tau at all and cannot be recovered from it.
    """
    mask = np.zeros(len(copula.param_names), dtype=bool)
    if isinstance(copula, EllipticalCopula):
        mask[: len(copula.params) - (1 if isinstance(copula, StudentCopula) else 0)] = True
    else:
        mask[0] = True
    return mask


def _carried_value(copula: Copula, fitted: Copula, index: int) -> float:
    """Value of a parameter inversion does not estimate: the input's, else the default."""
    value = float(copula.params[index])
    return value if np.isfinite(value) else float(fitted.params[index])


def _pairwise_correlations(
    copula: EllipticalCopula, stat: NDArray[np.float64], measure: str
) -> NDArray[np.float64]:
    r"""Invert pairwise tau or rho into correlations, pair by pair.

    Kendall's tau gives :math:`\sin(\pi\tau/2)` for every elliptical copula.
    Spearman's rho gives :math:`2\sin(\pi\rho_S/6)` only for the *Gaussian*;
    a t copula's rho is inverted numerically at its ``df``, exactly as
    :meth:`StudentCopula.from_rho` does in two dimensions. (The unstructured
    branch used the Gaussian relation for t copulas too, so ``"irho"`` on a 3-d
    t copula with ``df=2.5`` converged to 0.677 for a true correlation of 0.7.)
    """
    if measure == "tau":
        return np.sin(np.pi * stat / 2.0)
    if isinstance(copula, StudentCopula):
        df = float(copula.df)
        return np.array([float(StudentCopula.from_rho(float(s), df=df).params[0]) for s in stat])
    return 2.0 * np.sin(np.pi * stat / 6.0)


def _dcorrelation_dstat(
    copula: EllipticalCopula, correlation: NDArray[np.float64], measure: str
) -> NDArray[np.float64]:
    """Derivative of the inverse map (correlation as a function of tau/rho).

    Evaluated *at a correlation*, so it serves both the pairwise inversion
    (evaluated at each pair's own inverse) and R's convention for structured
    matrices (evaluated at the fitted correlation of each pair).
    """
    r = np.asarray(correlation, dtype=np.float64)
    if measure == "tau":
        # tau = (2 / pi) arcsin(r)  =>  dr / dtau = (pi / 2) sqrt(1 - r^2)
        return (np.pi / 2.0) * np.sqrt(np.clip(1.0 - r**2, 0.0, None))
    if isinstance(copula, StudentCopula):
        from rcopula.core.elliptical import _student_rho

        df, h = float(copula.df), 1e-5
        slope = np.array(
            [
                (
                    _student_rho(min(x + h, 1.0 - 1e-12), df)
                    - _student_rho(max(x - h, -1.0 + 1e-12), df)
                )
                / (min(x + h, 1.0 - 1e-12) - max(x - h, -1.0 + 1e-12))
                for x in r
            ]
        )
        return 1.0 / slope
    # rho_S = (6 / pi) arcsin(r / 2)  =>  dr / drho_S = (pi / 3) sqrt(1 - r^2 / 4)
    return (np.pi / 3.0) * np.sqrt(1.0 - r**2 / 4.0)


def _inverted_correlations(
    copula: EllipticalCopula, u: NDArray[np.float64], measure: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pairwise-inverted correlations of an unstructured elliptical copula.

    Returns the correlation vector and the pairwise statistics it came from.
    Correlations pinned with ``fix_params`` keep their values. The assembled
    matrix is repaired to the nearest correlation matrix (as R does with
    ``nearPD``); with pinned entries the repair is applied only when needed, and
    the pinned entries are then restored -- if the result is still not positive
    definite the pinned values are incompatible with the data and an error says
    so rather than silently moving them.
    """
    d = copula.dim
    stat = _pairwise_measure(u, measure)
    rho = _pairwise_correlations(copula, stat, measure)
    n_corr = rho.size
    corr_free = np.asarray(copula.free[:n_corr], dtype=bool)
    if corr_free.all():
        return P2p(nearest_correlation(p2P(rho, d))), stat

    pinned = np.asarray(copula.params[:n_corr], dtype=np.float64)
    rho = np.where(corr_free, rho, pinned)
    sigma = p2P(rho, d)
    if np.linalg.eigvalsh(sigma).min() <= 0.0:
        rho = np.where(corr_free, P2p(nearest_correlation(sigma)), pinned)
        if np.linalg.eigvalsh(p2P(rho, d)).min() <= 0.0:
            raise ValueError(
                f"cannot invert {measure} for the {copula.name} copula: the correlations "
                "pinned with fix_params cannot be completed to a positive-definite "
                "correlation matrix with the inverted free ones"
            )
    return rho, stat


def _correlation_cov(
    copula: EllipticalCopula, u: NDArray[np.float64], stat: NDArray[np.float64], measure: str
) -> NDArray[np.float64] | None:
    """Delta-method covariance of pairwise-inverted correlations (all pairs)."""
    # Each correlation depends only on its own pairwise statistic, so the
    # Jacobian is diagonal, evaluated at each pair's own inverse.
    jac = np.diag(
        _dcorrelation_dstat(copula, _pairwise_correlations(copula, stat, measure), measure)
    )
    return var_inversion_multi(u, jac, measure=measure)


def _structured_correlations(
    copula: EllipticalCopula, u: NDArray[np.float64], measure: str, estimate_variance: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Inversion estimate of a Toeplitz or AR(1) correlation structure, as R does it.

    Each pair's tau/rho is inverted to a correlation, the matrix is repaired to
    the nearest correlation matrix (R's ``nearPD``), and the structure is then
    fitted to those pairwise correlations by least squares (R's ``getXmat``
    regression):

    * ``"toep"`` -- the lag-``k`` parameter is the mean of the pairwise
      correlations ``|i - j| = k``;
    * ``"ar1"`` -- ``log r_ij = |i - j| log rho`` through the origin, i.e.
      ``rho = exp(sum(lag * log r) / sum(lag^2))``. When some pairwise
      correlation is not positive the logarithm does not exist (R fails there);
      ``rho`` is then the least-squares solution of ``r_ij = rho^|i - j|``.

    Before this existed the structured cases fell through to the
    one-parameter path, which averages *all* pairwise statistics: ``"toep"``
    came back with every lag equal, and ``"ar1"`` with the average over all
    lags -- 0.40 for a true lag-1 correlation of 0.6 in five dimensions.

    Returns the parameters for the free correlation entries' positions (fixed
    ones keep their values) and their delta-method covariance over *all*
    correlation parameters, with R's convention of evaluating the derivative
    at the fitted correlations.
    """
    d = copula.dim
    stat = _pairwise_measure(u, measure)
    icor = P2p(nearest_correlation(p2P(_pairwise_correlations(copula, stat, measure), d)))
    lags = P2p(np.abs(np.subtract.outer(np.arange(d), np.arange(d)))).astype(int)
    n_corr = 1 if copula.dispstr == "ar1" else d - 1
    corr_free = np.asarray(copula.free[:n_corr], dtype=bool)
    params = np.array(copula.params[:n_corr], dtype=np.float64)

    if copula.dispstr == "toep":
        # (n_corr, n_pairs) averaging matrix: row k averages the lag-(k+1) pairs.
        weights = np.array([(lags == k + 1) / np.sum(lags == k + 1) for k in range(n_corr)])
        params = np.where(corr_free, weights @ icor, params)
        fitted_pairs = params[lags - 1]
        jac = weights * _dcorrelation_dstat(copula, fitted_pairs, measure)[None, :]
    else:
        lag = lags.astype(np.float64)
        if np.all(icor > 0.0):
            rho = float(np.exp(np.sum(lag * np.log(icor)) / np.sum(lag**2)))
            fitted_pairs = rho**lag
            # d rho / d r_ij = rho * lag_ij / (r_ij * sum(lag^2)), at the fitted r_ij.
            dparam = rho * lag / (fitted_pairs * np.sum(lag**2))
        else:
            fit_ls = optimize.minimize_scalar(
                lambda r: float(np.sum((icor - r**lag) ** 2)),
                bounds=(-1.0 + 1e-10, 1.0 - 1e-10),
                method="bounded",
                options={"xatol": 1e-13},
            )
            rho = float(fit_ls.x)
            fitted_pairs = rho**lag
            # Implicit-function derivative of the least-squares root.
            grad = lag * rho ** (lag - 1.0)
            curv = np.sum(grad**2 - (icor - fitted_pairs) * lag * (lag - 1.0) * rho ** (lag - 2.0))
            dparam = grad / curv
        params = np.array([rho])
        jac = (dparam * _dcorrelation_dstat(copula, fitted_pairs, measure))[None, :]

    cov = var_inversion_multi(u, jac, measure=measure) if estimate_variance else None
    return params, cov


def _fit_by_inversion(
    copula: Copula,
    u: NDArray[np.float64],
    measure: str,
    estimate_variance: bool,
    warn: bool = True,
) -> CopulaFitResult:
    """Estimate by inverting Kendall's tau or Spearman's rho.

    Only the parameters that tau/rho determine are estimated (see
    :func:`_determined_by_inversion`); parameters pinned with ``fix_params``
    keep their values; any other free parameter (a t copula's ``df``) is held
    at its current value with a warning, and is not reported as an estimate.
    """
    d = copula.dim
    free = np.asarray(copula.free, dtype=bool)
    determined = _determined_by_inversion(copula)
    full = np.array(copula.params, dtype=np.float64)

    cov_all: NDArray[np.float64] | None = None
    if isinstance(copula, EllipticalCopula) and copula.dispstr == "un" and d > 2:
        # Invert each pair separately, then repair the matrix.
        n_corr = int(determined.sum())
        if not free[:n_corr].any():
            raise ValueError(
                f"cannot fit {_names(copula, free & ~determined)} by {measure} inversion: "
                "it estimates only the correlations, and they are all fixed; use "
                "method='mpl'"
            )
        corr, stat = _inverted_correlations(copula, u, measure)
        full[:n_corr] = corr
        if estimate_variance:
            cov_all = _correlation_cov(copula, u, stat, measure)
        reference: Copula = copula
    elif isinstance(copula, EllipticalCopula) and copula.dispstr in ("toep", "ar1") and d > 2:
        n_corr = int(determined.sum())
        if not free[:n_corr].any():
            raise ValueError(
                f"cannot fit {_names(copula, free & ~determined)} by {measure} inversion: "
                "it estimates only the correlations, and they are all fixed; use "
                "method='mpl'"
            )
        full[:n_corr], cov_all = _structured_correlations(copula, u, measure, estimate_variance)
        try:
            copula.with_params(full)
        except ValueError as exc:
            raise ValueError(
                f"inverting {measure} pair by pair gives a {copula.dispstr!r} correlation "
                f"structure that is not positive definite ({exc}); use method='mpl'"
            ) from exc
        reference = copula
    else:
        # One dependence parameter: average the pairwise statistics, invert once.
        if not free[determined].any():
            raise ValueError(
                f"cannot fit {_names(copula, free & ~determined)} by {measure} inversion: "
                f"{measure} determines only {_names(copula, determined)}, which is fixed; "
                "use method='mpl'"
            )
        scalar_stat = float(np.mean(_pairwise_measure(u, measure)))
        try:
            reference = _calibrate(copula, measure, scalar_stat)
        except (ValueError, NotImplementedError) as exc:
            raise ValueError(
                f"cannot invert {measure} = {scalar_stat:.4f} for the {copula.name} "
                f"family in dimension {d}: {exc}"
            ) from exc
        full[determined] = np.asarray(reference.params, dtype=np.float64)[determined]

        if estimate_variance and determined.sum() == 1:
            # Delta method needs g'(stat); differentiate the inverse map.
            index = int(np.flatnonzero(determined)[0])
            h = 1e-5
            try:
                hi = float(_calibrate(copula, measure, scalar_stat + h).params[index])
                lo = float(_calibrate(copula, measure, scalar_stat - h).params[index])
                cov_all = var_itau(u, (hi - lo) / (2.0 * h), measure=measure)
            except (ValueError, NotImplementedError):
                cov_all = None

    # Parameters tau/rho cannot see: keep their current value (or, when that is
    # unset, the family default the calibration used).
    for position in np.flatnonzero(~determined):
        full[position] = _carried_value(copula, reference, int(position))
    held = free & ~determined
    if warn and held.any():
        values = ", ".join(
            f"{name}={full[i]:g}" for i, name in enumerate(copula.param_names) if held[i]
        )
        warnings.warn(
            f"method='i{measure}' cannot estimate {_names(copula, held)}: {measure} does "
            f"not depend on it. Held at {values} and not reported as an estimate. Pin "
            "it with fix_params to silence this, or use method='mpl' (or 'itau.mpl' "
            "for a t copula) to estimate it.",
            UserWarning,
            stacklevel=3,
        )

    estimated = free & determined
    cov = None
    if cov_all is not None:
        keep = estimated[determined]
        cov = np.asarray(cov_all, dtype=np.float64)[np.ix_(keep, keep)]

    fitted = copula.with_params(full)
    loglik = _safe_loglik(fitted, u)
    return CopulaFitResult(
        copula=fitted,
        params=full[estimated],
        param_names=_names(copula, estimated),
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
        # Optimisers dislike -inf, so infeasible parameters get a large finite
        # penalty. It is deliberately flat: a sloped one carries a gradient of
        # order 1e10, which wrecks a quasi-Newton line search far more
        # thoroughly than the plateau it was meant to fix. The plateau is
        # handled below instead, by not trusting a single optimiser.
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

    def run(which: str, guess: NDArray[np.float64]) -> optimize.OptimizeResult:
        if which == "Nelder-Mead":
            return optimize.minimize(
                negative_loglik,
                guess,
                method="Nelder-Mead",
                bounds=span,
                options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 5000},
            )
        # L-BFGS-B's default stopping rule (relative change in f below 2.2e-9)
        # stops a log-likelihood of a few hundred ~1e-6 short of its maximum,
        # i.e. 1e-4 relative error in the estimates of a 6-10 parameter
        # unstructured fit; R's BFGS gets further. Tightened to the precision
        # the finite-difference gradient supports.
        options: dict[str, float | int] = {"maxiter": 5000}
        if which == "L-BFGS-B":
            options.update(ftol=1e-13, gtol=1e-9)
        return optimize.minimize(negative_loglik, guess, method=which, bounds=span, options=options)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run(method, x0)
        if optim_method is None and method != "Nelder-Mead":
            # Two ways a multi-parameter fit ends somewhere that is not the
            # maximum, both seen in practice:
            #
            #   * the quasi-Newton method steps onto the infeasible plateau,
            #     where the finite-difference gradient is exactly zero, and
            #     converges on the spot. This is what returned theta = 77
            #     instead of 3 for a Khoudraji fit on Linux and Windows while
            #     passing on macOS -- the platforms' BLAS simply took different
            #     paths into the same trap.
            #   * the starting value sits in a basin whose only local maximum is
            #     degenerate, and every optimiser walks obediently into it.
            #
            # A spread of trial starts detects both for the price of a few
            # likelihood evaluations: the first scores the 1e10 penalty, the
            # second scores about zero, and in each case a point chosen with no
            # search at all beats the converged answer -- which is the signal.
            # Re-running is then rare, so the common case pays only the
            # evaluations.
            lower = np.array([s[0] for s in span])
            upper = np.array([s[1] for s in span])
            spread = np.clip(_candidate_starts(copula, free), lower, upper)
            best = min(spread, key=negative_loglik)
            if negative_loglik(best) < float(res.fun):
                attempt = run("Nelder-Mead", best)
                if float(attempt.fun) < float(res.fun):
                    res = attempt
    res.x_full = unpack(res.x)
    return res


def _candidate_starts(copula: Copula, free: NDArray[np.bool_]) -> NDArray[np.float64]:
    """A spread of alternative starting values over the parameters' working box.

    Two details matter. The declared bounds are useless here -- half the
    families are unbounded above, and a point a tenth of the way to infinity is
    not a starting value -- so an infinite side becomes ten units past the
    finite one, which covers the range these families are actually used in.

    And the points must vary the parameters *independently*. Moving them all
    together along a diagonal is what a naive spread does, and it cannot find a
    Khoudraji copula whose right answer pairs a small theta with middling
    shapes: the diagonal offers small-with-small and large-with-large and
    nothing else. A scrambled Sobol set covers the box instead, and is
    deterministic at a fixed seed.
    """
    box = []
    for lo, hi in np.asarray(copula.param_bounds)[free]:
        low = float(lo) if np.isfinite(lo) else (float(hi) - 10.0)
        high = float(hi) if np.isfinite(hi) else (low + 10.0)
        box.append((low, high))
    span = np.asarray(box, dtype=np.float64)
    points = qmc.Sobol(d=span.shape[0], scramble=True, seed=0).random(16)
    return span[:, 0] + points * (span[:, 1] - span[:, 0])


def _starting_value(copula: Copula, u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Default start: the inversion-of-tau estimate, as in R."""
    try:
        return np.asarray(
            _fit_by_inversion(copula, u, "tau", estimate_variance=False, warn=False).copula.params,
            dtype=np.float64,
        )
    except (ValueError, NotImplementedError, np.linalg.LinAlgError):
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
    """Estimate a copula's parameters from data.

    Give it a copula family (for example ``ClaytonCopula()``) and a table of
    observations; it returns the best-fitting parameters together with
    standard errors, a log-likelihood and AIC/BIC. Your data can be raw
    values or already-uniform pseudo-observations -- raw values are turned
    into ranks first, so the margins do not matter.

    Parameters
    ----------
    copula : Copula
        Family to fit. Its parameter values are used only as a starting point;
        pass e.g. ``ClaytonCopula()`` with unspecified parameters. Parameters
        pinned with ``fix_params`` are held fixed.
    data : array_like of float or pandas.DataFrame, shape (n, d)
        Observations, one row per observation and one column per variable;
        ``d`` must equal ``copula.dim``. If every value is strictly inside
        ``(0, 1)`` the data are taken to be pseudo-observations as they are;
        otherwise each column is rank-transformed first. A 1-D input is
        treated as a single column.
    method : {"mpl", "ml", "itau", "irho", "itau.mpl"}, default "mpl"
        Estimation method:

        * ``"mpl"`` -- maximum pseudo-likelihood; the usual choice.
        * ``"ml"`` -- same estimate, standard errors that assume the margins
          are known.
        * ``"itau"`` / ``"irho"`` -- invert Kendall's tau / Spearman's rho;
          fast, no optimiser.
        * ``"itau.mpl"`` -- Student-t copula only: correlations from tau,
          degrees of freedom by pseudo-likelihood.

        See the module docstring for details.
    start : array_like of float, shape (p,), or None, default None
        Starting parameters (full vector, in the order of
        ``copula.param_names``; entries of fixed parameters are ignored).
        For ``"mpl"`` and ``"ml"``, ``None`` uses the inversion-of-tau
        estimate, as in R. For ``"itau.mpl"`` only the last element -- the
        starting ``df`` -- is used, since the correlations come from tau
        (a scalar is accepted too). Ignored by ``"itau"`` and ``"irho"``,
        which have nothing to start.
    optim_method : str or None, default None
        A ``scipy.optimize.minimize`` method name, e.g. ``"BFGS"``. ``None``
        uses ``L-BFGS-B`` for multi-parameter problems and ``Nelder-Mead``
        for one. Used by ``"mpl"`` and ``"ml"``, and by ``"itau.mpl"`` for
        its search over ``df`` (where ``None`` together with ``start=None``
        means R's bounded Brent search on ``(0.2, 200)``; giving either one
        switches to ``minimize`` with those bounds, ``Nelder-Mead`` by
        default). Ignored by ``"itau"`` and ``"irho"``.
    estimate_variance : bool, default True
        Whether to compute the asymptotic covariance matrix (and therefore
        standard errors). Set to ``False`` to save time. For ``"itau.mpl"``
        the correlations get their inversion covariance and the ``df``
        entries are ``nan`` (see Notes).
    ties_method : {"average", "min", "max", "dense", "ordinal", "random"}, default "average"
        How tied values are ranked; passed to
        :func:`~rcopula.dependence.pseudo_obs` when transforming raw data.

    Returns
    -------
    CopulaFitResult
        Fitted copula, estimates of the free parameters, standard errors
        (``bse``), log-likelihood, AIC/BIC and a printable ``summary()``.
        Whatever the method, ``params`` and ``param_names`` (plain ``str``)
        list the *estimated* parameters only -- those free in
        ``copula.free`` -- as R's ``fitCopula`` does; parameters pinned with
        ``fix_params`` appear only in ``result.copula.params``, which is
        always the full vector.

    Raises
    ------
    ValueError
        If ``method`` is not one of the five listed; if the number of data
        columns differs from ``copula.dim``; if ``"itau"``/``"irho"`` cannot
        invert the sample measure for this family (e.g. negative tau for a
        Gumbel copula), or the parameter tau/rho determines is fixed while
        another one is free; if correlations pinned with ``fix_params``
        cannot be completed to a valid correlation matrix; or if
        ``"itau.mpl"`` is used with anything other than a Student-t copula
        with ``dispstr="un"``.

    Warns
    -----
    UserWarning
        For ``"itau"``/``"irho"`` when a free parameter does not enter
        tau/rho at all -- a t copula's ``df`` -- and so cannot be estimated
        by inversion. It is held at its current value (``df=4`` by default)
        and left out of ``params``. Pin it with ``fix_params`` (or
        ``df_fixed=True``) to silence the warning, or use ``"itau.mpl"``.

    See Also
    --------
    rcopula.fit_joint : Fit the margins and the copula together.
    rcopula.select_copula : Fit several families and rank them.

    Notes
    -----
    ``"mpl"`` and ``"ml"`` produce the same point estimate and differ only
    in the variance. Every method respects parameters pinned with
    ``fix_params``: inversion keeps pinned correlations (repairing the matrix
    around them when needed) and ``"itau.mpl"`` skips the ``df`` search when
    ``df`` is pinned. For ``"itau.mpl"`` the correlations' standard errors are
    those of the inversion estimator -- the correlations *are* the ``"itau"``
    estimates -- while the ``df`` row and column of ``cov_params`` are
    ``nan``: a variance for the second stage would have to account for the
    first, which is not standard, and R reports no variance for this method
    at all. A copula with no free parameters is not optimised; its
    log-likelihood is still reported so it can take part in AIC/BIC
    comparisons.

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

    from rcopula.factor import FactorCopula, _fit_as_result

    if isinstance(copula, FactorCopula):
        # Thousands of loadings: generic optimisation is hopeless, and the
        # structured estimator is both fast and accurate (see fit_factor).
        return _fit_as_result(copula, u, method)

    if not np.any(copula.free):
        # Nothing to estimate -- either a parameter-free family such as
        # independence, or every parameter pinned by fix_params. There is still
        # a log-likelihood to report, and returning it lets such copulas take
        # part in AIC/BIC comparisons instead of raising inside an optimiser
        # that was handed an empty parameter vector.
        return CopulaFitResult(
            copula=copula,
            params=np.empty(0),
            param_names=(),
            loglik=_safe_loglik(copula, u),
            n_obs=n,
            method=method,
            cov_params=np.empty((0, 0)),
            converged=True,
            message="no free parameters to estimate",
        )

    if method in ("itau", "irho"):
        return _fit_by_inversion(copula, u, method[1:], estimate_variance)

    if method == "itau.mpl":
        return _fit_itau_mpl(copula, u, optim_method, estimate_variance, start)

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
        param_names=_names(copula, np.asarray(free, dtype=bool)),
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
    start: ArrayLike | None = None,
) -> CopulaFitResult:
    """Mashal-Zeevi two-stage estimator for the t copula.

    Correlations come from inverted Kendall's tau; only the degrees of freedom
    are then maximised over. Jointly optimising both is poorly conditioned in
    higher dimensions -- the likelihood is very flat in ``df`` once the
    correlations are even roughly right -- which is exactly the problem this
    avoids.

    Parameters pinned with ``fix_params`` are respected: pinned correlations
    keep their values, and a pinned ``df`` is not optimised at all. ``start``
    supplies the starting ``df`` (its last element) and ``optim_method`` picks
    the optimiser for it; with neither, a bounded Brent search over
    ``(0.2, 200)`` is used, as R does. With ``estimate_variance`` the
    correlations get their inversion (delta-method) covariance -- they *are*
    the ``itau`` estimates, so that variance is exact to first order -- while
    the ``df`` entries are ``nan``: the second stage's variance would have to
    account for the first, which is not standard (R reports no variance here at
    all).
    """
    if not isinstance(copula, StudentCopula):
        raise ValueError(
            f"method='itau.mpl' applies to the Student-t copula only, got {copula.name}"
        )
    if copula.dispstr != "un":
        raise ValueError(
            f"method='itau.mpl' requires dispstr='un', as in R; got dispstr={copula.dispstr!r}"
        )

    free = np.asarray(copula.free, dtype=bool)
    n_corr = free.size - 1
    corr, stat = _inverted_correlations(copula, u, "tau")

    def negative_loglik(df: float) -> float:
        value = loglik_copula(np.append(corr, df), u, copula)
        return 1e10 if not np.isfinite(value) else -value

    lower, upper = 0.2, 200.0
    if not free[-1]:
        df_hat = float(copula.df)
        neg = negative_loglik(df_hat)
        converged, message = True, "correlations by inverted tau; df fixed"
    elif start is None and optim_method is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = optimize.minimize_scalar(
                negative_loglik, bounds=(lower, upper), method="bounded", options={"xatol": 1e-8}
            )
        df_hat, neg = float(res.x), float(res.fun)
        converged = bool(res.success)
        message = "correlations by inverted tau; df by pseudo-likelihood"
    else:
        if start is not None:
            df0 = float(np.atleast_1d(np.asarray(start, dtype=np.float64))[-1])
        else:
            df0 = float(copula.df) if np.isfinite(copula.df) else 4.0
        df0 = float(np.clip(df0, lower, upper))
        method = optim_method or "Nelder-Mead"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = optimize.minimize(
                lambda x: negative_loglik(float(x[0])),
                np.array([df0]),
                method=method,
                bounds=[(lower, upper)],
            )
        df_hat, neg = float(np.atleast_1d(res.x)[0]), float(res.fun)
        converged = bool(res.success)
        message = f"correlations by inverted tau; df by pseudo-likelihood ({method})"

    full = np.append(corr, df_hat)
    fitted = copula.with_params(full)

    cov = None
    if estimate_variance:
        corr_cov = _correlation_cov(copula, u, stat, "tau")
        cov_full = np.full((free.size, free.size), np.nan)
        if corr_cov is not None:
            cov_full[:n_corr, :n_corr] = corr_cov
        cov = cov_full[np.ix_(free, free)]

    return CopulaFitResult(
        copula=fitted,
        params=full[free],
        param_names=_names(copula, free),
        loglik=-neg,
        n_obs=u.shape[0],
        method="itau.mpl",
        cov_params=cov,
        converged=converged,
        message=message,
    )
