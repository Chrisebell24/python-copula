r"""Bootstrap confidence intervals for dependence measures and fitted parameters.

R's ``copula`` reports **asymptotic** standard errors and nothing else. Those are
excellent when they apply and quietly wrong when they do not, and the cases where
they do not are the ones people care about:

* a statistic bounded at 1, near 1 -- a symmetric interval runs off the end of
  the parameter space;
* tail dependence, which is a *limit* estimated from a handful of corner points,
  so its sampling distribution is skewed at any realistic sample size;
* anything at n = 50, where the asymptotics have not arrived.

A bootstrap makes no distributional assumption and produces an interval that
respects the boundary. The cost is compute, which is why every function here
takes ``n_jobs``.

**Resampling rows, not columns.** Copula data is multivariate and the dependence
*is* the object of study, so a resample must take whole rows. Resampling each
coordinate independently would destroy exactly what is being measured -- and
would produce beautifully tight intervals around zero.

============================  ================================================
:func:`bootstrap`             The general machine: any statistic of a matrix.
:func:`bootstrap_measure`     tau, rho, beta or lambda, with one call.
:func:`bootstrap_fit`         Intervals for a fitted copula's parameters.
:class:`BootstrapResult`      Estimate, interval, standard error, replicates.
============================  ================================================

Three interval types:

``percentile``
    The replicate quantiles. Simple, and biased when the statistic's
    distribution is not centred on the estimate.
``basic``
    Reflects the replicates through the estimate. Corrects location bias, and
    can run outside the parameter space.
``bca``
    Bias-corrected and accelerated (Efron 1987): adjusts for both median bias
    and for the statistic's variance changing with its value. Second-order
    accurate and transformation-respecting, so it stays inside the parameter
    space. The default, and worth the extra jackknife pass.

Examples
--------
>>> import rcopula as rc
>>> from rcopula.bootstrap import bootstrap_measure
>>> u = rc.ClaytonCopula(2.0).rvs(400, random_state=0)
>>> result = bootstrap_measure(u, "tau", n_resamples=199, random_state=0)
>>> lower, upper = result.confidence_interval
>>> bool(lower < 0.5 < upper)          # the true tau of Clayton(2)
True

References
----------
Efron, B. (1987). Better bootstrap confidence intervals.
    *J. American Statistical Association* 82(397), 171-185.
    The BCa interval.
Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*.
    Chapman & Hall.
Davison, A. C. and Hinkley, D. V. (1997). *Bootstrap Methods and their
    Application*. Cambridge University Press.
Genest, C., Ghoudi, K. and Rivest, L.-P. (1995). A semiparametric estimation
    procedure of dependence parameters in multivariate families of
    distributions. *Biometrika* 82(3), 543-552.
    The asymptotic variance this complements rather than replaces.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from rcopula.core.base import Copula

__all__ = ["BootstrapResult", "bootstrap", "bootstrap_fit", "bootstrap_measure"]

Method = Literal["bca", "percentile", "basic"]

#: Replicates that raise are dropped rather than crashing the run: a resample
#: can be degenerate (every row identical in one coordinate) and an estimator is
#: entitled to refuse it. More than this fraction failing means something else
#: is wrong, and the result says so.
_MAX_FAILURE_FRACTION = 0.1


@dataclass
class BootstrapResult:
    """The outcome of a bootstrap.

    Attributes
    ----------
    estimate : float or ndarray
        The statistic on the original data.
    confidence_interval : tuple
        ``(lower, upper)``. Scalars for a scalar statistic, arrays otherwise.
    standard_error : float or ndarray
        Standard deviation of the replicates. Note this is a bootstrap estimate
        of the standard error, not the asymptotic one.
    replicates : ndarray, shape (n_resamples, ...)
        Every replicate, kept so the distribution can be plotted -- which is
        usually more informative than the interval.
    method : str
    level : float
    n_failed : int
        Resamples the statistic refused.
    """

    estimate: Any
    confidence_interval: tuple[Any, Any]
    standard_error: Any
    replicates: NDArray[np.float64] = field(repr=False)
    method: str
    level: float
    n_failed: int = 0

    @property
    def bias(self) -> Any:
        """Bootstrap estimate of bias: mean of the replicates minus the estimate.

        A bias comparable to the standard error is a warning that the statistic
        is not well behaved at this sample size, whatever the interval says.
        """
        return np.mean(self.replicates, axis=0) - self.estimate

    def summary(self) -> str:
        """A printable report."""
        estimate = np.atleast_1d(np.asarray(self.estimate, dtype=float))
        lower = np.atleast_1d(np.asarray(self.confidence_interval[0], dtype=float))
        upper = np.atleast_1d(np.asarray(self.confidence_interval[1], dtype=float))
        error = np.atleast_1d(np.asarray(self.standard_error, dtype=float))
        bias = np.atleast_1d(np.asarray(self.bias, dtype=float))

        percent = round(100 * self.level)
        lines = [
            f"Bootstrap ({self.method}, {self.replicates.shape[0]} resamples)",
            "=" * 68,
            f"  {'':>4}{'estimate':>13}{'SE':>12}{'bias':>12}"
            f"{f'{percent}% lower':>14}{f'{percent}% upper':>14}",
        ]
        for j in range(estimate.size):
            lines.append(
                f"  {j:>4}{estimate[j]:>13.6f}{error[j]:>12.6f}{bias[j]:>+12.6f}"
                f"{lower[j]:>14.6f}{upper[j]:>14.6f}"
            )
        if self.n_failed:
            lines += ["", f"  {self.n_failed} resample(s) were refused by the statistic"]
        return "\n".join(lines)


def _percentile_interval(
    replicates: NDArray[np.float64], level: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    alpha = 0.5 * (1.0 - level)
    return (
        np.percentile(replicates, 100 * alpha, axis=0),
        np.percentile(replicates, 100 * (1 - alpha), axis=0),
    )


def _basic_interval(
    replicates: NDArray[np.float64], estimate: NDArray[np.float64], level: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower, upper = _percentile_interval(replicates, level)
    # Reflect through the estimate: the *upper* percentile gives the lower limit.
    return 2 * estimate - upper, 2 * estimate - lower


def _bca_interval(
    replicates: NDArray[np.float64],
    estimate: NDArray[np.float64],
    jackknife: NDArray[np.float64],
    level: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Efron's bias-corrected and accelerated interval.

    ``z0`` corrects for the replicates not being centred on the estimate; ``a``
    corrects for the statistic's variance changing with its value, and comes
    from the skewness of the jackknife values.
    """
    n_resamples = replicates.shape[0]
    proportion = np.mean(replicates < estimate, axis=0)
    # A proportion of exactly 0 or 1 sends z0 to infinity. It means every
    # replicate fell on one side, which is a real signal but not one that can be
    # turned into a finite correction; nudge it to the smallest resolvable value.
    proportion = np.clip(proportion, 1.0 / (2 * n_resamples), 1.0 - 1.0 / (2 * n_resamples))
    bias_correction = stats.norm.ppf(proportion)

    centred = np.mean(jackknife, axis=0) - jackknife
    numerator = np.sum(centred**3, axis=0)
    denominator = 6.0 * np.sum(centred**2, axis=0) ** 1.5
    acceleration = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )

    alpha = 0.5 * (1.0 - level)
    quantiles = []
    for tail in (alpha, 1.0 - alpha):
        z = stats.norm.ppf(tail)
        adjusted = bias_correction + (bias_correction + z) / (
            1.0 - acceleration * (bias_correction + z)
        )
        quantiles.append(np.clip(stats.norm.cdf(adjusted), 0.0, 1.0))

    lower = np.empty_like(np.atleast_1d(estimate))
    upper = np.empty_like(lower)
    flat = replicates.reshape(n_resamples, -1)
    for j in range(flat.shape[1]):
        lower[j] = np.percentile(flat[:, j], 100 * np.atleast_1d(quantiles[0])[j])
        upper[j] = np.percentile(flat[:, j], 100 * np.atleast_1d(quantiles[1])[j])
    return lower, upper


def _evaluate(statistic: Callable[[NDArray], Any], sample: NDArray) -> Any:
    try:
        value = statistic(sample)
    except Exception:
        # A refused resample is data, not a bug -- see _MAX_FAILURE_FRACTION.
        return None
    value = np.asarray(value, dtype=float)
    return None if not np.all(np.isfinite(value)) else value


def _one_resample(args: tuple[Callable[[NDArray], Any], NDArray, int]) -> Any:
    """Top level so it can be pickled for a process pool."""
    statistic, x, seed = args
    rng = np.random.default_rng(seed)
    index = rng.integers(0, x.shape[0], size=x.shape[0])
    return _evaluate(statistic, x[index])


def bootstrap(
    x: ArrayLike,
    statistic: Callable[[NDArray], Any],
    *,
    n_resamples: int = 999,
    level: float = 0.95,
    method: Method = "bca",
    random_state: Any = None,
    n_jobs: int = 1,
) -> BootstrapResult:
    """Bootstrap any statistic of a data matrix, resampling whole rows.

    Parameters
    ----------
    x : array_like, shape (n, d)
        The data. Rows are observations.
    statistic : callable
        Takes an ``(n, d)`` array and returns a float or an array of floats.
        Raising is allowed and is treated as refusing that resample.
    n_resamples : int
        Number of bootstrap replicates. 999 rather than 1000 by convention: the
        percentile of ``B`` replicates is exact when ``(B+1)*alpha`` is an
        integer.
    level : float
        Coverage, e.g. 0.95.
    method : {"bca", "percentile", "basic"}
        See the module docstring. BCa additionally runs an ``n``-point
        jackknife, so it costs one extra pass over the data.
    random_state : None, int or Generator
    n_jobs : int
        Processes to spread the resamples over. ``1`` stays in this process;
        anything else needs ``statistic`` to be picklable, which rules out
        lambdas and closures -- use a module-level function or
        :func:`functools.partial`.

        The seeds are drawn up front, so results do not depend on ``n_jobs``:
        a parallel run returns replicate-for-replicate what a serial one does.

        Expect **sublinear** speedup, and a peak well below the core count.
        NumPy already runs several BLAS threads inside each process, so the
        workers compete with each other: on an 8-core machine, 200 t-copula
        fits took 39s serially, 21s at ``n_jobs=4``, and 29s at ``n_jobs=8``.
        Half the cores is a reasonable default, and for a statistic as cheap as
        Kendall's tau the process startup costs more than it saves.

    Returns
    -------
    BootstrapResult

    Examples
    --------
    >>> import numpy as np, rcopula as rc
    >>> from rcopula.bootstrap import bootstrap
    >>> u = rc.GumbelCopula(2.0).rvs(300, random_state=0)
    >>> result = bootstrap(u, lambda d: rc.cor_kendall(d)[0, 1], n_resamples=199,
    ...                    random_state=0)
    >>> lower, upper = result.confidence_interval
    >>> bool(lower < 0.5 < upper)
    True
    >>> result.replicates.shape
    (199,)
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] < 2:
        raise ValueError(f"need at least 2 observations to resample, got {x.shape[0]}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_resamples < 2:
        raise ValueError(f"n_resamples must be at least 2, got {n_resamples}")
    if method not in ("bca", "percentile", "basic"):
        raise ValueError(f"method must be bca, percentile or basic; got {method!r}")

    estimate = _evaluate(statistic, x)
    if estimate is None:
        raise ValueError("the statistic failed on the original data")

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**63 - 1, size=n_resamples)

    if n_jobs == 1:
        values = [_one_resample((statistic, x, int(seed))) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=None if n_jobs < 0 else n_jobs) as pool:
            values = list(pool.map(_one_resample, [(statistic, x, int(s)) for s in seeds]))

    kept = [v for v in values if v is not None]
    n_failed = n_resamples - len(kept)
    if n_failed > _MAX_FAILURE_FRACTION * n_resamples:
        raise RuntimeError(
            f"{n_failed} of {n_resamples} resamples were refused by the statistic. "
            "That is too many to be incidental -- check that it handles ties and "
            "degenerate columns."
        )
    replicates = np.asarray(kept, dtype=float)

    estimate = np.atleast_1d(estimate)
    scalar = estimate.size == 1 and np.asarray(_evaluate(statistic, x)).ndim == 0

    if method == "percentile":
        lower, upper = _percentile_interval(replicates, level)
    elif method == "basic":
        lower, upper = _basic_interval(
            replicates, estimate if replicates.ndim > 1 else estimate[0], level
        )
    else:
        jackknife = []
        for i in range(x.shape[0]):
            value = _evaluate(statistic, np.delete(x, i, axis=0))
            if value is not None:
                jackknife.append(value)
        lower, upper = _bca_interval(
            np.atleast_2d(replicates.reshape(replicates.shape[0], -1)),
            estimate,
            np.atleast_2d(np.asarray(jackknife, dtype=float).reshape(len(jackknife), -1)),
            level,
        )

    error = np.std(replicates, axis=0, ddof=1)
    if scalar:
        return BootstrapResult(
            estimate=float(estimate[0]),
            confidence_interval=(float(np.ravel(lower)[0]), float(np.ravel(upper)[0])),
            standard_error=float(np.ravel(error)[0]),
            replicates=replicates,
            method=method,
            level=level,
            n_failed=n_failed,
        )
    return BootstrapResult(
        estimate=estimate,
        confidence_interval=(
            np.asarray(lower).reshape(estimate.shape),
            np.asarray(upper).reshape(estimate.shape),
        ),
        standard_error=error,
        replicates=replicates,
        method=method,
        level=level,
        n_failed=n_failed,
    )


# --------------------------------------------------------------------------
# named statistics, at module level so a process pool can pickle them
# --------------------------------------------------------------------------


def _tau(x: NDArray[np.float64]) -> float:
    from rcopula.dependence import cor_kendall

    return float(np.asarray(cor_kendall(x))[0, 1])


def _rho(x: NDArray[np.float64]) -> float:
    from rcopula.dependence import cor_spearman

    return float(np.asarray(cor_spearman(x))[0, 1])


def _beta(x: NDArray[np.float64]) -> float:
    from rcopula.dependence import beta_n

    return float(beta_n(x))


def _lambda_upper(x: NDArray[np.float64], threshold: float = 0.95) -> float:
    """Nonparametric upper tail dependence, from the tail concentration function.

    ``P(U_1 > q, U_2 > q) / (1 - q)`` at a high ``q``. This is an estimate *of a
    limit* from a finite corner, so its sampling distribution is skewed and
    bounded -- exactly the case that motivates a bootstrap over an asymptotic
    standard error.
    """
    from rcopula.dependence import pseudo_obs

    u = pseudo_obs(x)
    return float(np.mean(np.all(np.asarray(u) > threshold, axis=1)) / (1.0 - threshold))


def _lambda_lower(x: NDArray[np.float64], threshold: float = 0.05) -> float:
    """Nonparametric lower tail dependence. See :func:`_lambda_upper`."""
    from rcopula.dependence import pseudo_obs

    u = pseudo_obs(x)
    return float(np.mean(np.all(np.asarray(u) < threshold, axis=1)) / threshold)


_MEASURES: dict[str, Callable[[NDArray[np.float64]], float]] = {
    "tau": _tau,
    "rho": _rho,
    "beta": _beta,
    "lambda_upper": _lambda_upper,
    "lambda_lower": _lambda_lower,
}


def bootstrap_measure(
    x: ArrayLike,
    measure: str = "tau",
    *,
    n_resamples: int = 999,
    level: float = 0.95,
    method: Method = "bca",
    random_state: Any = None,
    n_jobs: int = 1,
) -> BootstrapResult:
    """A confidence interval for a bivariate dependence measure.

    Parameters
    ----------
    x : array_like, shape (n, 2)
    measure : {"tau", "rho", "beta", "lambda_upper", "lambda_lower"}
    n_resamples, level, method, random_state, n_jobs
        As for :func:`bootstrap`.

    Returns
    -------
    BootstrapResult

    Examples
    --------
    Tail dependence is where the bootstrap earns its keep: the estimate is a
    ratio of small counts, so the interval is wide and asymmetric in a way no
    symmetric asymptotic interval would show.

    >>> import rcopula as rc
    >>> from rcopula.bootstrap import bootstrap_measure
    >>> u = rc.ClaytonCopula(2.0).rvs(500, random_state=0)
    >>> result = bootstrap_measure(u, "lambda_lower", n_resamples=299, random_state=0)
    >>> lower, upper = result.confidence_interval
    >>> bool(0.0 <= lower < result.estimate < upper)
    True
    """
    if measure not in _MEASURES:
        raise ValueError(f"unknown measure {measure!r}; available: {sorted(_MEASURES)}")
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 2:
        raise ValueError(f"a bivariate measure needs 2 columns, got {x.shape[1]}")
    return bootstrap(
        x,
        _MEASURES[measure],
        n_resamples=n_resamples,
        level=level,
        method=method,
        random_state=random_state,
        n_jobs=n_jobs,
    )


class _FitStatistic:
    """Refit a copula on a resample and return its parameters.

    A class rather than a closure so that a process pool can pickle it.
    """

    def __init__(self, copula: Copula, method: str) -> None:
        self.copula = copula
        self.method = method

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        from rcopula.dependence import pseudo_obs
        from rcopula.fit import fit

        result = fit(self.copula, pseudo_obs(x), method=self.method)
        return np.asarray(result.params, dtype=float)


def bootstrap_fit(
    x: ArrayLike,
    copula: Copula,
    *,
    fit_method: str = "mpl",
    n_resamples: int = 499,
    level: float = 0.95,
    method: Method = "bca",
    random_state: Any = None,
    n_jobs: int = 1,
) -> BootstrapResult:
    """Confidence intervals for a fitted copula's parameters.

    The nonparametric complement to the asymptotic standard errors from
    :func:`~rcopula.fit`: the whole estimation is redone on each resample, so
    the interval reflects the estimator actually used rather than its limiting
    distribution.

    Parameters
    ----------
    x : array_like, shape (n, d)
        Data. Converted to pseudo-observations inside each resample, which is
        the right order -- ranks must be recomputed on the resample, not carried
        over from the original sample.
    copula : Copula
        The family to fit.
    fit_method : str
        Passed to :func:`~rcopula.fit`.
    n_resamples : int
        Fewer than for a cheap statistic, because each one is a full fit.
    level, method, random_state, n_jobs
        As for :func:`bootstrap`.

    Returns
    -------
    BootstrapResult

    Notes
    -----
    Costs ``n_resamples`` fits, plus ``n`` more for the BCa jackknife. At
    ``n = 500`` that is a thousand fits; use ``method="percentile"`` or
    ``n_jobs > 1`` when that is too slow.

    Examples
    --------
    >>> import rcopula as rc
    >>> from rcopula.bootstrap import bootstrap_fit
    >>> u = rc.ClaytonCopula(2.0).rvs(300, random_state=0)
    >>> result = bootstrap_fit(u, rc.ClaytonCopula(1.0), n_resamples=99,
    ...                        method="percentile", random_state=0)
    >>> lower, upper = result.confidence_interval
    >>> bool(lower < 2.0 < upper)
    True
    """
    return bootstrap(
        x,
        _FitStatistic(copula, fit_method),
        n_resamples=n_resamples,
        level=level,
        method=method,
        random_state=random_state,
        n_jobs=n_jobs,
    )
