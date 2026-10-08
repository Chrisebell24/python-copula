r"""Copula-GARCH: dependence in the cross-section, volatility in time.

Fitting a copula directly to asset returns is almost always wrong. Returns are
not identically distributed -- they arrive in volatile and quiet regimes -- and a
copula fitted to the raw series confuses *volatility clustering* with
*dependence*. Two assets that are independent but share a calm month and a wild
month will look strongly tail-dependent, because both were large at the same
time for a reason that has nothing to do with either.

The copula-GARCH model separates the two:

.. math::

    r_{j,t} = \mu_j + \sigma_{j,t} z_{j,t}, \qquad
    \sigma_{j,t}^2 = \omega_j + \alpha_j \varepsilon_{j,t-1}^2
                     + \beta_j \sigma_{j,t-1}^2,

with the innovation vector :math:`(z_{1,t}, \dots, z_{d,t})` i.i.d. across time
and coupled by a copula. Each margin gets its own volatility dynamics; the
copula then describes what is left, which is dependence proper.

This is a **two-step** estimator (Patton 2006): fit each GARCH by
quasi-maximum-likelihood, take the standardised residuals, and fit the copula to
their pseudo-observations. The second step inherits the first step's estimation
error, which is why the copula standard errors reported here are conditional on
the fitted margins.

Forecasting is where the model earns its place. Simulating forward gives a joint
predictive distribution of returns over any horizon, from which portfolio VaR,
expected shortfall or an option payoff follows directly -- with tail dependence
and volatility persistence both present, which no single correlation number can
deliver.

============================  ================================================
:func:`fit_garch`             GARCH(1,1) by QMLE, in pure NumPy.
:class:`GarchResult`          A fitted margin, with forecasting.
:class:`CopulaGarch`          The joint model: GARCH margins plus a copula.
============================  ================================================

The GARCH implementation here is deliberately small -- constant mean, GARCH(1,1),
normal or Student-t innovations -- because that is what the copula literature
uses and it keeps ``rcopula`` dependency-free. For EGARCH, GJR, long-memory or
regime-switching margins, fit them with the ``arch`` package and pass the
standardised residuals to :func:`~rcopula.fit.fit` yourself; the second step is
unchanged.

References
----------
Patton, A. J. (2006). Modelling asymmetric exchange rate dependence.
    *International Economic Review* 47(2), 527-556.
    The two-step copula-GARCH estimator.
Jondeau, E. and Rockinger, M. (2006). The copula-GARCH model of conditional
    dependencies: an international stock market application.
    *Journal of International Money and Finance* 25(5), 827-853.
Bollerslev, T. (1986). Generalized autoregressive conditional
    heteroskedasticity. *Journal of Econometrics* 31(3), 307-327.
Bollerslev, T. and Wooldridge, J. M. (1992). Quasi-maximum likelihood estimation
    and inference in dynamic models with time-varying covariances.
    *Econometric Reviews* 11(2), 143-172.
    Why the normal-innovation GARCH estimate stays consistent under
    misspecification -- the "Q" in QMLE.
Barone-Adesi, G., Giannopoulos, K. and Vosper, L. (1999). VaR without
    correlations for portfolios of derivative securities.
    *Journal of Futures Markets* 19(5), 583-602.
    Filtered historical simulation, the ``innovations="empirical"`` option.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, signal, special, stats

from rcopula.core.base import Copula
from rcopula.dependence import pseudo_obs
from rcopula.fit import fit as fit_copula
from rcopula.risk import expected_shortfall, value_at_risk

__all__ = ["CopulaGarch", "GarchResult", "fit_garch"]

#: Largest persistence allowed. At alpha + beta = 1 the process is IGARCH and
#: the unconditional variance does not exist, which breaks the forecast formula.
_MAX_PERSISTENCE = 0.9999

#: Smallest degrees of freedom. Below 2 the Student-t has no variance, so it
#: cannot be standardised.
_MIN_DF = 2.05
_MAX_DF = 200.0


def _filter_variance(
    eps: NDArray[np.float64], omega: float, alpha: float, beta: float, sigma2_0: float
) -> NDArray[np.float64]:
    r"""Conditional variances from the GARCH recursion.

    The recursion :math:`\sigma_t^2 = (\omega + \alpha\varepsilon_{t-1}^2)
    + \beta\sigma_{t-1}^2` is a first-order linear filter in :math:`\sigma^2`
    driven by a known series, so it runs as one ``lfilter`` call rather than a
    Python loop -- roughly 100x faster, which matters because the optimiser
    evaluates it hundreds of times.
    """
    drive = omega + alpha * eps[:-1] ** 2
    tail = signal.lfilter([1.0], [1.0, -beta], drive, zi=np.array([beta * sigma2_0]))[0]
    return np.concatenate([[sigma2_0], tail])


def _standardised_t(df: float) -> Any:
    """Student-t rescaled to unit variance, the usual GARCH innovation."""
    return stats.t(df=df, scale=np.sqrt((df - 2.0) / df))


def _neg_loglik(
    theta: NDArray[np.float64], x: NDArray[np.float64], dist: str, sigma2_0: float
) -> float:
    mu, omega, alpha, beta = theta[:4]
    eps = x - mu
    sigma2 = _filter_variance(eps, omega, alpha, beta, sigma2_0)
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
        return np.inf

    z2 = eps**2 / sigma2
    if dist == "normal":
        ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + z2)
    else:
        nu = theta[4]
        const = (
            special.gammaln(0.5 * (nu + 1.0))
            - special.gammaln(0.5 * nu)
            - 0.5 * np.log(np.pi * (nu - 2.0))
        )
        ll = np.sum(const - 0.5 * np.log(sigma2) - 0.5 * (nu + 1.0) * np.log1p(z2 / (nu - 2.0)))
    return -float(ll)


@dataclass(frozen=True)
class GarchResult:
    r"""One asset's fitted volatility model: parameters, per-period volatility, forecasts.

    This is what :func:`fit_garch` returns. It tells you how volatile the
    asset is today, how volatile it is on average, how quickly a volatility
    spike fades, and it holds the "de-volatilised" returns (the standardised
    residuals) that the copula step is fitted to. You normally get one from
    :func:`fit_garch` rather than building it yourself.

    The model is a constant-mean GARCH(1,1):
    :math:`r_t = \mu + \sigma_t z_t`, with
    :math:`\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2`
    and :math:`\varepsilon_t = r_t - \mu`.

    Parameters
    ----------
    mu : float
        Constant mean return per period, in the units of the input returns.
    omega : float
        Variance intercept :math:`\omega`, in squared return units.
    alpha : float
        Reaction to the previous period's shock (ARCH coefficient), between 0 and 1.
    beta : float
        Weight on the previous period's variance (GARCH coefficient), between 0 and 1.
    df : float or None
        Student-t degrees of freedom of the innovations; ``None`` for normal
        innovations.
    sigma : numpy.ndarray of float, shape (n,)
        Fitted conditional standard deviation for each observation.
    resid : numpy.ndarray of float, shape (n,)
        Standardised residuals, unitless.
    loglik : float
        Maximised log-likelihood on the original data scale.
    dist : str
        ``"normal"`` or ``"t"``.
    name : str, default ""
        Series label.

    Attributes
    ----------
    mu, omega, alpha, beta : float
        Constant mean and variance-equation parameters, on the **original**
        scale of the data (``mu`` in return units, ``omega`` in squared return
        units; ``alpha`` and ``beta`` are unitless).
    df : float or None
        Innovation degrees of freedom; ``None`` for normal innovations.
    sigma : numpy.ndarray of float, shape (n,)
        Fitted conditional standard deviations, one per observation, in the
        units of the input returns (e.g. daily volatility for daily returns).
    resid : numpy.ndarray of float, shape (n,)
        Standardised residuals :math:`z_t = (r_t - \mu)/\sigma_t`: each period's
        return divided by that period's volatility. These are the input to the
        copula step.
    loglik : float
        Maximised log-likelihood.
    dist : str
        ``"normal"`` or ``"t"``.
    name : str
        Series label, carried through from a ``pandas`` column name; ``""``
        if none was given.
    """

    mu: float
    omega: float
    alpha: float
    beta: float
    df: float | None
    sigma: NDArray[np.float64]
    resid: NDArray[np.float64]
    loglik: float
    dist: str
    name: str = ""

    @property
    def persistence(self) -> float:
        r"""How long volatility shocks linger: :math:`\alpha + \beta`.

        Values near 1 (typical for daily equity returns: 0.97-0.99) mean a
        volatility spike fades slowly; values well below 1 mean it dies out
        within a few periods.

        Returns
        -------
        float
            :math:`\alpha + \beta`, between 0 and 1 (capped at 0.9999 by the fit).
        """
        return self.alpha + self.beta

    @property
    def unconditional_vol(self) -> float:
        r"""The long-run average volatility the model reverts to.

        Technically the unconditional standard deviation,
        :math:`\sqrt{\omega/(1-\alpha-\beta)}`. Forecasts drift towards it as
        the horizon grows.

        Returns
        -------
        float
            Per-period standard deviation, in the units of the input returns
            (e.g. daily volatility for daily returns; multiply by
            :math:`\sqrt{252}` for an annualised figure).
        """
        return float(np.sqrt(self.omega / (1.0 - self.persistence)))

    @property
    def half_life(self) -> float:
        """Number of periods for a volatility shock to fade by half.

        Computed as ``log(0.5) / log(persistence)``. The unit is one period of
        the input data: days for daily returns, weeks for weekly returns, and
        so on.

        Returns
        -------
        float
            Half-life in periods of the input data. ``0.0`` when
            ``persistence == 0`` (no persistence: a shock is gone by the next
            period), and ``inf`` when ``persistence >= 1`` (shocks never fade).
        """
        p = self.persistence
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return float("inf")
        return float(np.log(0.5) / np.log(p))

    @property
    def n_params(self) -> int:
        """Number of estimated parameters: 4 for normal innovations, 5 for Student-t.

        Returns
        -------
        int
            ``4`` (mu, omega, alpha, beta) or ``5`` (plus df).
        """
        return 4 if self.df is None else 5

    @property
    def aic(self) -> float:
        """Akaike information criterion; lower is better when comparing fits.

        Use it to compare, say, a normal and a Student-t fit of the same
        series. Computed as ``2 * n_params - 2 * loglik``.

        Returns
        -------
        float
            The AIC.
        """
        return 2.0 * self.n_params - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        """Bayesian information criterion; lower is better, and stricter than AIC.

        Computed as ``n_params * log(n) - 2 * loglik``, with ``n`` the number of
        observations.

        Returns
        -------
        float
            The BIC.
        """
        return self.n_params * float(np.log(self.sigma.size)) - 2.0 * self.loglik

    def innovation(self) -> Any:
        """The distribution of each period's volatility-adjusted "surprise" (variance 1).

        Standard normal for ``dist="normal"``; a Student-t rescaled to unit
        variance for ``dist="t"``. It is a frozen ``scipy.stats``
        distribution, so it plugs straight into
        :class:`~rcopula.distribution.CopulaDistribution`.

        Returns
        -------
        scipy.stats frozen distribution
            Mean 0, variance 1, with ``.ppf``, ``.cdf``, ``.rvs`` and so on.
        """
        return stats.norm() if self.df is None else _standardised_t(self.df)

    def forecast_variance(self, horizon: int = 1) -> NDArray[np.float64]:
        r"""Forecast the variance (volatility squared) for each of the next periods.

        Use it to see how today's elevated (or depressed) volatility is
        expected to drift back to its long-run level over the coming periods.

        Parameters
        ----------
        horizon : int, default 1
            Number of periods ahead to forecast; must be at least 1.

        Returns
        -------
        numpy.ndarray of float, shape (horizon,)
            Element ``h - 1`` is the expected variance ``h`` periods after the
            last observation, in squared return units. These are per-period
            variances, not cumulative ones.

        Raises
        ------
        ValueError
            If ``horizon < 1``.

        Notes
        -----
        One step ahead is exact; beyond that the forecast decays geometrically
        towards the unconditional variance,

        .. math::
            \mathbb{E}[\sigma_{n+h}^2] = \bar\sigma^2
                + (\alpha+\beta)^{h-1}\bigl(\sigma_{n+1}^2 - \bar\sigma^2\bigr).

        Examples
        --------
        >>> import numpy as np
        >>> from rcopula.garch import fit_garch
        >>> rng = np.random.default_rng(0)
        >>> x = rng.standard_normal(1500) * 0.01
        >>> res = fit_garch(x)
        >>> v = res.forecast_variance(250)
        >>> bool(abs(np.sqrt(v[-1]) / res.unconditional_vol - 1) < 0.05)
        True
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        eps_last = self.resid[-1] * self.sigma[-1]
        first = self.omega + self.alpha * eps_last**2 + self.beta * self.sigma[-1] ** 2
        long_run = self.unconditional_vol**2
        decay = self.persistence ** np.arange(horizon)
        return long_run + decay * (first - long_run)

    def forecast_vol(self, horizon: int = 1) -> NDArray[np.float64]:
        """Forecast the volatility (standard deviation) for each of the next periods.

        The square root of :meth:`forecast_variance`.

        Parameters
        ----------
        horizon : int, default 1
            Number of periods ahead to forecast; must be at least 1.

        Returns
        -------
        numpy.ndarray of float, shape (horizon,)
            Per-period volatility forecasts, in the units of the input returns.

        Raises
        ------
        ValueError
            If ``horizon < 1``.

        Examples
        --------
        >>> import numpy as np
        >>> from rcopula.garch import fit_garch
        >>> rng = np.random.default_rng(0)
        >>> res = fit_garch(rng.standard_normal(1000) * 0.01)
        >>> res.forecast_vol(5).shape
        (5,)
        """
        return np.sqrt(self.forecast_variance(horizon))

    def __repr__(self) -> str:
        label = f" {self.name}" if self.name else ""
        dof = "" if self.df is None else f", df={self.df:.2f}"
        return (
            f"GarchResult({self.dist}{label}: mu={self.mu:.4g}, omega={self.omega:.4g}, "
            f"alpha={self.alpha:.4f}, beta={self.beta:.4f}{dof}, "
            f"persistence={self.persistence:.4f})"
        )


def _fit_reparameterised(
    y: NDArray[np.float64],
    dist: str,
    theta: NDArray[np.float64],
    bounds: list[tuple[float, float]],
) -> NDArray[np.float64]:
    """Maximise over (mu, omega, persistence, alpha share[, df]) by L-BFGS-B."""
    persistence = min(float(theta[2] + theta[3]), _MAX_PERSISTENCE)
    share = float(theta[2] / (theta[2] + theta[3])) if theta[2] + theta[3] > 0 else 0.1

    def unpack(z: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.array(z, dtype=float)
        out[2], out[3] = z[2] * z[3], z[2] * (1.0 - z[3])
        return out

    start = np.array(theta, dtype=float)
    start[2], start[3] = persistence, share
    box = [bounds[0], bounds[1], (0.0, _MAX_PERSISTENCE), (0.0, 1.0), *bounds[4:]]
    opt = optimize.minimize(
        lambda z: _neg_loglik(unpack(z), y, dist, 1.0),
        start,
        method="L-BFGS-B",
        bounds=box,
    )
    return unpack(opt.x)


def fit_garch(
    x: ArrayLike,
    dist: Literal["normal", "t"] = "normal",
    name: str = "",
) -> GarchResult:
    r"""Estimate one asset's time-varying volatility from its return history.

    Fits a GARCH(1,1) model with constant mean by (quasi-)maximum likelihood.
    Use it to measure how volatile a series is today versus on average, to
    forecast volatility, or -- the main use in this package -- to strip
    volatility clustering out of returns before fitting a copula (see
    :class:`CopulaGarch`).

    Parameters
    ----------
    x : array_like of float, shape (n,)
        A single return series (e.g. daily log-returns), oldest first, with at
        least 50 finite observations. Any units work (0.01 or 1.0 for 1%); the
        fitted ``mu``, ``omega`` and ``sigma`` come back in the same units.
        Multi-dimensional input is flattened.
    dist : {"normal", "t"}, default "normal"
        Innovation distribution. ``"normal"`` is quasi-MLE -- consistent for the
        variance parameters even when returns are fat-tailed (Bollerslev &
        Wooldridge 1992), which is why it remains the default. ``"t"`` estimates
        the degrees of freedom as well and gives a better fit when you intend to
        *simulate* from the margin rather than only filter with it.
    name : str, default ""
        Label stored on the result (used as a row name by
        :meth:`CopulaGarch.summary`).

    Returns
    -------
    GarchResult
        Fitted parameters, the conditional volatility path ``sigma`` (shape
        ``(n,)``), the standardised residuals ``resid`` (shape ``(n,)``) and
        forecasting methods.

    Raises
    ------
    ValueError
        If ``x`` has fewer than 50 observations, contains NaN or infinite
        values, is constant, or if ``dist`` is not ``"normal"`` or ``"t"``.

    Notes
    -----
    The series is internally rescaled to unit variance before optimising and the
    parameters are mapped back afterwards. GARCH is exactly scale-equivariant --
    :math:`x \mapsto cx` sends :math:`(\mu,\omega) \mapsto (c\mu, c^2\omega)`
    and leaves :math:`\alpha,\beta` alone -- so this changes nothing statistically,
    but it keeps every quantity at order 1. Without it, daily returns give
    :math:`\omega \approx 10^{-6}`, which sits below the optimiser's convergence
    tolerance and produces silently unconverged fits.

    Examples
    --------
    Parameters are recovered from a simulated series:

    >>> import numpy as np
    >>> from rcopula.garch import fit_garch
    >>> rng = np.random.default_rng(0)
    >>> n, omega, alpha, beta = 8000, 0.05, 0.10, 0.85
    >>> s2, e = 1.0, 0.0
    >>> x = np.empty(n)
    >>> z = rng.standard_normal(n)
    >>> for i in range(n):
    ...     s2 = omega + alpha * e**2 + beta * s2
    ...     e = np.sqrt(s2) * z[i]
    ...     x[i] = e
    >>> res = fit_garch(x)
    >>> bool(abs(res.alpha - 0.10) < 0.03 and abs(res.beta - 0.85) < 0.05)
    True

    Volatility clustering is removed by the filter -- the whole point:

    >>> raw = np.corrcoef(x[1:] ** 2, x[:-1] ** 2)[0, 1]
    >>> filtered = np.corrcoef(res.resid[1:] ** 2, res.resid[:-1] ** 2)[0, 1]
    >>> bool(filtered < 0.25 * raw)
    True
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < 50:
        raise ValueError(f"need at least 50 observations to fit a GARCH, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains non-finite values")
    if dist not in ("normal", "t"):
        raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")

    scale = float(np.std(arr))
    if scale <= 0.0:
        raise ValueError("x is constant; there is no volatility to model")
    y = arr / scale

    # On the rescaled series the unconditional variance is 1, so omega =
    # 1 - alpha - beta is the natural start and the pre-sample variance is 1.
    start = [float(np.mean(y)), 0.05, 0.10, 0.85]
    bounds: list[tuple[float, float]] = [
        (-10.0, 10.0),
        (1e-8, 10.0),
        (0.0, _MAX_PERSISTENCE),
        (0.0, _MAX_PERSISTENCE),
    ]
    if dist == "t":
        start.append(8.0)
        bounds.append((_MIN_DF, _MAX_DF))

    with warnings.catch_warnings():
        # SLSQP evaluates trial points outside the box during its line search
        # and clips them, which older SciPy announces and newer SciPy does not.
        # It describes the optimiser working normally, so it is silenced -- but
        # by exact message, so a genuine RuntimeWarning from the likelihood
        # itself still surfaces rather than being swallowed with it.
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds",
            category=RuntimeWarning,
        )
        opt = optimize.minimize(
            _neg_loglik,
            np.array(start),
            args=(y, dist, 1.0),
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": lambda t: _MAX_PERSISTENCE - t[2] - t[3]},
            ],
            options={"maxiter": 500, "ftol": 1e-10},
        )
    theta = opt.x
    if theta[2] + theta[3] > _MAX_PERSISTENCE + 1e-9:
        # SLSQP only enforces alpha + beta < 1 at convergence; when it stops
        # early -- typical when the truth sits on the boundary -- the point it
        # returns can break it, giving a non-stationary model with no
        # unconditional variance. Refit with persistence and alpha's share of
        # it as box-bounded parameters, so the constraint cannot be broken.
        theta = _fit_reparameterised(y, dist, theta, bounds)
    mu, omega, alpha, beta = (float(v) for v in theta[:4])
    df = float(theta[4]) if dist == "t" else None

    sigma2 = _filter_variance(y - mu, omega, alpha, beta, 1.0)
    sigma = np.sqrt(sigma2)
    return GarchResult(
        mu=mu * scale,
        omega=omega * scale**2,
        alpha=alpha,
        beta=beta,
        df=df,
        sigma=sigma * scale,
        resid=(y - mu) / sigma,
        # The scaling shifts the log-likelihood by a constant Jacobian term,
        # n*log(scale); undo it so loglik/AIC/BIC refer to the original data.
        loglik=-_neg_loglik(theta, y, dist, 1.0) - arr.size * float(np.log(scale)),
        dist=dist,
        name=name,
    )


class CopulaGarch:
    """Multi-asset returns: each asset's own volatility, plus a copula linking the shocks.

    Each asset gets a GARCH(1,1) model so that its volatility can rise and
    fall over time; the copula then describes how the assets' de-volatilised
    shocks move together, including in the tails. Use it to simulate future
    joint returns and to produce forward-looking portfolio VaR and expected
    shortfall that reflect both today's volatility and crash co-movement.

    Most users build one with :meth:`fit` from a returns table. The
    constructor is for assembling a model from margins and a copula you have
    already fitted (or chosen) yourself.

    Parameters
    ----------
    margins : list of GarchResult, length d
        One fitted GARCH per series, e.g. from :func:`fit_garch`.
    copula : Copula
        Copula for the standardised innovations, with ``copula.dim == d``.
        It is used as given (not refitted).
    innovations : {"empirical", "parametric"}, default "empirical"
        How to invert the copula's uniforms when simulating. ``"empirical"``
        draws from the *observed* standardised residuals (filtered historical
        simulation) and so inherits their skew and kurtosis without assuming a
        shape; ``"parametric"`` uses the fitted normal or Student-t. Empirical
        cannot produce an innovation larger than the largest one seen, so use
        parametric for long-horizon or deep-tail work.

    Attributes
    ----------
    margins : list of GarchResult, length d
        The per-asset volatility models.
    copula : Copula
        The copula coupling the standardised innovations.
    innovations : str
        ``"empirical"`` or ``"parametric"``.

    Raises
    ------
    ValueError
        If ``len(margins) != copula.dim``, or ``innovations`` is not one of
        the two allowed strings.

    Examples
    --------
    See :meth:`fit`.
    """

    def __init__(
        self,
        margins: list[GarchResult],
        copula: Copula,
        innovations: Literal["empirical", "parametric"] = "empirical",
    ) -> None:
        if len(margins) != copula.dim:
            raise ValueError(f"{len(margins)} margins but the copula has dim={copula.dim}")
        if innovations not in ("empirical", "parametric"):
            raise ValueError(
                f"innovations must be 'empirical' or 'parametric', got {innovations!r}"
            )
        self.margins = list(margins)
        self.copula = copula
        self.innovations = innovations

    @property
    def dim(self) -> int:
        """Number of assets in the model.

        Returns
        -------
        int
            ``d``, the number of margins.
        """
        return len(self.margins)

    @property
    def names(self) -> list[str]:
        """Asset labels, falling back to ``"x0"``, ``"x1"``, ... for unnamed margins.

        Returns
        -------
        list of str, length d
            One label per asset, in column order.
        """
        return [m.name or f"x{j}" for j, m in enumerate(self.margins)]

    @classmethod
    def fit(
        cls,
        returns: ArrayLike,
        copula: Copula,
        dist: Literal["normal", "t"] = "normal",
        innovations: Literal["empirical", "parametric"] = "empirical",
        method: str = "mpl",
    ) -> CopulaGarch:
        r"""Fit the whole model to a table of asset returns.

        Two-step estimation (Patton 2006): first a GARCH(1,1) is fitted to each
        column to remove volatility clustering, then the copula is fitted to
        the resulting standardised residuals. This is the usual entry point.

        Parameters
        ----------
        returns : array_like of float or pandas.DataFrame, shape (n, d)
            Returns, one row per period (oldest first) and one column per
            asset, with at least 50 rows. Column names are kept if a frame is
            passed.
        copula : Copula
            Family to fit, with ``dim`` matching the number of columns. Any
            starting parameters are ignored -- it is refitted.
        dist : {"normal", "t"}, default "normal"
            Innovation distribution for the marginal GARCH models; see
            :func:`fit_garch`.
        innovations : {"empirical", "parametric"}, default "empirical"
            Simulation margin; see the class docstring.
        method : str, default "mpl"
            Copula estimation method, passed to :func:`~rcopula.fit.fit`.
            The default ``"mpl"`` is the standard choice here, since the
            residuals' distribution is not being claimed to be exactly the
            fitted one.

        Returns
        -------
        CopulaGarch
            The fitted joint model.

        Raises
        ------
        ValueError
            If ``returns`` is not 2-d, its column count differs from
            ``copula.dim``, or any column fails :func:`fit_garch` (too short,
            non-finite or constant).

        Examples
        --------
        Two *independent* series driven by a common volatility process. Filtering
        first is what stops the shared volatility being read as tail dependence:

        >>> import numpy as np
        >>> import rcopula as rc
        >>> from rcopula.garch import CopulaGarch
        >>> rng = np.random.default_rng(1)
        >>> h, e = np.zeros(3000), rng.standard_normal(3000)
        >>> for t in range(1, 3000):
        ...     h[t] = 0.98 * h[t - 1] + 0.25 * e[t]
        >>> r = rng.standard_normal((3000, 2)) * (0.01 * np.exp(h))[:, None]
        >>> model = CopulaGarch.fit(r, rc.StudentCopula(0.0, df=8.0, dim=2))
        >>> raw = rc.fit(rc.StudentCopula(0.0, df=8.0), rc.pseudo_obs(r), method="mpl")
        >>> bool(raw.copula.lambda_().upper > 3 * model.copula.lambda_().upper)
        True

        Note that the artifact is invisible to rank correlation -- both are near
        zero -- which is precisely why it goes unnoticed:

        >>> bool(abs(raw.copula.params[0]) < 0.06 and abs(model.copula.params[0]) < 0.06)
        True

        The fitted model then forecasts a joint distribution of returns:

        >>> paths = model.simulate(horizon=10, n=2000, random_state=0)
        >>> paths.shape
        (2000, 10, 2)
        """
        frame = returns if isinstance(returns, pd.DataFrame) else None
        arr = np.asarray(returns, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"returns must be 2-d, got {arr.ndim} dimension(s)")
        if arr.shape[1] != copula.dim:
            raise ValueError(
                f"returns has {arr.shape[1]} columns but the copula has dim={copula.dim}"
            )

        names = list(frame.columns.astype(str)) if frame is not None else [""] * arr.shape[1]
        margins = [fit_garch(arr[:, j], dist=dist, name=names[j]) for j in range(arr.shape[1])]
        resid = np.column_stack([m.resid for m in margins])
        fitted = fit_copula(copula, pseudo_obs(resid), method=method)
        return cls(margins, fitted.copula, innovations=innovations)

    def _innovation_ppf(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map copula uniforms to standardised innovations, column by column."""
        out = np.empty_like(u)
        for j, margin in enumerate(self.margins):
            if self.innovations == "parametric":
                out[:, j] = margin.innovation().ppf(u[:, j])
            else:
                out[:, j] = np.quantile(margin.resid, u[:, j], method="linear")
        return out

    def simulate(
        self,
        horizon: int = 1,
        n: int = 10_000,
        random_state: np.random.Generator | int | None = None,
    ) -> NDArray[np.float64]:
        r"""Generate ``n`` possible future return paths for all assets, starting from today.

        Use it for scenario analysis or any risk measure the built-in
        :meth:`forecast_risk` does not cover (e.g. a path-dependent payoff).

        Each step draws an innovation vector from the copula -- so the
        cross-sectional dependence is the fitted one -- and pushes it through
        each margin's own GARCH recursion, so volatility keeps clustering along
        the path. Both effects are present simultaneously, which is the reason
        to build the model at all. Paths start from the last observed
        volatility and shock.

        Parameters
        ----------
        horizon : int, default 1
            Number of future periods per path; must be at least 1.
        n : int, default 10_000
            Number of simulated paths.
        random_state : int, numpy.random.Generator or None, default None
            Seed or generator, for reproducible draws.

        Returns
        -------
        numpy.ndarray of float, shape (n, horizon, d)
            ``out[i, t, j]`` is the return of asset ``j`` in period ``t + 1``
            of path ``i``, in the units of the input returns.

        Raises
        ------
        ValueError
            If ``horizon < 1`` or ``n < 1``.

        Examples
        --------
        Simulated innovations carry the copula's dependence:

        >>> import numpy as np
        >>> from scipy import stats
        >>> import rcopula as rc
        >>> from rcopula.garch import CopulaGarch, fit_garch
        >>> rng = np.random.default_rng(0)
        >>> r = rng.standard_normal((1200, 2)) * 0.01
        >>> margins = [fit_garch(r[:, j]) for j in range(2)]
        >>> model = CopulaGarch(margins, rc.ClaytonCopula.from_tau(0.5))
        >>> paths = model.simulate(horizon=1, n=8000, random_state=0)
        >>> tau = stats.kendalltau(paths[:, 0, 0], paths[:, 0, 1]).statistic
        >>> bool(abs(tau - 0.5) < 0.03)
        True
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        d = self.dim
        mu = np.array([m.mu for m in self.margins])
        omega = np.array([m.omega for m in self.margins])
        alpha = np.array([m.alpha for m in self.margins])
        beta = np.array([m.beta for m in self.margins])

        # Start each path from the end of the observed sample: the last fitted
        # conditional variance and the last realised shock.
        sigma2 = np.tile([m.sigma[-1] ** 2 for m in self.margins], (n, 1))
        eps = np.tile([m.resid[-1] * m.sigma[-1] for m in self.margins], (n, 1))

        out = np.empty((n, horizon, d))
        for step in range(horizon):
            z = self._innovation_ppf(self.copula.rvs(n, random_state=rng))
            sigma2 = omega + alpha * eps**2 + beta * sigma2
            eps = np.sqrt(sigma2) * z
            out[:, step, :] = mu + eps
        return out

    def forecast(
        self,
        horizon: int = 1,
        n: int = 10_000,
        random_state: np.random.Generator | int | None = None,
    ) -> NDArray[np.float64]:
        """Simulate each asset's total return over the next ``horizon`` periods.

        Runs :meth:`simulate` and adds up each path's per-period returns.
        Sums the simulated log-returns, which is the usual convention. For
        simple returns compound them instead (simulate and use
        ``np.prod(1 + paths, axis=1) - 1``).

        Parameters
        ----------
        horizon : int, default 1
            Number of future periods to accumulate over; must be at least 1.
        n : int, default 10_000
            Number of simulated scenarios.
        random_state : int, numpy.random.Generator or None, default None
            Seed or generator, for reproducible draws.

        Returns
        -------
        numpy.ndarray of float, shape (n, d)
            Cumulative return of each asset in each scenario.

        Raises
        ------
        ValueError
            If ``horizon < 1`` or ``n < 1``.

        Examples
        --------
        >>> import numpy as np
        >>> import rcopula as rc
        >>> from rcopula.garch import CopulaGarch
        >>> rng = np.random.default_rng(0)
        >>> r = rng.standard_normal((1000, 2)) * 0.01
        >>> model = CopulaGarch.fit(r, rc.GaussianCopula(0.0, dim=2))
        >>> model.forecast(horizon=5, n=100, random_state=0).shape
        (100, 2)
        """
        return np.asarray(self.simulate(horizon, n, random_state).sum(axis=1))

    def forecast_risk(
        self,
        weights: ArrayLike | None = None,
        alpha: float = 0.99,
        horizon: int = 1,
        n: int = 50_000,
        random_state: np.random.Generator | int | None = None,
    ) -> dict[str, float]:
        r"""Forecast a portfolio's Value-at-Risk and expected shortfall over the horizon.

        This is the payoff of the whole construction: a forward-looking risk
        number that respects both current volatility -- which a static copula
        ignores -- and tail dependence, which a GARCH-only model ignores.
        Estimated by Monte Carlo from :meth:`forecast`.

        Parameters
        ----------
        weights : array_like of float, shape (d,), optional
            Portfolio exposures, one per asset in column order (e.g. ``[0.6,
            0.4]``). They are used **as given and not rescaled** to sum to 1,
            so they can describe a leveraged, long-short or partly invested
            book: the portfolio return is ``asset_returns @ weights``, and
            doubling the weights doubles every statistic. Equal-weighted
            (``1 / d`` each) if omitted.
        alpha : float, default 0.99
            Confidence level, e.g. 0.99 for a 99% VaR.
        horizon : int, default 1
            Forecast horizon in periods; must be at least 1.
        n : int, default 50_000
            Number of simulated scenarios. More scenarios give a less noisy
            estimate, especially at high ``alpha``.
        random_state : int, numpy.random.Generator or None, default None
            Seed or generator, for reproducible results.

        Returns
        -------
        dict of str to float
            Statistics of the portfolio return over the horizon, in the units of
            the input returns:

            - ``"var"`` -- Value-at-Risk, as a **loss** (positive means losing
              money): exceeded with probability ``1 - alpha``.
            - ``"expected_shortfall"`` -- average loss given the VaR is
              exceeded, as a **loss**.
            - ``"mean"`` -- mean portfolio return (a return, not a loss).
            - ``"volatility"`` -- sample standard deviation (``ddof=1``) of
              the simulated portfolio return.

        Raises
        ------
        ValueError
            If ``weights`` does not have length ``d``, ``horizon < 1`` or
            ``n < 2``.

        Examples
        --------
        Tail dependence raises the risk number at identical Kendall tau, which is
        exactly the comparison a correlation-based model cannot make:

        >>> import numpy as np
        >>> import rcopula as rc
        >>> from rcopula.garch import CopulaGarch, fit_garch
        >>> rng = np.random.default_rng(0)
        >>> r = rng.standard_normal((1500, 2)) * 0.01
        >>> margins = [fit_garch(r[:, j]) for j in range(2)]
        >>> gauss = CopulaGarch(margins, rc.GaussianCopula.from_tau(0.5))
        >>> clayton = CopulaGarch(margins, rc.ClaytonCopula.from_tau(0.5))
        >>> a = gauss.forecast_risk(alpha=0.99, n=40_000, random_state=0)
        >>> b = clayton.forecast_risk(alpha=0.99, n=40_000, random_state=0)
        >>> bool(b["var"] > a["var"])
        True
        """
        w = (
            np.full(self.dim, 1.0 / self.dim)
            if weights is None
            else np.asarray(weights, dtype=np.float64).ravel()
        )
        if w.size != self.dim:
            raise ValueError(f"weights has length {w.size}, expected {self.dim}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if n < 2:
            raise ValueError(f"n must be >= 2 to estimate a volatility, got {n}")

        returns = self.forecast(horizon, n, random_state) @ w
        losses = -returns
        return {
            "var": float(value_at_risk(losses, alpha)),
            "expected_shortfall": float(expected_shortfall(losses, alpha)),
            "mean": float(returns.mean()),
            "volatility": float(returns.std(ddof=1)),
        }

    def summary(self) -> pd.DataFrame:
        """A table of each asset's fitted volatility parameters, one row per asset.

        Returns
        -------
        pandas.DataFrame, shape (d, 8)
            Indexed by :attr:`names`, with float columns ``mu``, ``omega``,
            ``alpha``, ``beta``, ``df`` (NaN for normal innovations),
            ``persistence``, ``half_life`` (periods) and ``loglik``.

        Examples
        --------
        >>> import numpy as np
        >>> import rcopula as rc
        >>> from rcopula.garch import CopulaGarch
        >>> rng = np.random.default_rng(0)
        >>> r = rng.standard_normal((1000, 2)) * 0.01
        >>> model = CopulaGarch.fit(r, rc.GaussianCopula(0.0, dim=2))
        >>> list(model.summary().columns)
        ['mu', 'omega', 'alpha', 'beta', 'df', 'persistence', 'half_life', 'loglik']
        """
        return pd.DataFrame(
            [
                {
                    "mu": m.mu,
                    "omega": m.omega,
                    "alpha": m.alpha,
                    "beta": m.beta,
                    "df": np.nan if m.df is None else m.df,
                    "persistence": m.persistence,
                    "half_life": m.half_life,
                    "loglik": m.loglik,
                }
                for m in self.margins
            ],
            index=self.names,
        )

    def __repr__(self) -> str:
        return (
            f"CopulaGarch(d={self.dim}, copula={self.copula!r}, innovations={self.innovations!r})"
        )
