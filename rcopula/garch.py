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
    """A fitted GARCH(1,1) margin.

    Attributes
    ----------
    mu, omega, alpha, beta : float
        Constant mean and variance-equation parameters, on the **original**
        scale of the data.
    df : float or None
        Innovation degrees of freedom; ``None`` for normal innovations.
    sigma : ndarray
        Fitted conditional standard deviations, one per observation.
    resid : ndarray
        Standardised residuals :math:`z_t = (r_t - \\mu)/\\sigma_t`. These are
        the input to the copula step.
    loglik : float
        Maximised log-likelihood.
    dist : str
        ``"normal"`` or ``"t"``.
    name : str
        Series label, carried through from a ``pandas`` column name.
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
        """:math:`\\alpha + \\beta`. Near 1 means shocks decay slowly."""
        return self.alpha + self.beta

    @property
    def unconditional_vol(self) -> float:
        """Long-run standard deviation, :math:`\\sqrt{\\omega/(1-\\alpha-\\beta)}`."""
        return float(np.sqrt(self.omega / (1.0 - self.persistence)))

    @property
    def half_life(self) -> float:
        """Days for a variance shock to decay by half, ``log(0.5)/log(persistence)``."""
        return float(np.log(0.5) / np.log(self.persistence))

    @property
    def n_params(self) -> int:
        return 4 if self.df is None else 5

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        return self.n_params * float(np.log(self.sigma.size)) - 2.0 * self.loglik

    def innovation(self) -> Any:
        """The fitted innovation law, standardised to unit variance.

        A frozen ``scipy.stats`` distribution, so it plugs straight into
        :class:`~rcopula.distribution.CopulaDistribution`.
        """
        return stats.norm() if self.df is None else _standardised_t(self.df)

    def forecast_variance(self, horizon: int = 1) -> NDArray[np.float64]:
        r"""Conditional variance forecasts for the next ``horizon`` steps.

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
        """Conditional standard deviation forecasts; see :meth:`forecast_variance`."""
        return np.sqrt(self.forecast_variance(horizon))

    def __repr__(self) -> str:
        label = f" {self.name}" if self.name else ""
        dof = "" if self.df is None else f", df={self.df:.2f}"
        return (
            f"GarchResult({self.dist}{label}: mu={self.mu:.4g}, omega={self.omega:.4g}, "
            f"alpha={self.alpha:.4f}, beta={self.beta:.4f}{dof}, "
            f"persistence={self.persistence:.4f})"
        )


def fit_garch(
    x: ArrayLike,
    dist: Literal["normal", "t"] = "normal",
    name: str = "",
) -> GarchResult:
    r"""Fit a GARCH(1,1) with constant mean by (quasi-)maximum likelihood.

    Parameters
    ----------
    x : array_like
        A single return series.
    dist : {"normal", "t"}
        Innovation distribution. ``"normal"`` is quasi-MLE -- consistent for the
        variance parameters even when returns are fat-tailed (Bollerslev &
        Wooldridge 1992), which is why it remains the default. ``"t"`` estimates
        the degrees of freedom as well and gives a better fit when you intend to
        *simulate* from the margin rather than only filter with it.

    Returns
    -------
    GarchResult

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
        loglik=-float(opt.fun) - arr.size * float(np.log(scale)),
        dist=dist,
        name=name,
    )


class CopulaGarch:
    """Joint model: GARCH margins coupled by a copula.

    Parameters
    ----------
    margins : sequence of GarchResult
        One fitted GARCH per series.
    copula : Copula
        Fitted copula for the standardised innovations.
    innovations : {"empirical", "parametric"}
        How to invert the copula's uniforms when simulating. ``"empirical"``
        draws from the *observed* standardised residuals (filtered historical
        simulation) and so inherits their skew and kurtosis without assuming a
        shape; ``"parametric"`` uses the fitted normal or Student-t. Empirical
        cannot produce an innovation larger than the largest one seen, so use
        parametric for long-horizon or deep-tail work.

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
        return len(self.margins)

    @property
    def names(self) -> list[str]:
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
        r"""Two-step estimation: GARCH per column, then a copula on the residuals.

        Parameters
        ----------
        returns : array_like or DataFrame
            ``(n, d)`` returns. Column names are kept if a frame is passed.
        copula : Copula
            Family to fit, with ``dim`` matching the number of columns. Any
            starting parameters are ignored -- it is refitted.
        dist : {"normal", "t"}
            Innovation distribution for the marginal GARCH models.
        innovations : {"empirical", "parametric"}
            Simulation margin; see the class docstring.
        method : str
            Copula estimation method, passed to :func:`~rcopula.fit.fit`.
            The default ``"mpl"`` is the standard choice here, since the
            residuals' distribution is not being claimed to be exactly the
            fitted one.

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
        r"""Simulate ``n`` joint return paths of length ``horizon``.

        Each step draws an innovation vector from the copula -- so the
        cross-sectional dependence is the fitted one -- and pushes it through
        each margin's own GARCH recursion, so volatility keeps clustering along
        the path. Both effects are present simultaneously, which is the reason
        to build the model at all.

        Returns
        -------
        ndarray
            Shape ``(n, horizon, d)``.

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
        """Cumulative return over ``horizon``, shape ``(n, d)``.

        Sums the simulated log-returns, which is the usual convention. For
        simple returns compound them instead.
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
        r"""Portfolio VaR and expected shortfall from the predictive distribution.

        This is the payoff of the whole construction: a forward-looking risk
        number that respects both current volatility -- which a static copula
        ignores -- and tail dependence, which a GARCH-only model ignores.

        Parameters
        ----------
        weights : array_like, optional
            Portfolio weights; equal-weighted if omitted.
        alpha : float
            Confidence level.
        horizon : int
            Forecast horizon in periods.

        Returns
        -------
        dict
            ``var``, ``expected_shortfall``, ``mean`` and ``volatility`` of the
            horizon return, all as **losses** for the two risk measures and as
            returns for the two moments.

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

        returns = self.forecast(horizon, n, random_state) @ w
        losses = -returns
        return {
            "var": float(value_at_risk(losses, alpha)),
            "expected_shortfall": float(expected_shortfall(losses, alpha)),
            "mean": float(returns.mean()),
            "volatility": float(returns.std()),
        }

    def summary(self) -> pd.DataFrame:
        """Per-margin parameters and diagnostics as a frame.

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
