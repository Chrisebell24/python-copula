r"""Portfolio construction and copula pairs trading.

Two applications that both turn on the same idea: **the conditional
distribution is the signal**.

**Pairs trading.** The classical approach standardises a price spread and trades
its deviations, which implicitly assumes the two assets are jointly normal with
a stable linear relationship. A copula replaces that with
:math:`h(u_1 \mid u_2) = P(U_1 \le u_1 \mid U_2 = u_2)`: the probability of
seeing asset 1 this low *given where asset 2 actually is*. When that probability
is 2%, asset 1 is cheap relative to its partner in a sense that survives
non-normal margins, asymmetric dependence and tail clustering -- none of which a
z-scored spread can express. The quantity is uniform under the fitted model, so
the same thresholds mean the same thing across every pair.

**Mean-CVaR optimisation.** Mean-variance treats upside and downside alike and
is blind to tail dependence. Minimising conditional value at risk instead
targets the loss that actually matters, and Rockafellar & Uryasev showed the
problem is a **linear program** once returns are represented by scenarios --
which is exactly what a copula model produces.

==============================  ==============================================
:func:`mispricing_index`        Conditional probabilities for a pair.
:func:`pairs_signal`            Entry and exit signals from those.
:func:`backtest_pairs`          Walk-forward backtest of the strategy.
:func:`simulate_returns`        Scenario returns from a copula model.
:func:`mean_cvar_weights`       CVaR-optimal weights, by linear programming.
:func:`efficient_frontier`      The mean-CVaR frontier.
:func:`min_variance_weights`    Markowitz benchmark, for comparison.
==============================  ==============================================

References
----------
Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional
    value-at-risk. *Journal of Risk* 2(3), 21-42.
    The linear-programming formulation used by :func:`mean_cvar_weights`.
Liew, R. Q. and Wu, Y. (2013). Pairs trading: a copula approach.
    *Journal of Derivatives & Hedge Funds* 19(1), 12-30.
Stander, Y., Marais, D. and Botha, I. (2013). Trading strategies with copulas.
    *Journal of Economic and Financial Sciences* 6(1), 83-107.
Xie, W., Liew, R. Q., Wu, Y. and Zou, X. (2016). Pairs trading with copulas.
    *Journal of Trading* 11(3), 41-52.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from rcopula.core.base import Copula
from rcopula.dependence import pseudo_obs
from rcopula.distribution import CopulaDistribution, Margin
from rcopula.fit import fit
from rcopula.risk import expected_shortfall
from rcopula.transforms import conditional_cdf

__all__ = [
    "BacktestResult",
    "backtest_pairs",
    "efficient_frontier",
    "mean_cvar_weights",
    "min_variance_weights",
    "mispricing_index",
    "pairs_signal",
    "simulate_returns",
]


# ======================================================================
# Pairs trading
# ======================================================================


def mispricing_index(copula: Copula, u: ArrayLike) -> tuple[NDArray, NDArray]:
    r"""Conditional probabilities for both legs of a pair.

    Returns :math:`h_1 = P(U_1 \le u_1 \mid U_2 = u_2)` and
    :math:`h_2 = P(U_2 \le u_2 \mid U_1 = u_1)`.

    Both are uniform under the fitted copula, so a value of 0.02 always means
    "only a 2% chance of being this low" whatever the pair, the margins or the
    dependence shape. That comparability is what a z-scored price spread cannot
    give you.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.portfolio import mispricing_index
    >>> cop = ClaytonCopula(3.0)
    >>> h1, h2 = mispricing_index(cop, cop.rvs(5000, random_state=0))
    >>> bool(abs(h1.mean() - 0.5) < 0.02 and abs(h2.mean() - 0.5) < 0.02)
    True

    Asset 1 unusually low given a high asset 2 shows up as a small ``h1``:

    >>> h1, h2 = mispricing_index(cop, [[0.05, 0.90]])
    >>> bool(h1[0] < 0.05 and h2[0] > 0.95)
    True
    """
    arr = np.atleast_2d(np.asarray(u, dtype=np.float64))
    if arr.shape[1] != 2:
        raise ValueError(f"pairs trading is bivariate; got {arr.shape[1]} columns")
    return conditional_cdf(copula, arr, given=1), conditional_cdf(copula, arr, given=0)


def pairs_signal(
    copula: Copula,
    u: ArrayLike,
    entry: float = 0.05,
    exit_band: float = 0.5,
) -> NDArray[np.int_]:
    r"""Trading signal from the conditional copula probabilities.

    Returns ``+1`` to go long the spread (long asset 1, short asset 2), ``-1``
    for the reverse, and ``0`` to stand aside.

    A position opens when **both** legs agree that one asset is mispriced
    relative to the other -- ``h1`` below ``entry`` *and* ``h2`` above
    ``1 - entry``. Requiring both is what distinguishes relative mispricing from
    a common move: if the whole market falls, both conditionals stay near 0.5
    and no signal fires.

    Parameters
    ----------
    entry : float
        Threshold for opening. Smaller means rarer, higher-conviction trades.
    exit_band : float
        Positions close once the conditional returns within ``exit_band`` of
        0.5. The default of 0.5 exits only at full reversion.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.portfolio import pairs_signal
    >>> cop = ClaytonCopula(3.0)
    >>> u = np.array([[0.02, 0.95], [0.95, 0.02], [0.50, 0.50]])
    >>> pairs_signal(cop, u)
    array([ 1, -1,  0])

    Signals are rare by construction -- a few percent of observations:

    >>> signals = pairs_signal(cop, cop.rvs(5000, random_state=0))
    >>> bool(0.0 < np.mean(signals != 0) < 0.15)
    True
    """
    if not 0.0 < entry < 0.5:
        raise ValueError(f"entry must lie in (0, 0.5), got {entry}")
    h1, h2 = mispricing_index(copula, u)

    signal = np.zeros(h1.size, dtype=int)
    signal[(h1 <= entry) & (h2 >= 1.0 - entry)] = 1
    signal[(h1 >= 1.0 - entry) & (h2 <= entry)] = -1
    return signal


class BacktestResult(NamedTuple):
    """Outcome of a pairs backtest."""

    total_return: float
    annualised_sharpe: float
    n_trades: int
    hit_rate: float
    returns: NDArray[np.float64]
    positions: NDArray[np.int_]

    def __repr__(self) -> str:
        return (
            f"BacktestResult(total_return={self.total_return:.4%}, "
            f"sharpe={self.annualised_sharpe:.2f}, trades={self.n_trades}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


def backtest_pairs(
    returns: ArrayLike,
    copula: Copula,
    train: int = 250,
    entry: float = 0.05,
    periods_per_year: int = 252,
    refit_every: int = 0,
) -> BacktestResult:
    r"""Walk-forward backtest of the copula pairs strategy.

    Fits the copula on a trailing window, forms a signal from the *next*
    observation only, and holds the resulting position for one period. The
    signal at time ``t`` never sees data from time ``t``, which is the part
    most easily got wrong.

    Parameters
    ----------
    returns : array_like
        ``(n, 2)`` period returns for the two assets.
    copula : Copula
        Family to fit. Parameters are estimated from each training window.
    train : int
        Length of the trailing window.
    refit_every : int
        Refit cadence. ``0`` refits every period; larger values are faster and
        make the position depend on a slightly staler model.

    Returns
    -------
    BacktestResult

    Notes
    -----
    Deliberately frictionless: no transaction costs, no borrow cost, no
    slippage, no capacity limit, and a spread traded at equal notional rather
    than a hedge ratio. Pairs strategies live or die on costs, so treat the
    Sharpe here as an upper bound on what the *signal* could deliver, not as a
    strategy result.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, CopulaDistribution
    >>> from rcopula.portfolio import backtest_pairs
    >>> from scipy import stats
    >>> mv = CopulaDistribution(ClaytonCopula(4.0), [stats.norm(0, 0.01)] * 2)
    >>> r = mv.rvs(600, random_state=0)
    >>> result = backtest_pairs(r, ClaytonCopula(), train=250, refit_every=25)
    >>> bool(result.n_trades >= 0 and np.isfinite(result.annualised_sharpe))
    True
    """
    r = np.asarray(
        returns.to_numpy() if isinstance(returns, pd.DataFrame) else returns,
        dtype=np.float64,
    )
    if r.ndim != 2 or r.shape[1] != 2:
        raise ValueError(f"returns must be (n, 2); got shape {r.shape}")
    n = r.shape[0]
    if n <= train + 1:
        raise ValueError(f"need more than train+1 = {train + 1} observations, got {n}")

    positions = np.zeros(n, dtype=int)
    fitted: Copula | None = None

    for t in range(train, n - 1):
        if fitted is None or (refit_every <= 0) or ((t - train) % refit_every == 0):
            window = r[t - train : t]
            fitted = fit(copula, pseudo_obs(window), estimate_variance=False).copula

        # Rank the newest observation inside the training window, so the signal
        # uses only information available at time t.
        window = r[t - train : t + 1]
        u_now = np.asarray(pseudo_obs(window))[-1:]
        positions[t + 1] = int(pairs_signal(fitted, u_now, entry=entry)[0])

    # Long the spread means long asset 1 and short asset 2.
    strategy = positions * (r[:, 0] - r[:, 1])
    traded = positions != 0

    total = float(np.prod(1.0 + strategy) - 1.0)
    active = strategy[traded]
    sharpe = (
        float(active.mean() / active.std(ddof=1) * np.sqrt(periods_per_year))
        if active.size > 1 and active.std(ddof=1) > 0
        else 0.0
    )
    return BacktestResult(
        total_return=total,
        annualised_sharpe=sharpe,
        n_trades=int(np.sum(np.diff(positions) != 0)),
        hit_rate=float(np.mean(active > 0)) if active.size else 0.0,
        returns=strategy,
        positions=positions,
    )


# ======================================================================
# Portfolio optimisation
# ======================================================================


def simulate_returns(
    copula: Copula,
    margins: Margin | list[Margin],
    n: int = 20_000,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """Scenario returns from a copula model, for the optimisers below.

    Examples
    --------
    >>> from scipy import stats
    >>> from rcopula import StudentCopula
    >>> from rcopula.portfolio import simulate_returns
    >>> r = simulate_returns(StudentCopula(0.4, dim=4, df=5), stats.norm(0.0005, 0.012),
    ...                      n=5000, random_state=0)
    >>> r.shape
    (5000, 4)
    """
    return np.asarray(
        CopulaDistribution(copula, margins).rvs(n, random_state=random_state),
        dtype=np.float64,
    )


def mean_cvar_weights(
    scenarios: ArrayLike,
    alpha: float = 0.95,
    target_return: float | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> NDArray[np.float64]:
    r"""Weights minimising conditional value at risk, by linear programming.

    Rockafellar & Uryasev (2000) showed that minimising

    .. math::
        \mathrm{CVaR}_\alpha(w) = \min_{z}\ z +
            \frac{1}{(1-\alpha)n}\sum_i \bigl(-r_i^{\top}w - z\bigr)^{+}

    is a linear program in :math:`(w, z, s)` once returns are given as
    scenarios. That makes it exact and fast -- no gradient descent, no local
    optima -- and it is why scenario-based CVaR optimisation is practical at all.

    Parameters
    ----------
    scenarios : array_like
        ``(n, d)`` scenario **returns** (not losses).
    alpha : float
        CVaR confidence level.
    target_return : float, optional
        Minimum required mean return. Omit for the global CVaR minimum.
    bounds : tuple
        ``(lower, upper)`` bound on each weight. The default forbids shorting.

    Returns
    -------
    ndarray
        Weights summing to one.

    Examples
    --------
    The optimiser avoids the asset with the fat left tail, even when its mean is
    identical:

    >>> import numpy as np
    >>> from rcopula.portfolio import mean_cvar_weights
    >>> rng = np.random.default_rng(0)
    >>> safe = rng.normal(0.001, 0.01, 4000)
    >>> risky = rng.standard_t(2.5, 4000) * 0.006 + 0.001
    >>> w = mean_cvar_weights(np.column_stack([safe, risky]))
    >>> bool(w[0] > w[1])
    True
    >>> bool(abs(w.sum() - 1.0) < 1e-8)
    True
    """
    r = np.atleast_2d(np.asarray(scenarios, dtype=np.float64))
    n, d = r.shape
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")

    # Variables: [w (d), z (1), s (n)].
    scale = 1.0 / ((1.0 - alpha) * n)
    cost = np.concatenate([np.zeros(d), [1.0], np.full(n, scale)])

    # -r_i . w - z - s_i <= 0
    a_ub = np.hstack([-r, -np.ones((n, 1)), -np.eye(n)])
    b_ub = np.zeros(n)

    if target_return is not None:
        row = np.concatenate([-r.mean(axis=0), [0.0], np.zeros(n)])
        a_ub = np.vstack([a_ub, row])
        b_ub = np.append(b_ub, -float(target_return))

    a_eq = np.concatenate([np.ones(d), [0.0], np.zeros(n)]).reshape(1, -1)
    var_bounds = [bounds] * d + [(None, None)] + [(0.0, None)] * n

    result = optimize.linprog(
        cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=[1.0],
        bounds=var_bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(
            f"the CVaR program is infeasible: {result.message}. "
            "A target return above the best attainable is the usual cause."
        )
    return np.asarray(result.x[:d], dtype=np.float64)


def min_variance_weights(
    scenarios: ArrayLike, bounds: tuple[float, float] = (0.0, 1.0)
) -> NDArray[np.float64]:
    """Minimum-variance weights, as a Markowitz benchmark.

    Provided for comparison: mean-variance treats gains and losses
    symmetrically and sees only the covariance matrix, so it cannot distinguish
    two portfolios with the same covariance but different tail dependence.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.portfolio import min_variance_weights
    >>> rng = np.random.default_rng(0)
    >>> r = np.column_stack([rng.normal(0, 0.01, 2000), rng.normal(0, 0.03, 2000)])
    >>> w = min_variance_weights(r)
    >>> bool(w[0] > w[1])          # the quieter asset gets the weight
    True
    """
    r = np.atleast_2d(np.asarray(scenarios, dtype=np.float64))
    d = r.shape[1]
    cov = np.atleast_2d(np.cov(r, rowvar=False))

    # Rescale so the objective is O(1). Daily-return covariances are ~1e-4,
    # which sits below SLSQP's default ftol: left unscaled the optimiser
    # declares convergence at the starting point and silently returns equal
    # weights. Scaling by the mean variance does not move the argmin.
    scale = float(np.mean(np.diag(cov)))
    normalised = cov / scale if scale > 0 else cov

    result = optimize.minimize(
        lambda w: float(w @ normalised @ w),
        x0=np.full(d, 1.0 / d),
        jac=lambda w: 2.0 * normalised @ w,
        bounds=[bounds] * d,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:  # pragma: no cover - SLSQP rarely fails here
        raise ValueError(f"minimum-variance optimisation failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def efficient_frontier(
    scenarios: ArrayLike,
    alpha: float = 0.95,
    n_points: int = 20,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Mean-CVaR efficient frontier.

    Returns
    -------
    returns, cvars, weights : ndarray
        Target return, achieved CVaR, and the weights at each frontier point.
        Infeasible targets are dropped rather than raising.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.portfolio import efficient_frontier
    >>> rng = np.random.default_rng(0)
    >>> r = np.column_stack([rng.normal(0.0004, 0.01, 3000),
    ...                      rng.normal(0.0010, 0.02, 3000)])
    >>> mu, cvar, w = efficient_frontier(r, n_points=8)
    >>> bool(np.all(np.diff(cvar) >= -1e-9))     # more return costs more risk
    True
    """
    r = np.atleast_2d(np.asarray(scenarios, dtype=np.float64))
    means = r.mean(axis=0)
    lowest = float(mean_cvar_weights(r, alpha, None, bounds) @ means)
    targets = np.linspace(lowest, float(means.max()), int(n_points))

    kept_mu, kept_cvar, kept_w = [], [], []
    for target in targets:
        try:
            w = mean_cvar_weights(r, alpha, float(target), bounds)
        except ValueError:
            continue
        kept_mu.append(float(w @ means))
        kept_cvar.append(expected_shortfall(-(r @ w), alpha))
        kept_w.append(w)

    return np.array(kept_mu), np.array(kept_cvar), np.array(kept_w)
