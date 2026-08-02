r"""Portfolio and systemic risk measures under copula dependence.

The reason copulas matter for risk is narrow and specific: **the dependence
model changes the tail of the aggregate loss even when every margin and every
pairwise correlation is held fixed.** Two portfolios calibrated to the same
Kendall's tau can differ by tens of percent in 99.9% VaR, because a Gaussian
copula says extreme losses arrive independently and a t or Clayton copula says
they arrive together. This module makes that difference measurable.

What is here:

===========================  =================================================
:func:`value_at_risk`        The quantile of the loss distribution.
:func:`expected_shortfall`   Mean loss beyond VaR -- coherent, unlike VaR.
:func:`simulate_losses`      Aggregate portfolio loss under a copula model.
:func:`risk_contributions`   Euler allocation of ES to positions.
:func:`diversification_benefit`  How much dependence costs you.
:func:`covar` / :func:`delta_covar`  Systemic risk: system VaR given a firm.
:func:`marginal_expected_shortfall`  A firm's loss in a system-wide crisis.
:func:`stress_scenario`      Conditional simulation given a stressed factor.
===========================  =================================================

**Sign convention.** Everything here works in *losses*: positive numbers are
bad, and VaR at level ``alpha`` is the ``alpha``-quantile of the loss. If you
have returns, negate them first.

.. warning::

   That convention flips which tail matters, and it is easy to get backwards.
   Aggregate loss risk is driven by **upper** tail dependence -- the tendency of
   large losses to arrive together. Clayton is *lower*-tail dependent, so
   applying it directly to losses clusters the *small* ones and understates
   risk. Measured on an equally weighted five-name portfolio, all calibrated to
   the same Kendall's tau of 0.5:

   ==========================  =============  ==========
   copula                      99% ES         lambda_U
   ==========================  =============  ==========
   Clayton (lower tail)        4.91           0.00
   Gaussian (no tail dep)      7.16           0.00
   Student t, df=4 (both)      7.61           0.40
   Gumbel (upper tail)         8.41           0.59
   ==========================  =============  ==========

   The ordering follows ``lambda_U`` exactly, and Clayton comes out *below*
   Gaussian. "Crashes cluster, so use Clayton" is right about **returns** and
   wrong about **losses**: negating flips the tails. For losses, reach for
   Gumbel, a Student-t, or a survival (180-degree rotated) Clayton.

References
----------
McNeil, A. J., Frey, R. and Embrechts, P. (2015). *Quantitative Risk
    Management: Concepts, Techniques and Tools*, 2nd ed. Princeton.
    Chapters 2 and 8 for risk measures and their aggregation.
Acerbi, C. and Tasche, D. (2002). On the coherence of expected shortfall.
    *Journal of Banking & Finance* 26(7), 1487-1503.
Adrian, T. and Brunnermeier, M. K. (2016). CoVaR.
    *American Economic Review* 106(7), 1705-1741.
Acharya, V. V., Pedersen, L. H., Philippon, T. and Richardson, M. (2017).
    Measuring systemic risk. *Review of Financial Studies* 30(1), 2-47.
    Marginal expected shortfall.
Tasche, D. (2007). Euler allocation: theory and practice. arXiv:0708.2542.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.base import Copula
from rcopula.distribution import CopulaDistribution, Margin

__all__ = [
    "RiskSummary",
    "covar",
    "delta_covar",
    "diversification_benefit",
    "expected_shortfall",
    "marginal_expected_shortfall",
    "rank_reorder",
    "risk_contributions",
    "simulate_losses",
    "stress_scenario",
    "value_at_risk",
]


class RiskSummary(NamedTuple):
    """VaR and expected shortfall at one confidence level."""

    alpha: float
    var: float
    expected_shortfall: float

    def __repr__(self) -> str:
        return (
            f"RiskSummary(alpha={self.alpha:.4g}, var={self.var:.6g}, "
            f"expected_shortfall={self.expected_shortfall:.6g})"
        )


def value_at_risk(losses: ArrayLike, alpha: float = 0.99) -> float:
    r"""Value at Risk: the ``alpha``-quantile of the loss distribution.

    Parameters
    ----------
    losses : array_like
        Simulated or realised losses. Positive means a loss.
    alpha : float
        Confidence level in ``(0, 1)``, e.g. ``0.99`` for 99% VaR.

    Notes
    -----
    VaR is **not coherent**: it can penalise diversification, because it says
    nothing about how bad the tail is beyond the quantile. Regulatory frameworks
    have been moving to expected shortfall for exactly that reason. It is
    reported here because it is still ubiquitous, not because it is the better
    measure.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.risk import value_at_risk
    >>> losses = np.arange(1, 1001, dtype=float)
    >>> float(value_at_risk(losses, 0.99))
    990.0
    """
    x = np.asarray(losses, dtype=np.float64).ravel()
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    if x.size == 0:
        raise ValueError("cannot compute VaR from an empty sample")
    return float(np.quantile(x, alpha, method="inverted_cdf"))


def expected_shortfall(losses: ArrayLike, alpha: float = 0.99) -> float:
    r"""Expected shortfall (CVaR): the mean loss given that VaR is exceeded.

    .. math::  \mathrm{ES}_\alpha = \mathbb{E}[L \mid L \ge \mathrm{VaR}_\alpha].

    Unlike VaR this is **coherent** -- in particular subadditive, so it never
    penalises diversification -- which is why Basel III replaced 99% VaR with
    97.5% ES.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.risk import expected_shortfall, value_at_risk
    >>> losses = np.arange(1, 1001, dtype=float)
    >>> float(expected_shortfall(losses, 0.99))
    995.0

    ES always dominates VaR at the same level:

    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_t(3, size=100_000)
    >>> bool(expected_shortfall(x, 0.99) > value_at_risk(x, 0.99))
    True
    """
    x = np.asarray(losses, dtype=np.float64).ravel()
    threshold = value_at_risk(x, alpha)
    tail = x[x >= threshold]
    # The tail is never empty: the quantile is attained by at least one point.
    return float(tail.mean())


def rank_reorder(
    samples: ArrayLike,
    copula: Copula,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    r"""Impose a copula's dependence on given marginal samples, by reordering.

    Each column is sorted and then re-ordered to follow the ranks of a draw from
    ``copula``. The marginal distributions survive **exactly** -- the same values
    come back, only rearranged -- while the dependence becomes the copula's.

    This is the standard move in risk aggregation when the marginals are already
    fixed: capital models arrive with an approved loss distribution per business
    line and the question is only how to combine them. Refitting the margins to
    make a joint model tractable would change numbers that have been signed off;
    reordering does not touch them.

    Parameters
    ----------
    samples : array_like
        ``(n, d)`` marginal samples, one column per risk.
    copula : Copula
        The dependence structure to impose.

    Examples
    --------
    The margins are preserved to the last value:

    >>> import numpy as np
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.risk import rank_reorder
    >>> rng = np.random.default_rng(0)
    >>> x = np.column_stack([rng.lognormal(size=5000), rng.exponential(size=5000)])
    >>> y = rank_reorder(x, ClaytonCopula(4.0), random_state=0)
    >>> bool(np.array_equal(np.sort(x, axis=0), np.sort(y, axis=0)))
    True

    ...while the dependence becomes the copula's:

    >>> from scipy import stats
    >>> bool(abs(stats.kendalltau(y[:, 0], y[:, 1]).statistic - 2 / 3) < 0.03)
    True
    """
    x = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    n, d = x.shape
    if d != copula.dim:
        raise ValueError(f"samples have {d} columns but the copula has dim={copula.dim}")

    u = copula.rvs(n, random_state=random_state)
    ranks = np.argsort(np.argsort(u, axis=0), axis=0)
    return np.column_stack([np.sort(x[:, j])[ranks[:, j]] for j in range(d)])


def _summary(losses: NDArray[np.float64], alpha: float) -> RiskSummary:
    return RiskSummary(alpha, value_at_risk(losses, alpha), expected_shortfall(losses, alpha))


def simulate_losses(
    copula: Copula,
    margins: Margin | list[Margin],
    weights: ArrayLike | None = None,
    n: int = 100_000,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    r"""Simulate aggregate portfolio losses under a copula dependence model.

    Draws from the copula, pushes each coordinate through its marginal loss
    distribution, and combines with ``weights``.

    Parameters
    ----------
    copula : Copula
        Dependence between the individual loss drivers.
    margins : sequence of frozen distributions
        Marginal *loss* distributions, one per position.
    weights : array_like, optional
        Position weights. Defaults to equal weighting summing to one. Pass
        exposures directly for an unnormalised total.
    n : int
        Number of simulation draws.

    Returns
    -------
    ndarray
        ``n`` aggregate losses.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.risk import simulate_losses, value_at_risk
    >>> losses = simulate_losses(
    ...     ClaytonCopula(2.0, dim=3), stats.lognorm(0.8), n=50_000, random_state=0
    ... )
    >>> losses.shape
    (50000,)
    >>> bool(value_at_risk(losses, 0.99) > np.mean(losses))
    True
    """
    d = copula.dim
    w = np.full(d, 1.0 / d) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.size != d:
        raise ValueError(f"got {w.size} weight(s) for a copula of dimension {d}")

    joint = CopulaDistribution(copula, margins)
    draws = np.asarray(joint.rvs(n, random_state=random_state), dtype=np.float64)
    return draws @ w


def diversification_benefit(
    copula: Copula,
    margins: Margin | list[Margin],
    alpha: float = 0.99,
    weights: ArrayLike | None = None,
    n: int = 100_000,
    random_state: np.random.Generator | int | None = None,
) -> dict[str, float]:
    r"""How much the dependence structure costs relative to comonotonicity.

    Under the comonotone copula, risk measures are simply additive across
    positions -- that is the worst case, and the benchmark regulators use for a
    "no diversification" capital charge. The benefit is the shortfall against it.

    Returns
    -------
    dict
        ``es``, ``es_comonotone``, ``benefit`` (the absolute reduction) and
        ``benefit_pct``.

    Examples
    --------
    Independence diversifies; strong dependence does not:

    >>> from scipy import stats
    >>> from rcopula import ClaytonCopula, IndependenceCopula
    >>> from rcopula.risk import diversification_benefit
    >>> margins = stats.lognorm(0.6)
    >>> free = diversification_benefit(
    ...     IndependenceCopula(4), margins, n=40_000, random_state=0
    ... )
    >>> tied = diversification_benefit(
    ...     ClaytonCopula(8.0, dim=4), margins, n=40_000, random_state=0
    ... )
    >>> bool(free["benefit_pct"] > tied["benefit_pct"])
    True
    """
    from rcopula.core.other import FrechetUpperCopula

    losses = simulate_losses(copula, margins, weights, n, random_state)
    comonotone = simulate_losses(FrechetUpperCopula(copula.dim), margins, weights, n, random_state)
    es = expected_shortfall(losses, alpha)
    es_como = expected_shortfall(comonotone, alpha)
    return {
        "es": es,
        "es_comonotone": es_como,
        "benefit": es_como - es,
        "benefit_pct": 100.0 * (es_como - es) / es_como if es_como else 0.0,
    }


def risk_contributions(
    copula: Copula,
    margins: Margin | list[Margin],
    weights: ArrayLike | None = None,
    alpha: float = 0.99,
    n: int = 100_000,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    r"""Euler allocation of expected shortfall to individual positions.

    Each position's contribution is its **average loss conditional on the
    portfolio being in its own tail**, weighted by exposure:

    .. math::
        \mathrm{ES}_i = w_i\,\mathbb{E}[L_i \mid L_{\text{portfolio}}
                        \ge \mathrm{VaR}_\alpha].

    Contributions sum exactly to portfolio ES, which is what makes the Euler
    rule the standard allocation: capital charged to desks adds up to the
    capital held. Note that a position can carry a large share of the risk while
    being a small share of the portfolio, if it loads on the same tail events.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.risk import expected_shortfall, risk_contributions, simulate_losses
    >>> cop, margins = ClaytonCopula(3.0, dim=3), stats.lognorm(0.7)
    >>> parts = risk_contributions(cop, margins, n=60_000, random_state=0)
    >>> total = expected_shortfall(
    ...     simulate_losses(cop, margins, n=60_000, random_state=0), 0.99
    ... )
    >>> bool(np.isclose(parts.sum(), total))
    True
    """
    d = copula.dim
    w = np.full(d, 1.0 / d) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    joint = CopulaDistribution(copula, margins)
    draws = np.asarray(joint.rvs(n, random_state=random_state), dtype=np.float64)
    portfolio = draws @ w

    threshold = value_at_risk(portfolio, alpha)
    tail = portfolio >= threshold
    return w * draws[tail].mean(axis=0)


# ======================================================================
# Systemic risk
# ======================================================================


def covar(
    system: ArrayLike,
    firm: ArrayLike,
    alpha: float = 0.95,
    beta: float = 0.95,
    band: float = 0.05,
) -> float:
    r"""CoVaR: the system's VaR conditional on a firm being in distress.

    :math:`\mathrm{CoVaR}_{\alpha|\beta}` is the ``alpha``-quantile of the
    system loss, conditional on the firm's own loss sitting at its
    ``beta``-quantile (Adrian & Brunnermeier 2016).

    Parameters
    ----------
    system, firm : array_like
        Paired loss series.
    alpha : float
        Confidence level for the system's VaR.
    beta : float
        The firm's distress quantile.
    band : float
        Half-width of the conditioning window, as a quantile fraction.
        Conditioning on an exact quantile has probability zero, so a window is
        unavoidable; ``0.05`` keeps roughly 10% of the sample.

    Examples
    --------
    A firm whose losses are tied to the system raises CoVaR above the
    unconditional VaR; an unrelated firm does not:

    >>> import numpy as np
    >>> from rcopula import ClaytonCopula, IndependenceCopula
    >>> from rcopula.risk import covar, value_at_risk
    >>> tied = ClaytonCopula(6.0).rvs(60_000, random_state=0)
    >>> free = IndependenceCopula(2).rvs(60_000, random_state=0)
    >>> base = value_at_risk(free[:, 0], 0.95)
    >>> bool(covar(tied[:, 0], tied[:, 1], 0.95) > covar(free[:, 0], free[:, 1], 0.95))
    True
    """
    s = np.asarray(system, dtype=np.float64).ravel()
    f = np.asarray(firm, dtype=np.float64).ravel()
    if s.size != f.size:
        raise ValueError(f"system and firm must be the same length, got {s.size} and {f.size}")

    lo = np.quantile(f, max(beta - band, 0.0))
    hi = np.quantile(f, min(beta + band, 1.0))
    stressed = s[(f >= lo) & (f <= hi)]
    if stressed.size < 10:
        raise ValueError(
            f"only {stressed.size} observations fall in the conditioning window; "
            "widen `band` or simulate more draws"
        )
    return value_at_risk(stressed, alpha)


def delta_covar(
    system: ArrayLike,
    firm: ArrayLike,
    alpha: float = 0.95,
    beta: float = 0.95,
    band: float = 0.05,
) -> float:
    r"""ΔCoVaR: the firm's *marginal* contribution to system risk.

    The difference between the system's VaR when the firm is in distress and
    when it is at its median. This is the quantity Adrian & Brunnermeier argue
    should drive systemic capital surcharges -- a firm can be small and safe on
    its own yet have a large ΔCoVaR, which is precisely the case unconditional
    measures miss.

    Examples
    --------
    >>> from rcopula import ClaytonCopula, IndependenceCopula
    >>> from rcopula.risk import delta_covar
    >>> tied = ClaytonCopula(6.0).rvs(60_000, random_state=0)
    >>> free = IndependenceCopula(2).rvs(60_000, random_state=0)
    >>> bool(delta_covar(tied[:, 0], tied[:, 1]) > delta_covar(free[:, 0], free[:, 1]))
    True
    """
    distressed = covar(system, firm, alpha, beta, band)
    median = covar(system, firm, alpha, 0.5, band)
    return distressed - median


def marginal_expected_shortfall(firm: ArrayLike, system: ArrayLike, alpha: float = 0.95) -> float:
    r"""MES: a firm's average loss when the *system* is in its tail.

    .. math::
        \mathrm{MES}_\alpha = \mathbb{E}[L_{\text{firm}}
                              \mid L_{\text{system}} \ge \mathrm{VaR}_\alpha].

    The conditioning runs the opposite way to CoVaR -- MES asks what a firm
    loses in a crisis, CoVaR asks what a firm's distress does to the system --
    and the two can rank firms differently.

    Examples
    --------
    >>> from rcopula import ClaytonCopula, IndependenceCopula
    >>> from rcopula.risk import marginal_expected_shortfall as mes
    >>> tied = ClaytonCopula(6.0).rvs(60_000, random_state=0)
    >>> free = IndependenceCopula(2).rvs(60_000, random_state=0)
    >>> bool(mes(tied[:, 0], tied[:, 1]) > mes(free[:, 0], free[:, 1]))
    True
    """
    f = np.asarray(firm, dtype=np.float64).ravel()
    s = np.asarray(system, dtype=np.float64).ravel()
    if f.size != s.size:
        raise ValueError(f"firm and system must be the same length, got {f.size} and {s.size}")
    return float(f[s >= value_at_risk(s, alpha)].mean())


# ======================================================================
# Stress testing
# ======================================================================


def stress_scenario(
    copula: Copula,
    margins: Margin | list[Margin],
    stressed: dict[int, float],
    n: int = 200_000,
    band: float = 0.02,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    r"""Simulate the portfolio conditional on some factors being stressed.

    Answers "if factor 0 hits its 99th percentile, what happens to the rest?"
    -- with the *dependence structure* supplying the answer rather than an
    assumed correlation.

    Parameters
    ----------
    stressed : dict
        Maps column index to the quantile to condition on, e.g.
        ``{0: 0.99}`` for a 99th-percentile shock to the first factor.
    band : float
        Half-width of the conditioning window in quantile units.

    Returns
    -------
    ndarray
        The retained draws, on the marginal scale. Fewer than ``n`` rows: only
        the draws satisfying the conditioning survive.

    Notes
    -----
    Conditioning is done by **rejection on a window**, not analytically. That
    keeps it valid for every family in the package, at the cost of discarding
    most draws -- roughly ``2 * band`` of them survive per conditioned factor,
    so conditioning on several at once needs a large ``n``. An exact
    conditional sampler via the Rosenblatt transform is the natural upgrade.

    Examples
    --------
    Under Clayton, stressing one factor drags the others down with it:

    >>> import numpy as np
    >>> from scipy import stats
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.risk import stress_scenario
    >>> draws = stress_scenario(
    ...     ClaytonCopula(6.0, dim=3), stats.norm(), {0: 0.99}, n=100_000, random_state=0
    ... )
    >>> bool(draws[:, 1].mean() > 0.5)      # unconditionally it would be 0
    True
    """
    if not stressed:
        raise ValueError("`stressed` must name at least one factor to condition on")
    for index in stressed:
        if not 0 <= index < copula.dim:
            raise ValueError(f"factor index {index} is outside 0..{copula.dim - 1}")

    joint = CopulaDistribution(copula, margins)
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    u = copula.rvs(n, random_state=rng)

    keep = np.ones(n, dtype=bool)
    for index, level in stressed.items():
        keep &= (u[:, index] >= level - band) & (u[:, index] <= level + band)

    if keep.sum() < 10:
        raise ValueError(
            f"only {keep.sum()} of {n} draws satisfy the conditioning; increase `n` or widen `band`"
        )
    return np.column_stack([m.ppf(u[keep, j]) for j, m in enumerate(joint.margins)])
