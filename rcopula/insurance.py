r"""Insurance, actuarial and operational-risk utilities.

Insurance is where copulas are least optional. An insurer's aggregate loss is a
**compound** distribution -- a random number of claims, each of random size --
and compound distributions have no closed form worth using, no elliptical
structure, and margins that are wildly non-normal. There is no covariance matrix
that summarises how two lines of business co-move; there is only the joint
distribution, and a copula is how you write one down.

The regulatory question is also the tail, not the middle. Solvency II asks for
the 99.5% one-year VaR and Basel's operational-risk framework asks for 99.9%.
At those quantiles the difference between "correlated" and "tail-dependent" is
the whole answer.

=================================  ===========================================
:func:`aggregate_loss`             Compound frequency-severity distribution.
:func:`operational_risk_capital`   Loss-distribution approach across cells.
:func:`excess_of_loss`             Reinsurance layer recovery.
:func:`layer_statistics`           Expected loss and attachment probabilities.
:func:`reinsurance_premium`        Premium for a layer, with loading.
:func:`catastrophe_bond`           Cat bond / ILS payoff analysis.
=================================  ===========================================

References
----------
Klugman, S. A., Panjer, H. H. and Willmot, G. E. (2019). *Loss Models: From
    Data to Decisions*, 5th ed. Wiley.
Frees, E. W. and Valdez, E. A. (1998). Understanding relationships using
    copulas. *North American Actuarial Journal* 2(1), 1-25.
    The loss-ALAE study that introduced copulas to actuarial work.
Frachot, A., Roncalli, T. and Salomon, E. (2004). The correlation problem in
    operational risk. Groupe de Recherche Operationnelle, Credit Lyonnais.
Embrechts, P., Kluppelberg, C. and Mikosch, T. (1997). *Modelling Extremal
    Events for Insurance and Finance*. Springer.
Cummins, J. D. (2008). CAT bonds and other risk-linked securities.
    *Risk Management and Insurance Review* 11(1), 23-47.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.base import Copula
from rcopula.risk import expected_shortfall, rank_reorder, value_at_risk

__all__ = [
    "LayerStatistics",
    "Variate",
    "aggregate_loss",
    "catastrophe_bond",
    "excess_of_loss",
    "layer_statistics",
    "operational_risk_capital",
    "reinsurance_premium",
]


@runtime_checkable
class Variate(Protocol):
    """What a frequency or severity distribution must provide: the ability to draw.

    Deliberately weaker than :class:`~rcopula.distribution.Margin`, which needs
    ``cdf``/``pdf``/``ppf``. A compound distribution is built by *simulation*, so
    all that is required here is ``rvs`` -- satisfied by every ``scipy.stats``
    frozen distribution, discrete or continuous, and by anything else that draws.
    """

    def rvs(self, size: Any = ..., random_state: Any = ...) -> Any: ...


class LayerStatistics(NamedTuple):
    """Summary of a reinsurance layer."""

    attachment: float
    limit: float
    expected_loss: float
    attachment_probability: float
    exhaustion_probability: float
    expected_loss_ratio: float

    def __repr__(self) -> str:
        return (
            f"LayerStatistics({self.attachment:g} xs {self.limit:g}, "
            f"EL={self.expected_loss:.4g}, "
            f"P(attach)={self.attachment_probability:.4%}, "
            f"P(exhaust)={self.exhaustion_probability:.4%})"
        )


def aggregate_loss(
    frequency: Variate,
    severity: Variate,
    n: int = 100_000,
    random_state: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    r"""Simulate a compound frequency-severity aggregate loss.

    :math:`S = \sum_{i=1}^{N} X_i` with :math:`N` from ``frequency`` and each
    :math:`X_i` from ``severity``. This is the actuarial workhorse: no closed
    form in general, but its first two moments are exact and worth checking
    against,

    .. math::
        \mathbb{E}[S] = \mathbb{E}[N]\,\mathbb{E}[X], \qquad
        \mathrm{Var}[S] = \mathbb{E}[N]\,\mathrm{Var}[X]
                          + \mathrm{Var}[N]\,\mathbb{E}[X]^2.

    Parameters
    ----------
    frequency : frozen discrete distribution
        Claim count, e.g. ``scipy.stats.poisson(50)``.
    severity : frozen continuous distribution
        Individual claim size, e.g. ``scipy.stats.lognorm(1.5, scale=1000)``.

    Examples
    --------
    The compound moments come out right:

    >>> import numpy as np
    >>> from scipy import stats
    >>> from rcopula.insurance import aggregate_loss
    >>> freq, sev = stats.poisson(40), stats.lognorm(0.8, scale=1000)
    >>> s = aggregate_loss(freq, sev, n=60_000, random_state=0)
    >>> expected = freq.mean() * sev.mean()
    >>> bool(abs(s.mean() / expected - 1) < 0.02)
    True

    The aggregate is right-skewed even when the severity is only mildly so:

    >>> bool(stats.skew(s) > 0)
    True
    """
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    counts = np.asarray(frequency.rvs(size=n, random_state=rng), dtype=np.int64)
    if np.any(counts < 0):
        raise ValueError("frequency produced a negative claim count")

    total = int(counts.sum())
    if total == 0:
        return np.zeros(n)

    claims = np.asarray(severity.rvs(size=total, random_state=rng), dtype=np.float64)
    # Segment the flat claim vector by policy-year using the cumulative counts.
    boundaries = np.concatenate([[0], np.cumsum(counts)])
    cumulative = np.concatenate([[0.0], np.cumsum(claims)])
    return cumulative[boundaries[1:]] - cumulative[boundaries[:-1]]


def operational_risk_capital(
    cells: list[tuple[Variate, Variate]],
    copula: Copula | None = None,
    alpha: float = 0.999,
    n: int = 100_000,
    random_state: np.random.Generator | int | None = None,
) -> dict[str, float]:
    r"""Operational-risk capital by the loss-distribution approach.

    Each *cell* -- a business line crossed with an event type -- gets its own
    compound distribution. The cells are then combined under ``copula``, by
    rank reordering, so **each cell's approved loss distribution is preserved
    exactly** while the dependence becomes the copula's. That matters in
    practice: cell distributions are validated and signed off individually, and
    a joint model that quietly changes them is not usable.

    Capital is the difference between the aggregate VaR and the expected loss,
    which is the Basel definition -- expected loss is provisioned, not
    capitalised.

    Parameters
    ----------
    cells : list of (frequency, severity)
        One pair of frozen distributions per cell.
    copula : Copula, optional
        Dependence across cells. ``None`` means independence, which is the
        assumption that produces the largest diversification credit.
    alpha : float
        Confidence level; 0.999 is the Basel operational-risk standard.

    Returns
    -------
    dict
        ``var``, ``expected_shortfall``, ``expected_loss``, ``capital``, and
        ``diversification_benefit`` against the sum of standalone capitals.

    Examples
    --------
    Dependence between cells consumes the diversification credit:

    >>> from scipy import stats
    >>> from rcopula import GaussianCopula
    >>> from rcopula.insurance import operational_risk_capital
    >>> cells = [(stats.poisson(30), stats.lognorm(1.2, scale=500))] * 3
    >>> free = operational_risk_capital(cells, None, n=40_000, random_state=0)
    >>> tied = operational_risk_capital(
    ...     cells, GaussianCopula(0.8, dim=3), n=40_000, random_state=0)
    >>> bool(tied["capital"] > free["capital"])
    True
    >>> bool(free["diversification_benefit"] > tied["diversification_benefit"])
    True
    """
    if not cells:
        raise ValueError("need at least one cell")
    if copula is not None and copula.dim != len(cells):
        raise ValueError(f"copula has dim={copula.dim} but {len(cells)} cell(s) were supplied")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    per_cell = np.column_stack([aggregate_loss(freq, sev, n, rng) for freq, sev in cells])
    if copula is not None:
        per_cell = rank_reorder(per_cell, copula, random_state=rng)

    total = per_cell.sum(axis=1)
    var = value_at_risk(total, alpha)
    mean = float(total.mean())

    standalone = sum(
        value_at_risk(per_cell[:, j], alpha) - float(per_cell[:, j].mean())
        for j in range(per_cell.shape[1])
    )
    capital = var - mean
    return {
        "var": var,
        "expected_shortfall": expected_shortfall(total, alpha),
        "expected_loss": mean,
        "capital": capital,
        "standalone_capital": standalone,
        "diversification_benefit": standalone - capital,
    }


# ======================================================================
# Reinsurance
# ======================================================================


def excess_of_loss(losses: ArrayLike, attachment: float, limit: float) -> NDArray[np.float64]:
    r"""Reinsurer's recovery on an excess-of-loss layer.

    A "``limit`` excess of ``attachment``" layer pays
    :math:`\min(\max(L - a, 0), \ell)`: nothing until the cedant's retention is
    exhausted, then pound for pound, then nothing once the limit is used up.
    Structurally identical to a CDO tranche, which is not a coincidence -- both
    are call spreads on an aggregate loss.

    Examples
    --------
    >>> import numpy as np
    >>> from rcopula.insurance import excess_of_loss
    >>> losses = np.array([0.0, 50.0, 150.0, 400.0, 1000.0])
    >>> excess_of_loss(losses, attachment=100.0, limit=200.0)
    array([  0.,   0.,  50., 200., 200.])
    """
    if attachment < 0 or limit <= 0:
        raise ValueError(f"need attachment >= 0 and limit > 0, got {attachment} and {limit}")
    x = np.asarray(losses, dtype=np.float64)
    return np.clip(x - attachment, 0.0, limit)


def layer_statistics(losses: ArrayLike, attachment: float, limit: float) -> LayerStatistics:
    """Expected loss and attachment probabilities for a layer.

    Examples
    --------
    >>> from scipy import stats
    >>> from rcopula.insurance import aggregate_loss, layer_statistics
    >>> s = aggregate_loss(stats.poisson(40), stats.lognorm(1.0, scale=1000),
    ...                    n=40_000, random_state=0)
    >>> low = layer_statistics(s, 60_000, 20_000)
    >>> high = layer_statistics(s, 150_000, 20_000)
    >>> bool(low.attachment_probability > high.attachment_probability)
    True
    >>> bool(low.expected_loss > high.expected_loss)
    True
    """
    x = np.asarray(losses, dtype=np.float64)
    recovery = excess_of_loss(x, attachment, limit)
    expected = float(recovery.mean())
    return LayerStatistics(
        attachment=float(attachment),
        limit=float(limit),
        expected_loss=expected,
        attachment_probability=float(np.mean(x > attachment)),
        exhaustion_probability=float(np.mean(x >= attachment + limit)),
        expected_loss_ratio=expected / limit,
    )


def reinsurance_premium(
    losses: ArrayLike,
    attachment: float,
    limit: float,
    loading: float = 0.25,
    method: str = "expected_value",
    alpha: float = 0.99,
) -> float:
    r"""Premium for an excess-of-loss layer.

    Three classical principles:

    ``expected_value``
        :math:`(1 + \theta)\,\mathbb{E}[R]`. Simple, and the usual starting
        point, but it charges the same for two layers with equal mean and wildly
        different volatility.
    ``standard_deviation``
        :math:`\mathbb{E}[R] + \theta\,\mathrm{sd}(R)`. Prices the uncertainty.
    ``expected_shortfall``
        :math:`(1-\theta)\mathbb{E}[R] + \theta\,\mathrm{ES}_\alpha(R)`.
        Prices the tail, which is what a high layer is actually selling.

    Examples
    --------
    Tail-sensitive principles charge more for a volatile layer:

    >>> from scipy import stats
    >>> from rcopula.insurance import aggregate_loss, reinsurance_premium
    >>> s = aggregate_loss(stats.poisson(40), stats.lognorm(1.2, scale=1000),
    ...                    n=40_000, random_state=0)
    >>> ev = reinsurance_premium(s, 100_000, 50_000, method="expected_value")
    >>> es = reinsurance_premium(s, 100_000, 50_000, method="expected_shortfall")
    >>> bool(es > ev)
    True
    """
    recovery = excess_of_loss(losses, attachment, limit)
    mean = float(recovery.mean())

    if method == "expected_value":
        return (1.0 + loading) * mean
    if method == "standard_deviation":
        return mean + loading * float(recovery.std(ddof=1))
    if method == "expected_shortfall":
        return (1.0 - loading) * mean + loading * expected_shortfall(recovery, alpha)
    raise ValueError(
        "method must be 'expected_value', 'standard_deviation' or "
        f"'expected_shortfall', got {method!r}"
    )


# ======================================================================
# Insurance-linked securities
# ======================================================================


def catastrophe_bond(
    losses: ArrayLike,
    attachment: float,
    exhaustion: float,
    coupon: float = 0.06,
    risk_free: float = 0.03,
    maturity: float = 1.0,
) -> dict[str, float]:
    r"""Analyse a catastrophe bond with an indemnity trigger.

    Principal is written down linearly between ``attachment`` and
    ``exhaustion``, so the investor is short a call spread on the sponsor's
    loss -- the mirror image of :func:`excess_of_loss`.

    Returns the three numbers the ILS market quotes: **expected loss** (the
    fraction of principal expected to be lost), **attachment probability**, and
    the **multiple** -- spread divided by expected loss -- which is how relative
    value is judged across deals.

    Examples
    --------
    A remote layer has a low expected loss and therefore a high multiple:

    >>> from scipy import stats
    >>> from rcopula.insurance import aggregate_loss, catastrophe_bond
    >>> s = aggregate_loss(stats.poisson(20), stats.lognorm(1.5, scale=2000),
    ...                    n=60_000, random_state=0)
    >>> near = catastrophe_bond(s, 60_000, 120_000)
    >>> far = catastrophe_bond(s, 200_000, 300_000)
    >>> bool(far["expected_loss"] < near["expected_loss"])
    True
    >>> bool(far["multiple"] > near["multiple"])
    True
    """
    if not 0 <= attachment < exhaustion:
        raise ValueError(f"need 0 <= attachment < exhaustion, got {attachment} and {exhaustion}")
    x = np.asarray(losses, dtype=np.float64)
    width = exhaustion - attachment
    principal_lost = np.clip(x - attachment, 0.0, width) / width

    expected = float(principal_lost.mean())
    spread = coupon - risk_free
    return {
        "expected_loss": expected,
        "attachment_probability": float(np.mean(x > attachment)),
        "exhaustion_probability": float(np.mean(x >= exhaustion)),
        "conditional_severity": float(
            principal_lost[principal_lost > 0].mean() if np.any(principal_lost > 0) else 0.0
        ),
        "spread": spread,
        # Spread per unit of expected loss: the ILS market's relative-value yardstick.
        "multiple": spread / expected if expected > 0 else float("inf"),
        "expected_return": spread - expected,
        "fair_price": float(np.exp(-risk_free * maturity) * (1.0 + coupon * maturity - expected)),
    }
