"""Conditional copula probabilities as a trading signal.

The classical spread trade standardises a price difference, which assumes the
two assets are jointly normal with a stable linear relationship. The copula
version asks a sharper question: how unlikely is asset 1 being this low *given
where asset 2 actually is*.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc

heading("The signal is a probability, so it means the same thing everywhere")

cop = rc.StudentCopula(0.85, df=5.0)
u = cop.rvs(3000, random_state=0)
first, second = rc.portfolio.mispricing_index(cop, u)

show("h(u1 | u2) mean", float(first.mean()))
show("h(u2 | u1) mean", float(second.mean()))

# Under the true copula these are uniform -- that is what makes a fixed
# threshold mean the same thing across pairs with different dependence.
for label, series in [("h(u1|u2)", first), ("h(u2|u1)", second)]:
    p = stats.kstest(series, "uniform").pvalue
    check(f"{label} is uniform (KS p = {p:.3f})", p > 0.01)

heading("A worked signal")

# A point where asset 1 is mid-range but asset 2 is high: unremarkable
# marginally, strongly mispriced conditionally.
points = np.array([[0.50, 0.50], [0.50, 0.95], [0.50, 0.05], [0.05, 0.50]])
print(f"  {'u1':>6}{'u2':>6}{'h(u1|u2)':>12}{'reading':>28}")
for point in points:
    h = float(rc.conditional_cdf(cop, [point], given=1)[0])
    reading = (
        "asset 1 cheap given asset 2"
        if h < 0.05
        else "asset 1 rich given asset 2"
        if h > 0.95
        else "nothing to do"
    )
    print(f"  {point[0]:>6.2f}{point[1]:>6.2f}{h:>12.4f}{reading:>28}")

check(
    "a mid-range asset 1 IS a signal when asset 2 is extreme",
    rc.conditional_cdf(cop, [[0.50, 0.95]], given=1)[0] < 0.05,
)
check(
    "and the same asset 1 is not a signal when asset 2 is mid-range",
    0.4 < rc.conditional_cdf(cop, [[0.50, 0.50]], given=1)[0] < 0.6,
)

heading("A walk-forward backtest")

# Cointegrated-ish pair: a shared factor plus idiosyncratic noise, with the
# spread mean-reverting. Simulated, so the point is the machinery, not the
# Sharpe ratio.
rng = np.random.default_rng(0)
n = 1500
common = rng.normal(0, 0.01, n)
spread = np.zeros(n)
for t in range(1, n):
    spread[t] = 0.92 * spread[t - 1] + rng.normal(0, 0.008)
returns = np.column_stack(
    [
        common + np.diff(np.concatenate([[0.0], spread])),
        common - np.diff(np.concatenate([[0.0], spread])),
    ]
)

for family in (rc.StudentCopula(dim=2), rc.ClaytonCopula(), rc.GaussianCopula()):
    result = rc.portfolio.backtest_pairs(returns, family, train=250, entry=0.05)
    print(
        f"  {family.name:<10} trades {result.n_trades:4d}   "
        f"total return {result.total_return:+8.4f}   "
        f"Sharpe {result.annualised_sharpe:+7.3f}   hit {result.hit_rate:.3f}"
    )

result = rc.portfolio.backtest_pairs(returns, rc.StudentCopula(dim=2), train=250, entry=0.05)
check("the backtest produced trades", result.n_trades > 0)
check("and a finite Sharpe ratio", np.isfinite(result.annualised_sharpe))

heading("The signal cannot see the future")

# A look-ahead bug is the classic way a pairs backtest lies. Changing data
# strictly after time t must not change the signal at time t.
tampered = returns.copy()
tampered[1200:] *= 5.0
original = rc.portfolio.backtest_pairs(returns, rc.StudentCopula(dim=2), train=250)
altered = rc.portfolio.backtest_pairs(tampered, rc.StudentCopula(dim=2), train=250)
check(
    "positions before the tampering are identical",
    np.array_equal(original.positions[:1200], altered.positions[:1200]),
)
