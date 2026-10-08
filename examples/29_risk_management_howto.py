"""How to run risk management on a trading book with copulas, step by step.

A plain-language walk-through for a risk manager. Five desks, one book, and the
questions that come up every morning:

    1. Get each desk's daily P&L history     5. Which desk is the risk coming from?
    2. Fit volatility and dependence         6. How much does diversification buy?
    3. Today's VaR and expected shortfall    7. If one desk blows up, what happens?
    4. Backtest: was yesterday's VaR right?  8. Reverse stress: what does a disaster look like?

The data are simulated so the script runs offline and every claim can be
checked against the truth that generated it. To use your own book, replace
step 1 with a DataFrame of daily desk returns -- nothing else changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import check, heading, show
from scipy import stats

import rcopula as rc
from rcopula.garch import CopulaGarch, fit_garch
from rcopula.risk import (
    delta_covar,
    diversification_benefit,
    risk_contributions,
    value_at_risk,
)

DESKS = ["US equity", "EU equity", "Credit", "Rates", "Commodities"]
EXPOSURE = np.array([40.0, 30.0, 25.0, 40.0, 15.0])  # $ millions in each desk

# ---------------------------------------------------------------------------
heading("Step 1. Get each desk's daily returns")
# ---------------------------------------------------------------------------
# With real data this is a DataFrame of daily returns, one column per desk --
# from your P&L system, or for a proxy book:
#
#     import yfinance as yf
#     prices = yf.download(["SPY", "VGK", "HYG", "IEF", "DBC"], start="2005-01-01")["Close"]
#     returns = np.log(prices).diff().dropna()
#
# Here a hidden "true market" stands in. Its dependence is a Student-t copula
# with 3 degrees of freedom -- desks that crash together far more often than a
# correlation matrix suggests -- and volatility clusters through GARCH.

TRUE_CORR = np.array(
    [
        [1.00, 0.80, 0.55, -0.30, 0.35],
        [0.80, 1.00, 0.55, -0.25, 0.35],
        [0.55, 0.55, 1.00, -0.10, 0.25],
        [-0.30, -0.25, -0.10, 1.00, -0.10],
        [0.35, 0.35, 0.25, -0.10, 1.00],
    ]
)


def simulate_book(n: int, seed: int) -> pd.DataFrame:
    truth = rc.StudentCopula(rc.P2p(TRUE_CORR), dim=5, dispstr="un", df=3.0)
    z = stats.t(5).ppf(truth.rvs(n, random_state=seed)) / np.sqrt(5 / 3)
    daily_vol = np.array([0.011, 0.012, 0.005, 0.004, 0.015])
    omega, alpha, beta = 0.05, 0.08, 0.90
    r = np.empty_like(z)
    s2 = np.ones(len(DESKS))
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return pd.DataFrame(r * daily_vol / np.sqrt(omega / (1 - alpha - beta)), columns=DESKS)


returns = simulate_book(6000, seed=11)
history, future = returns.iloc[:2000], returns.iloc[2000:]  # fit on 8 years, test on 16
show("days to fit on", len(history))
show("days held back for the backtest", len(future))

# ---------------------------------------------------------------------------
heading("Step 2. Fit volatility per desk, then dependence between desks")
# ---------------------------------------------------------------------------
# GARCH handles "how volatile is each desk right now". The copula handles "how
# do desks move together once that is accounted for". Fit two copulas to
# compare: Gaussian (what a correlation matrix assumes) and Student-t (which
# lets desks crash together).

margins = [fit_garch(history[d], dist="t", name=d) for d in DESKS]
u = rc.pseudo_obs(pd.DataFrame({m.name: m.resid for m in margins}))
# "itau" reads each pair's correlation off Kendall's tau; "itau.mpl" does the
# same and then lets the data pick the Student-t's degrees of freedom -- the
# lower, the more often desks crash together.
gauss = rc.fit(rc.GaussianCopula(dim=5, dispstr="un"), u, method="itau").copula
student = rc.fit(rc.StudentCopula(dim=5, dispstr="un"), u, method="itau.mpl").copula
show("Student-t degrees of freedom (true: 3)", student.df)
check("the data ask for fat joint tails (df well below 10)", student.df < 6)

ll_gauss = float(np.sum(gauss.logpdf(u)))
ll_student = float(np.sum(student.logpdf(u)))
show("log-likelihood gain of Student-t over Gaussian", ll_student - ll_gauss)
check("Student-t fits better, by far more than its one extra parameter", ll_student - ll_gauss > 10)

# ---------------------------------------------------------------------------
heading("Step 3. Today's VaR and expected shortfall, in dollars")
# ---------------------------------------------------------------------------
# VaR 99%: the loss you should exceed on only 1 day in 100.
# Expected shortfall (ES) 99%: the average loss on those 1-in-100 days.
# Weights are $ exposures, so the answers come out in $ millions.

model_g = CopulaGarch(margins, gauss, innovations="parametric")
model_t = CopulaGarch(margins, student, innovations="parametric")
risk_g = model_g.forecast_risk(EXPOSURE, alpha=0.99, n=200_000, random_state=0)
risk_t = model_t.forecast_risk(EXPOSURE, alpha=0.99, n=200_000, random_state=0)
show("1-day 99% VaR, Gaussian copula ($m)", risk_g["var"])
show("1-day 99% VaR, Student-t copula ($m)", risk_t["var"])
show("1-day 99% ES, Gaussian copula ($m)", risk_g["expected_shortfall"])
show("1-day 99% ES, Student-t copula ($m)", risk_t["expected_shortfall"])
check(
    "the Gaussian copula reports less tail risk",
    risk_g["expected_shortfall"] < risk_t["expected_shortfall"],
)

# ---------------------------------------------------------------------------
heading("Step 4. Backtest: was each day's VaR right?")
# ---------------------------------------------------------------------------
# Walk forward through 4,000 days the models never saw. Each morning, scale the
# copula's simulated shocks by that day's GARCH volatility to get the VaR; in
# the evening, see whether the real loss broke through it. A 99% VaR should be
# broken on about 1% of days -- 40 here. Kupiec's test asks whether the count
# is too far from 40 to be bad luck.


def garch_vol(m, x: np.ndarray) -> np.ndarray:
    """Each day's volatility forecast, using only the days before it."""
    s2 = np.empty(len(x))
    s2[0] = m.sigma[-1] ** 2
    for t in range(1, len(x)):
        s2[t] = m.omega + m.alpha * (x[t - 1] - m.mu) ** 2 + m.beta * s2[t - 1]
    return np.sqrt(s2)


def kupiec_pvalue(breaks: int, days: int, p: float) -> float:
    """Kupiec (1995) proportion-of-failures test: is the break rate p?"""
    rate = breaks / days
    null = (days - breaks) * np.log(1 - p) + breaks * np.log(p)
    alt = (days - breaks) * np.log(1 - rate) + breaks * np.log(rate)
    return float(stats.chi2.sf(-2 * (null - alt), df=1))


full = pd.concat([history.iloc[-1:], future])  # yesterday's return seeds today's vol
vols = np.column_stack(
    [garch_vol(m, full[m.name].to_numpy())[1:] for m in margins]
)  # (4000 days, 5 desks)
mus = np.array([m.mu for m in margins])
pnl = future.to_numpy() @ EXPOSURE

backtest = {}
for name, cop in [("Gaussian", gauss), ("Student-t", student)]:
    shocks = np.column_stack(
        [
            m.innovation().ppf(col)
            for m, col in zip(margins, cop.rvs(50_000, random_state=1).T, strict=True)
        ]
    )
    # Portfolio P&L on a simulated day = sum over desks of exposure * (mu + vol * shock)
    var_each_day = np.array(
        [-np.quantile((mus + vols[t] * shocks) @ EXPOSURE, 0.01) for t in range(len(future))]
    )
    breaks = int(np.sum(-pnl > var_each_day))
    backtest[name] = breaks
    show(
        f"{name}: VaR breaks (expect 40), Kupiec p",
        f"{breaks}, p = {kupiec_pvalue(breaks, len(future), 0.01):.3f}",
    )

check(
    "the Gaussian copula's VaR is broken more often", backtest["Gaussian"] > backtest["Student-t"]
)
check(
    "the Student-t VaR passes Kupiec at 5%",
    kupiec_pvalue(backtest["Student-t"], len(future), 0.01) > 0.05,
)

# Be honest about what this shows: the gap is a handful of breaks out of 4,000
# days, and even the Gaussian is not formally rejected here. Rerun with other
# seeds and the Gaussian is broken a few times more in every one, but whether
# either passes Kupiec moves around with the luck of the history -- daily 99%
# VaR is driven mostly by volatility, which both models share through GARCH.
# The copula earns its keep deeper in the tail: steps 3, 6, 7 and 8.

# ---------------------------------------------------------------------------
heading("Step 5. Which desk is the risk coming from?")
# ---------------------------------------------------------------------------
# Euler allocation: each desk's share of ES is its average loss on the book's
# worst days. The shares add up exactly to the total, so they can be used to
# charge capital to desks. A desk's share of *risk* can be very different from
# its share of *capital*.

today = np.array([m.forecast_vol(1)[0] for m in margins])
loss_margins = [  # each desk's loss per $1 tomorrow: minus (mean + vol * shock)
    stats.t(m.df, loc=-m.mu, scale=s * np.sqrt((m.df - 2) / m.df))
    for m, s in zip(margins, today, strict=True)
]
parts = risk_contributions(student, loss_margins, EXPOSURE, alpha=0.99, n=200_000, random_state=2)
table = pd.DataFrame(
    {
        "exposure %": 100 * EXPOSURE / EXPOSURE.sum(),
        "ES contribution $m": parts,
        "ES share %": 100 * parts / parts.sum(),
    },
    index=DESKS,
).round(2)
print(table.to_string())
check(
    "the equity desks carry more of the risk than of the exposure",
    table.loc[["US equity", "EU equity"], "ES share %"].sum()
    > table.loc[["US equity", "EU equity"], "exposure %"].sum(),
)
check(
    "rates is over a quarter of the exposure but almost none of the risk",
    abs(table.loc["Rates", "ES share %"]) < 2,
)

# ---------------------------------------------------------------------------
heading("Step 6. How much does diversification actually buy?")
# ---------------------------------------------------------------------------
# Compare the book's ES with the worst case where every desk has its bad day at
# once (the sum of standalone ES). The gap is the diversification benefit --
# and a model with fat joint tails gives you less of it.

div_g = diversification_benefit(gauss, loss_margins, 0.99, EXPOSURE, n=200_000, random_state=3)
div_t = diversification_benefit(student, loss_margins, 0.99, EXPOSURE, n=200_000, random_state=3)
show("ES if every desk loses at once ($m)", div_t["es_comonotone"])
show("diversification benefit, Gaussian copula", f"{div_g['benefit_pct']:.1f}%")
show("diversification benefit, Student-t copula", f"{div_t['benefit_pct']:.1f}%")
check("joint crashes eat into the benefit", div_t["benefit_pct"] < div_g["benefit_pct"])

# ---------------------------------------------------------------------------
heading("Step 7. If one desk is in trouble, what happens to the book?")
# ---------------------------------------------------------------------------
# Delta-CoVaR: how much the *rest of the book's* VaR rises when one desk is
# having a 1-in-20 bad day, compared with an ordinary day for that desk. It
# measures contagion, not size -- a desk can be small and still drag everything
# down with it.

sims = rc.CopulaDistribution(student, loss_margins).rvs(400_000, random_state=4) * EXPOSURE
sims_g = rc.CopulaDistribution(gauss, loss_margins).rvs(400_000, random_state=4) * EXPOSURE


def contagion(draws: np.ndarray) -> pd.Series:
    book_total = draws.sum(axis=1)
    return pd.Series(
        [delta_covar(book_total - draws[:, j], draws[:, j]) for j in range(len(DESKS))],
        index=DESKS,
    )


spread = pd.DataFrame({"Student-t": contagion(sims), "Gaussian": contagion(sims_g)}).round(2)
print(spread.rename_axis("Delta-CoVaR of the rest, $m").to_string())
check(
    "under Student-t, trouble on any desk raises the rest of the book's VaR",
    (spread["Student-t"] > 0).all(),
)
check("the rates desk spreads the least", spread["Student-t"].idxmin() == "Rates")

# The rates line is the lesson. Under the Gaussian copula a bad day for rates
# is a *good* day for the equity-heavy rest of the book, so its VaR falls. Under
# the Student-t it rises: a 1-in-20 move anywhere signals a turbulent day
# everywhere, and turbulence hurts the rest of the book whichever way it goes.
# A hedge that works on average does not necessarily work in a crisis.
check(
    "the Gaussian copula says rates trouble calms the rest of the book; Student-t disagrees",
    spread.loc["Rates", "Gaussian"] < spread.loc["Rates", "Student-t"],
)

# ---------------------------------------------------------------------------
heading("Step 8. Reverse stress test: what does a disaster look like?")
# ---------------------------------------------------------------------------
# Instead of "what if equities fall 10%", ask the model: of all the simulated
# days that lose more than $X, what typically happened? Here X is the book's
# 1-in-1,000-day loss (about once in four years).

book = sims.sum(axis=1)
disaster = value_at_risk(book, 0.999)
worst = sims[book > disaster]
show("1-in-1,000-day loss threshold ($m)", disaster)
show("average loss on those days ($m)", float(worst.sum(axis=1).mean()))
print(pd.Series(worst.mean(axis=0), index=DESKS).round(2).to_frame("average loss $m").to_string())
check(
    "a disaster is the equity and credit desks losing together",
    all(worst[:, DESKS.index(d)].mean() > 0 for d in ("US equity", "EU equity", "Credit")),
)

# How often does the Gaussian model think a loss that big happens?
book_g = sims_g.sum(axis=1)
p_t, p_g = float(np.mean(book > disaster)), float(np.mean(book_g > disaster))
show("Student-t: one day in", f"{1 / p_t:,.0f}")
show("Gaussian:  one day in", f"{1 / p_g:,.0f}")
check("the Gaussian copula calls the disaster rarer than it is", p_g < p_t)
