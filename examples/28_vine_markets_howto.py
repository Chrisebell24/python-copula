"""How to use a vine copula on market returns, step by step.

A plain-language walk-through for someone who has never fitted a copula. It
goes from a table of daily returns to the answers a desk actually asks for:

    1. Get returns                     5. Simulate tomorrow and measure risk
    2. Strip out volatility (GARCH)    6. Ask "how often do they crash together?"
    3. Fit the vine                    7. Stress test: "if stocks fall 3 sigma..."
    4. Read what it found              8. Build a crash-aware portfolio

The data are simulated so the script runs offline and every claim can be
checked against the truth that generated it. To use real prices, replace step 1
with the yfinance lines in the comment there -- nothing else changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import check, heading, show
from scipy import stats

import rcopula as rc
from rcopula.garch import CopulaGarch, fit_garch
from rcopula.portfolio import mean_cvar_weights
from rcopula.risk import expected_shortfall

TICKERS = ["SPY", "QQQ", "IWM", "HYG", "TLT"]  # large caps, tech, small caps, credit, bonds

# ---------------------------------------------------------------------------
heading("Step 1. Get a table of daily returns")
# ---------------------------------------------------------------------------
# With real data this step is just:
#
#     import yfinance as yf
#     prices = yf.download(TICKERS, start="2015-01-01")["Close"]
#     returns = np.log(prices).diff().dropna()
#
# Here a hidden "true market" stands in for it: a C-vine rooted at SPY, so the
# S&P drives everything, with equities crashing together (Clayton, Student-t)
# and Treasuries moving the other way (negative Frank). Volatility clusters
# through a GARCH recursion, exactly as real returns do.


def simulate_market(n: int, seed: int) -> pd.DataFrame:
    truth = rc.VineCopula(
        [
            [  # tree 1: everything against SPY
                rc.StudentCopula(0.88, df=4.0),  # QQQ: very close, crashes and rallies together
                rc.ClaytonCopula(2.5),  # IWM: small caps fall harder with SPY
                rc.ClaytonCopula(1.2),  # HYG: credit sells off in equity crashes
                rc.FrankCopula(-2.0),  # TLT: bonds rally when stocks fall
            ],
            [rc.GaussianCopula(0.2), rc.GaussianCopula(0.15), rc.FrankCopula(-0.5)],
            [rc.IndependenceCopula(2), rc.IndependenceCopula(2)],
            [rc.IndependenceCopula(2)],
        ],
        structure="C",
    )
    z = stats.t(5).ppf(truth.rvs(n, random_state=seed)) / np.sqrt(5 / 3)
    daily_vol = np.array([0.011, 0.014, 0.014, 0.005, 0.009])
    omega, alpha, beta = 0.05, 0.08, 0.90
    r = np.empty_like(z)
    s2 = np.ones(len(TICKERS))
    for t in range(n):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return pd.DataFrame(r * daily_vol / np.sqrt(omega / (1 - alpha - beta)), columns=TICKERS)


returns = simulate_market(2500, seed=7)  # ten years of trading days
show("days x assets", returns.shape)
print(returns.describe().loc[["mean", "std", "min", "max"]].round(4).to_string())

# ---------------------------------------------------------------------------
heading("Step 2. Strip out volatility with GARCH, keep the surprises")
# ---------------------------------------------------------------------------
# Calm and stormy periods hit every asset at once. Fit a copula to raw returns
# and it reads that shared volatility as dependence. So filter each series
# first: what remains -- the standardised residual -- is "how surprising was
# today, given how volatile things already were".

margins = [fit_garch(returns[c], dist="t", name=c) for c in TICKERS]
residuals = pd.DataFrame({m.name: m.resid for m in margins})
for m in margins:
    show(f"{m.name}: persistence (alpha+beta), t df", f"{m.persistence:.3f}, {m.df:.1f}")
check("volatility is persistent in every series", all(m.persistence > 0.9 for m in margins))

# A copula only sees ranks. pseudo_obs turns each column into numbers in (0, 1):
# 0.01 means "a 1-in-100 bad day for this asset", 0.99 a 1-in-100 good one.
u = rc.pseudo_obs(residuals)

# ---------------------------------------------------------------------------
heading("Step 3. Fit the vine")
# ---------------------------------------------------------------------------
# One line. A C-vine puts the most connected asset at the centre (here it will
# find SPY on its own) and links every other asset to it; higher trees model
# what is left once SPY is accounted for. Each link gets its own family,
# chosen by AIC from Gaussian, Student-t, Clayton, Gumbel, Frank or none.

vine = rc.fit_vine(u, structure="C")
print(vine.describe())
check("it put SPY at the centre", TICKERS[vine.order[0]] == "SPY")

# ---------------------------------------------------------------------------
heading("Step 4. Read what it found")
# ---------------------------------------------------------------------------
# Tree 1 is the part to read. Each link says how strongly an asset moves with
# SPY (Kendall's tau, -1 to 1) and how likely they are to crash or rally
# *together* (tail dependence, 0 to 1). A Gaussian copula forces both tails to
# zero; that single assumption is why it underestimates joint crashes.

root = TICKERS[vine.order[0]]
rows = []
for i, pair in enumerate(vine.pair_copulas[0]):
    asset = TICKERS[vine.order[i + 1]]
    tail = pair.lambda_()
    rows.append(
        {
            "link": f"{root}-{asset}",
            "family": pair.name,
            "tau": round(pair.tau(), 2),
            "crash together": round(tail.lower, 2),
            "rally together": round(tail.upper, 2),
        }
    )
links = pd.DataFrame(rows).set_index("link")
print(links.to_string())
check(
    "SPY-IWM shares crashes (lower tail) more than rallies",
    links.loc["SPY-IWM", "crash together"] > links.loc["SPY-IWM", "rally together"] + 0.2,
)
check("SPY-TLT moves in opposite directions", links.loc["SPY-TLT", "tau"] < 0)

# Is the extra flexibility worth its parameters? Compare with the Gaussian
# copula, which is what a correlation matrix implicitly assumes.
gauss = rc.fit(rc.GaussianCopula(dim=5, dispstr="un"), u).copula
k_vine = sum(c.n_params for level in vine.pair_copulas for c in level)
aic_vine = 2 * k_vine - 2 * vine.loglik(u)
aic_gauss = 2 * gauss.n_params - 2 * float(np.sum(gauss.logpdf(u)))
show("AIC, vine (lower is better)", aic_vine)
show("AIC, Gaussian copula", aic_gauss)
check("the vine fits better", aic_vine < aic_gauss)

# ---------------------------------------------------------------------------
heading("Step 5. Simulate tomorrow and measure portfolio risk")
# ---------------------------------------------------------------------------
# Glue the GARCH margins and the vine back together. The model starts from
# *today's* volatility, so the risk number moves with the market.

weights = np.array([0.30, 0.15, 0.15, 0.10, 0.30])
model_vine = CopulaGarch(margins, vine, innovations="parametric")
model_gauss = CopulaGarch(margins, gauss, innovations="parametric")
risk_vine = model_vine.forecast_risk(weights, alpha=0.99, n=200_000, random_state=0)
risk_gauss = model_gauss.forecast_risk(weights, alpha=0.99, n=200_000, random_state=0)
show("1-day 99% VaR, vine", f"{risk_vine['var']:.2%}")
show("1-day 99% VaR, Gaussian copula", f"{risk_gauss['var']:.2%}")
show("1-day 99% expected shortfall, vine", f"{risk_vine['expected_shortfall']:.2%}")
show("1-day 99% expected shortfall, Gaussian", f"{risk_gauss['expected_shortfall']:.2%}")
check(
    "the Gaussian copula understates the bad-day loss",
    risk_vine["expected_shortfall"] > risk_gauss["expected_shortfall"],
)

# ---------------------------------------------------------------------------
heading("Step 6. How often do the three equity funds crash together?")
# ---------------------------------------------------------------------------
# "All of SPY, QQQ and IWM have a 1-in-50 bad day at once." Count it in the
# data, then ask each model.

eq = [TICKERS.index(t) for t in ("SPY", "QQQ", "IWM")]


def joint_crash(v: np.ndarray, q: float = 0.02) -> float:
    return float(np.mean(np.all(np.asarray(v)[:, eq] < q, axis=1)))


observed = joint_crash(u)
p_vine = joint_crash(vine.rvs(400_000, random_state=1))
p_gauss = joint_crash(gauss.rvs(400_000, random_state=1))
show("in the data", f"{observed:.3%}")
show("vine", f"{p_vine:.3%}")
show("Gaussian copula", f"{p_gauss:.3%}")
show("if they were independent", f"{0.02**3:.4%}")
check(
    "the vine is closer to the data than the Gaussian",
    abs(p_vine - observed) < abs(p_gauss - observed),
)

# ---------------------------------------------------------------------------
heading("Step 7. Stress test: if SPY has a 1-in-100 day, what happens to the rest?")
# ---------------------------------------------------------------------------
# Simulate many days, keep the ones where SPY is at its worst 1%, and look at
# everything else on those days. This is a reverse question a correlation
# matrix answers badly, because it assumes the dependence is the same in a
# crash as on an ordinary Tuesday.

draws = vine.rvs(500_000, random_state=2)
spy = TICKERS.index("SPY")
crash_days = draws[draws[:, spy] < 0.01]
for name in ("QQQ", "IWM", "HYG", "TLT"):
    j = TICKERS.index(name)
    worst_day_odds = float(np.mean(crash_days[:, j] < 0.05))
    show(f"{name}: chance of its own worst-5% day", f"{worst_day_odds:.0%}")
check(
    "small caps almost always crash with SPY",
    np.mean(crash_days[:, TICKERS.index("IWM")] < 0.05) > 0.6,
)
check("bonds rarely do", np.mean(crash_days[:, TICKERS.index("TLT")] < 0.05) < 0.05)

# ---------------------------------------------------------------------------
heading("Step 8. Build a portfolio that minimises the bad-day loss")
# ---------------------------------------------------------------------------
# Feed simulated days to an optimiser that minimises expected shortfall (the
# average loss on the worst 1% of days), with no more than 40% in any one fund.
# Do it twice -- once on scenarios from the vine, once on scenarios from the
# Gaussian copula -- and score both portfolios on fresh vine scenarios.

bounds = (0.0, 0.40)
w_vine = mean_cvar_weights(model_vine.forecast(1, 5_000, random_state=3), 0.99, bounds=bounds)
w_gauss = mean_cvar_weights(model_gauss.forecast(1, 5_000, random_state=3), 0.99, bounds=bounds)
print(pd.DataFrame({"vine": w_vine, "Gaussian": w_gauss}, index=TICKERS).round(3).to_string())

fresh = model_vine.forecast(1, 200_000, random_state=4)
es_vine = expected_shortfall(-(fresh @ w_vine), 0.99)
es_gauss = expected_shortfall(-(fresh @ w_gauss), 0.99)
show("99% expected shortfall, vine-optimised", f"{es_vine:.2%}")
show("99% expected shortfall, Gaussian-optimised", f"{es_gauss:.2%}")
check(
    "weights are long-only, capped and sum to one",
    np.all(w_vine >= -1e-9) and np.all(w_vine <= 0.4 + 1e-9) and abs(w_vine.sum() - 1) < 1e-6,
)
check(
    "the vine portfolio holds more of the crash hedge (TLT)",
    w_vine[TICKERS.index("TLT")] > w_gauss[TICKERS.index("TLT")],
)
check("and has the smaller bad-day loss", es_vine < es_gauss)

# The gain is real but modest -- a few percent of the shortfall. Most of the
# benefit of a vine is in steps 5-7: knowing the risk number is too low, and by
# how much, before the bad day rather than after it.
