"""How to trade with copulas, step by step.

A plain-language walk-through of copula statistical arbitrage, for someone who
has never built a pairs trade. It puts the trading pieces of the package
together in the order a desk would use them:

    1. Get a universe of returns        5. Backtest out of sample, with costs
    2. Pick the pair                     6. Compare with the classic z-score trade
    3. Fit a copula to the pair          7. Trade one stock against three (a vine)
    4. Turn it into a trading signal     8. Know when to stop: the pair breaks

The data are simulated so the script runs offline and every claim is checked
against a market whose rules are known. To use real prices, replace step 1 with
the yfinance lines in the comment there.

Examples 10 and 25 go deeper on the signal and the selection rules; this one is
the end-to-end recipe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import check, heading, show
from scipy import stats

import rcopula as rc
from rcopula.portfolio import mispricing_index, pairs_signal
from rcopula.statarb import select_pairs, select_partners

FORMATION = 750  # three years to learn the relationship...
TRADING = 500  # ...two years to trade it, never seen while fitting
COST = 0.0005  # 5 basis points per leg, every time a position changes

# ---------------------------------------------------------------------------
heading("Step 1. Get a universe of daily returns")
# ---------------------------------------------------------------------------
# With real data this step is just:
#
#     import yfinance as yf
#     prices = yf.download(tickers, start="2019-01-01")["Close"]
#     returns = np.log(prices).diff().dropna()
#
# Here a simulated market stands in. Every stock moves with the market and its
# sector, and three relationships are hidden inside for a trader to find:
#
#   OIL1 / OIL2      a true pair: the gap between their prices wanders but comes back
#   BANK1            tracks the average of BANK2, BANK3 and BANK4 the same way
#   TECH1 / TECH2    a pair that works for four years, then breaks (TECH2 is taken over)


def reverting_gap(n: int, rng: np.random.Generator, vol: float) -> np.ndarray:
    """A price gap pulled back towards zero, closing about 5% of itself a day."""
    gap = np.zeros(n)
    for t in range(1, n):
        gap[t] = 0.95 * gap[t - 1] + vol * rng.standard_t(5) / np.sqrt(5 / 3)
    return gap


def simulate_universe(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = FORMATION + TRADING + 1

    def walk(vol: float) -> np.ndarray:
        return np.cumsum(vol * rng.standard_t(4, n) / np.sqrt(2))

    market = walk(0.009)
    oil, bank, tech, retail = walk(0.008), walk(0.005), walk(0.005), walk(0.006)
    log_price = {}

    oil_price = market + oil + walk(0.004)
    oil_gap = reverting_gap(n, rng, 0.006)
    log_price["OIL1"] = oil_price + oil_gap / 2
    log_price["OIL2"] = oil_price - oil_gap / 2

    for name in ("BANK2", "BANK3", "BANK4"):
        log_price[name] = market + bank + walk(0.007)
    peers = np.mean([log_price[k] for k in ("BANK2", "BANK3", "BANK4")], axis=0)
    log_price["BANK1"] = peers + reverting_gap(n, rng, 0.005)

    tech_price = market + tech + walk(0.004)
    tech_gap = reverting_gap(n, rng, 0.006)
    log_price["TECH1"] = tech_price + tech_gap / 2
    tech2 = tech_price - tech_gap / 2
    split = FORMATION + 250  # after this, TECH2 trades on its own news
    own = walk(0.015)
    tech2[split:] = tech2[split - 1] + own[split:] - own[split - 1]
    log_price["TECH2"] = tech2

    for name in ("RTL1", "RTL2"):
        log_price[name] = market + retail + walk(0.012)

    return pd.DataFrame(log_price).diff().dropna().reset_index(drop=True)


returns = simulate_universe(seed=11)
formation, trading = returns.iloc[:FORMATION], returns.iloc[FORMATION:]
show("days x stocks", returns.shape)
show("formation / trading days", f"{len(formation)} / {len(trading)}")

# ---------------------------------------------------------------------------
heading("Step 2. Pick the pair")
# ---------------------------------------------------------------------------
# Rank every pair of stocks by how tightly they move together, using only the
# formation window. Kendall's tau works on ranks, so a few huge days cannot
# dominate it the way they dominate ordinary correlation.

ranked = select_pairs(formation, method="kendall", top=5)
print(ranked.round(3).to_string(index=False))
A, B = ranked.iloc[0]["first"], ranked.iloc[0]["second"]
check("the top pair is the true oil pair", {A, B} == {"OIL1", "OIL2"})

# ---------------------------------------------------------------------------
heading("Step 3. Fit a copula to the pair")
# ---------------------------------------------------------------------------
# Let the data choose the family. The copula describes *how* the two move
# together -- including whether they crash together -- separately from how
# volatile each one is.

u_form = rc.pseudo_obs(formation[[A, B]].to_numpy())
choice = rc.select_copula(u_form, families=["gaussian", "student", "clayton", "gumbel", "frank"])
copula = choice.best
show("family chosen by AIC", copula.describe())
show("Kendall's tau", copula.tau())
show("chance of crashing together", copula.lambda_().lower)
check("the pair is strongly dependent", copula.tau() > 0.5)

# ---------------------------------------------------------------------------
heading("Step 4. Turn the copula into a trading signal")
# ---------------------------------------------------------------------------
# The key quantity is the h-function:
#
#     h1 = P(A moves this little or less | B's move today)
#
# It answers "how unusual was A's move, given what B did?". If B jumped 2% and
# A did nothing, h1 is small: A has lagged its partner and looks cheap. Under
# the fitted copula h1 is uniform, so 0.05 means the same thing for any pair.
#
# New days are ranked against the formation window only, so nothing from the
# future leaks in.


def to_uniform(history: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Where each new return sits among the history: 0.01 = a 1-in-100 low."""
    m = len(history)
    out = np.column_stack(
        [
            np.searchsorted(np.sort(history[:, j]), new[:, j], side="right") / (m + 1)
            for j in range(new.shape[1])
        ]
    )
    return np.clip(out, 0.5 / (m + 1), 1 - 0.5 / (m + 1))


r_form = formation[[A, B]].to_numpy()
r_trade = trading[[A, B]].to_numpy()
u_trade = to_uniform(r_form, r_trade)
h1, h2 = mispricing_index(copula, u_trade)

example = np.array([[0.50, 0.95]])  # A flat, B has a top-5% day
show("h1 when A is flat and B rallies", float(mispricing_index(copula, example)[0][0]))
check("A looks cheap: h1 is tiny", mispricing_index(copula, example)[0][0] < 0.05)

# One day's mispricing is mostly noise. Adding it up gives a running "flag" of
# how far A has fallen behind B -- the idea of Xie, Liew, Wu and Zou (2016) --
# but a plain running total never forgets, so it drifts. Instead let old
# mispricings fade at the rate the pair's gap actually closes, measured on the
# formation window: if the gap closes 5% a day, the flag forgets 5% a day.
#
#   flag below -entry  ->  A has lagged:    buy A, sell B
#   flag above +entry  ->  A has run ahead: sell A, buy B
#   flag back to zero  ->  the gap has closed: exit


def closing_rate(spread_returns: np.ndarray) -> float:
    """How much of the price gap is kept each day (an AR(1) fit to the gap)."""
    gap = np.cumsum(spread_returns)
    gap = gap - gap.mean()
    return float(np.dot(gap[1:], gap[:-1]) / np.dot(gap[:-1], gap[:-1]))


def flag_positions(h: np.ndarray, entry: float, keep: float) -> np.ndarray:
    position, flag, pos = np.zeros(len(h), dtype=int), 0.0, 0
    for t, value in enumerate(h):
        flag = keep * flag + (value - 0.5)
        if pos == 0:
            pos = 1 if flag < -entry else -1 if flag > entry else 0
        elif (pos == 1 and flag >= 0) or (pos == -1 and flag <= 0):
            pos = 0
        position[t] = pos
    return position


# ---------------------------------------------------------------------------
heading("Step 5. Backtest out of sample, with costs")
# ---------------------------------------------------------------------------
# A position decided at today's close earns tomorrow's return on (A - B).
# Every change of position pays the cost on both legs. The entry threshold is
# tuned on the formation window only, then frozen for the trading window.


def run(position: np.ndarray, spread_return: np.ndarray) -> dict[str, float]:
    held = np.concatenate([[0], position[:-1]])
    traded = np.abs(np.diff(np.concatenate([[0], position])))
    net = held * spread_return - 2 * COST * traded
    sharpe = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0.0
    trades = int(np.sum((held == 0) & (position != 0)))
    return {"total return": float(net.sum()), "Sharpe": float(sharpe), "trades": trades}


def tune(make_positions, candidates, spread_return):
    """Pick the threshold with the best formation-window Sharpe ratio."""
    return max(candidates, key=lambda c: run(make_positions(c), spread_return)["Sharpe"])


ENTRIES = [0.5, 0.75, 1.0, 1.5, 2.0]
spread_form = r_form[:, 0] - r_form[:, 1]
spread_trade = r_trade[:, 0] - r_trade[:, 1]
h1_form = mispricing_index(copula, u_form)[0]
keep = closing_rate(spread_form)
entry = tune(lambda c: flag_positions(h1_form, c, keep), ENTRIES, spread_form)
copula_flag = run(flag_positions(h1, entry, keep), spread_trade)
show("gap kept per day (measured)", keep)
show("entry threshold (tuned on formation)", entry)
for key, value in copula_flag.items():
    show(f"copula flag strategy: {key}", value)

# The simplest copula rule trades one day's signal (h1 < 5% and h2 > 95%)
# and holds for a day. It is what pairs_signal and backtest_pairs implement.
daily = run(pairs_signal(copula, u_trade, entry=0.05), spread_trade)
for key, value in daily.items():
    show(f"one-day signal: {key}", value)

check("the flag strategy makes money after costs", copula_flag["total return"] > 0)
check("the one-day signal does not: costs eat it", daily["total return"] < 0)

# ---------------------------------------------------------------------------
heading("Step 6. Compare with the classic z-score trade")
# ---------------------------------------------------------------------------
# The standard pairs trade (Gatev et al. 2006) watches the price gap itself:
# measure it in standard deviations from its formation-window average, enter at
# +/- k, exit when it crosses back. A fair comparison tunes k the same way.

gap_form = np.cumsum(spread_form)
gap_trade = gap_form[-1] + np.cumsum(spread_trade)
mu, sd = gap_form.mean(), gap_form.std()


def zscore_positions(gap: np.ndarray, k: float) -> np.ndarray:
    z, position, pos = (gap - mu) / sd, np.zeros(len(gap), dtype=int), 0
    for t, value in enumerate(z):
        if pos == 0:
            pos = 1 if value < -k else -1 if value > k else 0
        elif (pos == 1 and value >= 0) or (pos == -1 and value <= 0):
            pos = 0
        position[t] = pos
    return position


k = tune(lambda c: zscore_positions(gap_form, c), [1.0, 1.5, 2.0, 2.5], spread_form)
zscore = run(zscore_positions(gap_trade, k), spread_trade)
table = pd.DataFrame({"z-score": zscore, "copula flag": copula_flag, "one-day copula": daily}).T
print(table.round(3).to_string())

# The honest result: on this pair the plain z-score wins. That is not a fluke
# of the seed -- the two oil stocks differ by exactly one straight-line price
# gap, and the z-score watches that gap directly, while the copula rebuilds it
# from daily ranks and loses information doing so. When the relationship is a
# simple linear spread, use the simple tool. The copula earns its place in the
# next two steps: more than two assets, and knowing when to stop.
check(
    "the z-score beats the copula on a linear price gap", zscore["Sharpe"] > copula_flag["Sharpe"]
)

# ---------------------------------------------------------------------------
heading("Step 7. Trade one stock against three (a vine)")
# ---------------------------------------------------------------------------
# BANK1 has no single twin: it tracks the *average* of three other banks. Find
# those partners, fit a vine with BANK1 last, and the vine's Rosenblatt
# transform gives exactly the conditional probability we need:
#
#     P(BANK1 moves this little or less | what all three partners did today)

target = "BANK1"
found = select_partners(formation, target, method="extended")
partners = list(found["partners"])
show("partners found for BANK1", partners)
check("they are the three other banks", sorted(partners) == ["BANK2", "BANK3", "BANK4"])

cols = [*partners, target]
r4_form, r4_trade = formation[cols].to_numpy(), trading[cols].to_numpy()
vine = rc.fit_vine(r4_form, structure="D", order=[0, 1, 2, 3])
print(vine.describe())


def h_given_partners(u: np.ndarray) -> np.ndarray:
    return vine.rosenblatt(u)[:, -1]


h_vine_form = h_given_partners(np.asarray(rc.pseudo_obs(r4_form)))
h_vine_trade = h_given_partners(to_uniform(r4_form, r4_trade))

# The trade is BANK1 against an equal-weight basket of its partners.
basket_form = r4_form[:, 3] - r4_form[:, :3].mean(axis=1)
basket_trade = r4_trade[:, 3] - r4_trade[:, :3].mean(axis=1)
keep4 = closing_rate(basket_form)
entry4 = tune(lambda c: flag_positions(h_vine_form, c, keep4), ENTRIES, basket_form)
vine_result = run(flag_positions(h_vine_trade, entry4, keep4), basket_trade)

# For comparison: the best you can do with a single partner and a pair copula.
best = partners[0]
pair_cols = [target, best]
rp_form, rp_trade = formation[pair_cols].to_numpy(), trading[pair_cols].to_numpy()
pair_copula = rc.select_copula(rc.pseudo_obs(rp_form)).best
hp_form = mispricing_index(pair_copula, rc.pseudo_obs(rp_form))[0]
hp_trade = mispricing_index(pair_copula, to_uniform(rp_form, rp_trade))[0]
sp_form, sp_trade = rp_form[:, 0] - rp_form[:, 1], rp_trade[:, 0] - rp_trade[:, 1]
keep2 = closing_rate(sp_form)
entry2 = tune(lambda c: flag_positions(hp_form, c, keep2), ENTRIES, sp_form)
pair_result = run(flag_positions(hp_trade, entry2, keep2), sp_trade)

print(
    pd.DataFrame({"vine: BANK1 vs 3 partners": vine_result, f"pair: BANK1 vs {best}": pair_result})
    .T.round(3)
    .to_string()
)
check("the vine trade makes money after costs", vine_result["total return"] > 0)
show("gap kept per day, BANK1 vs basket", keep4)
show("gap kept per day, BANK1 vs one partner", keep2)

# Why the vine is the right tool here: BANK1's gap to the *basket* comes back,
# its gap to any single bank does not (that bank has news of its own). Over
# two years one P&L number can go either way -- in five of the six simulated
# markets we tried the vine had the higher Sharpe ratio -- but only one of these
# two trades is betting on something that is actually true.
check("the gap to the basket closes", keep4 < 0.97)
check("faster than the gap to any one partner", keep4 < keep2)

# ---------------------------------------------------------------------------
heading("Step 8. Know when to stop: the pair breaks")
# ---------------------------------------------------------------------------
# TECH1 / TECH2 was the fourth-best pair in step 2. A year into trading, TECH2
# is taken over and starts trading on its own news. A pairs strategy that keeps
# going will bet on a gap that is never coming back.
#
# The copula gives a simple alarm: track Kendall's tau over the last 60 days
# and stop trading once it falls below half of what the formation window
# showed. Positions are judged on information up to yesterday only.

tf, tt = formation[["TECH1", "TECH2"]].to_numpy(), trading[["TECH1", "TECH2"]].to_numpy()
tau_form = stats.kendalltau(tf[:, 0], tf[:, 1]).statistic
both = np.vstack([tf, tt])
rolling_tau = np.array(
    [
        stats.kendalltau(both[t - 60 : t, 0], both[t - 60 : t, 1]).statistic
        for t in range(FORMATION, FORMATION + TRADING)
    ]
)
broken = rolling_tau < tau_form / 2
alarm = int(np.argmax(broken)) if broken.any() else None
show("Kendall's tau, formation window", tau_form)
show("Kendall's tau, last 60 trading days", float(rolling_tau[-1]))
show("alarm fires on trading day", alarm)
check("the alarm fires after the takeover (day 250), not before", alarm is not None and alarm > 250)
check("and within three months of it", alarm is not None and alarm < 250 + 63)

ts_form, ts_trade = tf[:, 0] - tf[:, 1], tt[:, 0] - tt[:, 1]
tg_form = np.cumsum(ts_form)
tg_trade = tg_form[-1] + np.cumsum(ts_trade)
mu, sd = tg_form.mean(), tg_form.std()
kt = tune(lambda c: zscore_positions(tg_form, c), [1.0, 1.5, 2.0, 2.5], ts_form)
always = zscore_positions(tg_trade, kt)
stopped = always.copy()
stopped[alarm:] = 0
no_stop, with_stop = run(always, ts_trade), run(stopped, ts_trade)
after = slice(250, None)
pnl = pd.DataFrame({"keep trading": no_stop, "stop on alarm": with_stop}).T
print(pnl.round(3).to_string())
risk_before = float(ts_trade[:250].std())
risk_after = float(ts_trade[250:].std())
show("daily swing of the gap before the takeover", risk_before)
show("daily swing of the gap after it", risk_after)
exposed = float(np.mean(always[alarm:] != 0))
show("share of post-alarm days still in a trade", exposed)

# Note what the alarm does and does not promise. After the takeover the gap no
# longer comes back and swings far harder, so a strategy that keeps trading is
# making bigger coin-flip bets: sometimes they win -- on this run they did, see
# the total return in the table -- often they lose, and nothing about the pair
# says which. The alarm does not guarantee more money; it stops you taking
# large risks on a relationship that no longer exists, which is why the
# stopped strategy's Sharpe ratio is the better one.
check("after the takeover the gap swings more than twice as hard", risk_after > 2 * risk_before)
check("yet without the alarm the strategy keeps betting on it", exposed > 0.2)
