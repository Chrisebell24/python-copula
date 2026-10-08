"""How to model a basket of 800 stocks with copulas, step by step.

Copulas that work beautifully for five assets break at 800. This walk-through
shows why, and builds the three architectures that do scale:

    0. Why the obvious copulas fail at 800 stocks
    1. Filter each stock with GARCH
    2. Fit three scalable copulas -- a factor copula, a sector-nested copula
       and a truncated vine -- and grade them against the truth
    3. Simulate 50,000 days for all 800 stocks
    4. Read off VaR and expected shortfall, and watch diversification vanish

The market is simulated, so the script runs offline and every model can be
checked against the dependence that really generated the data. To use real
prices, replace step 0's simulation with the yfinance lines there.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from _common import check, heading, show
from scipy import special

import rcopula as rc
from rcopula.core.elliptical import P2p
from rcopula.dynamic import fit_dynamic
from rcopula.garch import fit_garch
from rcopula.risk import expected_shortfall, value_at_risk
from rcopula.structural import NestedArchimedean

START = time.time()
N_STOCKS, N_SECTORS = 800, 10
N_TRAIN, N_TEST = 1000, 500  # four years to learn from, two to grade on
SECTOR = np.repeat(np.arange(N_SECTORS), N_STOCKS // N_SECTORS)
SAME_SECTOR = SECTOR[:, None] == SECTOR[None, :]
POSITION = 125_000.0  # dollars in each stock: a $100m equal-weight book
TRUE_DF = 4.0

# ---------------------------------------------------------------------------
heading("Step 0. Why the obvious copulas fail at 800 stocks")
# ---------------------------------------------------------------------------
# With real data:
#
#     import yfinance as yf
#     prices = yf.download(tickers, start="2018-01-01")["Close"]  # 800 tickers
#     returns = np.log(prices).diff().dropna()
#
# Here the hidden truth is a Student-t *factor* market: every stock loads on
# one market factor and on its own sector's factor, and a shared "panic"
# variable makes them all crash together now and then (the Student-t part).
# rc.FactorCopula is exactly that model.

rng = np.random.default_rng(32)
true_market = rng.uniform(0.45, 0.65, N_STOCKS)
true_sector = rng.uniform(0.30, 0.50, N_STOCKS)
truth = rc.FactorCopula(true_market, true_sector, groups=SECTOR, df=TRUE_DF)
true_corr = truth.sigma()  # market x market, plus sector x sector within a sector


# Each stock gets its own GARCH volatility and fat-tailed shocks.
vol = rng.uniform(0.010, 0.030, N_STOCKS)
g_alpha = rng.uniform(0.04, 0.10, N_STOCKS)
g_beta = rng.uniform(0.84, 0.89, N_STOCKS)  # alpha + beta < 1: stationary
g_omega = vol**2 * (1.0 - g_alpha - g_beta)
true_u = truth.rvs(N_TRAIN + N_TEST, random_state=rng)
shocks = special.stdtrit(5.0, true_u)
shocks /= np.sqrt(5.0 / 3.0)
returns = np.empty_like(shocks)
s2 = vol**2
for t in range(N_TRAIN + N_TEST):
    returns[t] = np.sqrt(s2) * shocks[t]
    s2 = g_omega + g_alpha * returns[t] ** 2 + g_beta * s2
TRUE_NEXT_VOL = np.sqrt(s2)
returns = pd.DataFrame(returns, columns=[f"S{j:03d}" for j in range(N_STOCKS)])
show("days x stocks", returns.shape)

# How many numbers does each obvious model need?
n_pairs = N_STOCKS * (N_STOCKS - 1) // 2
show("parameters of a full correlation matrix", f"{n_pairs:,}")
show("parameters of one Clayton/Gumbel copula", 1)
check("a full matrix has 319,600 entries to estimate", n_pairs == 319_600)

# The one-parameter copula: one tail number for every pair. Compare the
# dependence it is forced to assume with what the data show.
z_all = special.ndtri(rc.pseudo_obs(returns.to_numpy()))


# All 319,600 pairwise Kendall's taus at once: rc.cor_kendall computes the whole
# matrix in one blocked pass (seconds), identical to calling scipy pair by pair.
t0 = time.time()
tau_all = rc.cor_kendall(z_all)
show("seconds for all 319,600 Kendall's taus", time.time() - t0)
OFF_DIAGONAL = ~np.eye(N_STOCKS, dtype=bool)


def mean_tau(tau, mask):
    """Average Kendall's tau over the pairs selected by ``mask``."""
    return float(tau[mask & OFF_DIAGONAL].mean())


tau_within = mean_tau(tau_all, SAME_SECTOR)
tau_across = mean_tau(tau_all, ~SAME_SECTOR)
show("Kendall's tau, two stocks in the same sector", tau_within)
show("Kendall's tau, two stocks in different sectors", tau_across)
one_clayton = rc.ClaytonCopula.from_tau(0.1 * tau_within + 0.9 * tau_across)
show("one Clayton copula forces every pair to tau", one_clayton.tau())
check(
    "same-sector pairs are far more dependent than one number allows",
    tau_within > 1.4 * one_clayton.tau(),
)

# The full matrix: estimate it from four years of data and look at the
# smallest eigenvalue. Too little data for too many numbers makes the
# estimate nearly singular -- it "finds" portfolios that look riskless.
z_train, z_test = z_all[:N_TRAIN], z_all[N_TRAIN:]
sample_corr = np.corrcoef(z_train, rowvar=False)
show("smallest eigenvalue, true matrix", float(np.linalg.eigvalsh(true_corr)[0]))
show("smallest eigenvalue, estimated from 1,000 days", float(np.linalg.eigvalsh(sample_corr)[0]))
check(
    "the estimate is far closer to singular than the truth",
    np.linalg.eigvalsh(sample_corr)[0] < 0.2 * np.linalg.eigvalsh(true_corr)[0],
)

# ---------------------------------------------------------------------------
heading("Step 1. Filter each stock with GARCH")
# ---------------------------------------------------------------------------
# Strip out each stock's own volatility clustering so the copula only sees
# how stocks move *together*. rcopula fits GARCH(1,1); it has no GJR
# (asymmetric) variant, so a stock whose volatility jumps more after falls
# than after rises keeps a little of that in its residuals.

t0 = time.time()
margins = [fit_garch(returns[c], dist="t", name=c) for c in returns.columns]
show("seconds to fit 800 GARCH models", time.time() - t0)
resid = np.column_stack([m.resid for m in margins])
check("every fitted model is stationary", all(m.persistence < 1.0 for m in margins))

# Ranks turn each stock's residuals into uniforms on (0, 1). The margins were
# fitted on all six years, a small look-ahead that only touches the margins;
# the copulas below are fitted on the first four years and graded on the last
# two.
u = rc.pseudo_obs(resid)
u_train, u_test = u[:N_TRAIN], u[N_TRAIN:]
z_train, z_test = special.ndtri(u_train), special.ndtri(u_test)

# ---------------------------------------------------------------------------
heading("Step 2A. A factor copula: a few factors drive everything")
# ---------------------------------------------------------------------------
# Assume each stock = market loading x market factor + sector loading x
# sector factor + its own noise. Given the factors, stocks are independent,
# so 800 stocks need 800 market loadings + 800 sector loadings + 1 tail
# number, not 319,600 correlations. rc.fit_factor estimates the loadings by
# matching the correlations implied by Kendall's tau, sin(pi * tau / 2) --
# each loading is fitted to 799 correlations, which averages away most of
# their noise -- and then profiles the Student-t likelihood over the degrees
# of freedom.

t0 = time.time()
factor_t = rc.fit_factor(u_train, groups=SECTOR, family="student")
show("seconds to fit the 800-stock factor copula", time.time() - t0)
fit_market, fit_sector, fit_df = factor_t.market, factor_t.group_loadings, factor_t.df
show("average loading error, market factor", float(np.mean(np.abs(fit_market - true_market))))
show("average loading error, sector factor", float(np.mean(np.abs(fit_sector - true_sector))))
check(
    "the loadings are recovered to within a few hundredths",
    np.mean(np.abs(fit_market - true_market)) < 0.05
    and np.mean(np.abs(fit_sector - true_sector)) < 0.05,
)
show("Student-t degrees of freedom (true: 4)", fit_df)
show("parameters: factor copula", f"{factor_t.n_params:,}")
check("the data ask for fat joint tails", fit_df <= 6)
check("1,601 parameters", factor_t.n_params == 2 * N_STOCKS + 1)

# The Gaussian factor copula with the same loadings: Kendall's tau does not
# depend on the family, so the loadings would come out identical.
factor_g = rc.FactorCopula(fit_market, fit_sector, groups=SECTOR, family="gaussian")

# Grade on the two years the models never saw: log-likelihood per day. The
# factor copulas' densities never invert an 800 x 800 matrix (Woodbury does it
# with an 11 x 11 one); the full sample matrix is the dense GaussianCopula.
full_matrix = rc.GaussianCopula(P2p(np.corrcoef(z_train, rowvar=False)), dim=N_STOCKS, dispstr="un")
oos = {
    "full sample matrix (Gaussian)": float(np.sum(full_matrix.logpdf(u_test))),
    "factor, Gaussian": factor_g.loglik(u_test),
    "factor, Student-t": factor_t.loglik(u_test),
}
for name, value in oos.items():
    show(f"held-out log-likelihood per day: {name}", value / N_TEST)
check(
    "1,601 parameters beat 319,600 on unseen days",
    oos["factor, Gaussian"] > oos["full sample matrix (Gaussian)"],
)
check("and fat tails beat thin ones", oos["factor, Student-t"] > oos["factor, Gaussian"])
tau_train = rc.cor_kendall(u_train)
tau_gap = float(np.mean(np.abs(factor_t.tau_matrix() - tau_train)[OFF_DIAGONAL]))
show("average gap, model vs sample Kendall's tau", tau_gap)
check("the factor copula's 319,600 pairwise taus track the sample's", tau_gap < 0.03)

# Dynamic loadings. Real loadings drift -- a stock's beta rises in a crisis --
# and rcopula's GAS recursion lets a parameter move day by day. It is
# bivariate, so try it on one stock against the market proxy.
index_u = rc.pseudo_obs(z_train.mean(axis=1, keepdims=True))[:, 0]
pair = np.column_stack([index_u, u_train[:, 0]])
gas = fit_dynamic(pair, rc.GaussianCopula(0.5), driver="gas")
show("GAS gain over a constant loading (log-lik)", gas.loglik - gas.constant_loglik)
show("range of the filtered correlation", f"{gas.path.min():.2f} to {gas.path.max():.2f}")
check(
    "GAS contains the constant model, so it can only add", gas.loglik >= gas.constant_loglik - 1e-6
)
# Read that gain with care. The true loadings here are *constant*, yet on
# other simulated histories this same fit has found gains of up to ~20:
# a Gaussian recursion mistakes the Student-t's shared panic days for a
# correlation that moves. The likelihood-ratio test also sits on a boundary
# (see rcopula.dynamic), so a gain of a few units is not evidence of drift.

# ---------------------------------------------------------------------------
heading("Step 2B. A nested copula: one copula per sector, one linking them")
# ---------------------------------------------------------------------------
# A Clayton copula inside each sector catches its crashes; a weaker Clayton
# at the top links the sectors. 11 parameters in all, each the Clayton whose
# tau matches the average sample tau of the pairs it governs -- the estimator
# rcopula's fit_nested uses -- read off the full tau matrix.

theta_root = rc.ClaytonCopula.from_tau(mean_tau(tau_train, ~SAME_SECTOR)).theta
children = []
for s in range(N_SECTORS):
    block = np.where(SECTOR == s)[0]
    in_block = np.zeros_like(SAME_SECTOR)
    in_block[np.ix_(block, block)] = True
    theta = rc.ClaytonCopula.from_tau(mean_tau(tau_train, in_block)).theta
    children.append(NestedArchimedean(rc.ClaytonCopula(max(theta, theta_root)), list(block)))
nested = NestedArchimedean(rc.ClaytonCopula(theta_root), children=children)
show(
    "root theta / average sector theta",
    f"{theta_root:.2f} / {np.mean([c.theta for c in children]):.2f}",
)
check("sectors are bound tighter than the market", all(c.theta > theta_root for c in children))

# ---------------------------------------------------------------------------
heading("Step 2C. A truncated vine with the index at its root")
# ---------------------------------------------------------------------------
# A full 800-stock vine has 319,600 pair-copulas. Truncate it after the first
# tree, and put an equal-weight index at the root of a C-vine: tree 1 then
# links every stock to the index, and "truncated" means the stocks are
# independent once the index is known -- a one-factor copula, but each link
# free to choose its own family. It cannot see sectors; that is the price of
# stopping after one tree.

t0 = time.time()
with_index = np.column_stack([index_u, u_train])
vine = rc.fit_vine(
    with_index,
    structure="C",
    order=list(range(N_STOCKS + 1)),
    families=("gaussian", "student"),
    truncate=1,
)
show("seconds to fit 800 pair-copulas", time.time() - t0)
families = pd.Series([c.name for c in vine.pair_copulas[0]]).value_counts()
show("tree-1 families chosen", {k: int(v) for k, v in families.items()})
check("most links pick the fat-tailed family", families.get("Student", 0) > N_STOCKS / 2)

# ---------------------------------------------------------------------------
heading("Grading all of them: how often do 10% of the stocks move together?")
# ---------------------------------------------------------------------------
# Count days on which 80 or more of the 800 stocks have their own worst-1%
# day (a mass crash), and days on which 80 or more have their best-1% day (a
# mass rally). The truth is known, so each model can be graded directly.


def mass_moves(v):
    v = np.asarray(v)
    crash = float(np.mean((v < 0.01).sum(axis=1) >= 80))
    rally = float(np.mean((v > 0.99).sum(axis=1) >= 80))
    return crash, rally


n_sim = 20_000
# vine.rvs knows the vine is truncated after tree 1 (vine.truncation_level is
# 1), so it unwinds only that tree: 800 pair-copulas, not 319,600.
t0 = time.time()
vine_draws = vine.rvs(n_sim, random_state=2)[:, 1:]  # drop the index column
show("seconds to draw 20,000 days from the 801-variable vine", time.time() - t0)

graded = pd.DataFrame(
    {
        "truth": mass_moves(truth.rvs(n_sim, random_state=rng)),
        "factor, Student-t": mass_moves(factor_t.rvs(n_sim, random_state=rng)),
        "factor, Gaussian": mass_moves(factor_g.rvs(n_sim, random_state=rng)),
        "nested Clayton": mass_moves(nested.rvs(n_sim, random_state=1)),
        "vine (index root)": mass_moves(vine_draws),
    },
    index=["mass crash", "mass rally"],
).T
print((100 * graded).round(2).astype(str).add("%").to_string())
miss = (graded - graded.loc["truth"]).abs().drop("truth")
check(
    "the Student-t factor copula gets both within 20% of the truth",
    bool(np.all(np.abs(graded.loc["factor, Student-t"] / graded.loc["truth"] - 1) < 0.2)),
)
check(
    "nested Clayton nails crashes but misses rallies -- it has no upper tail",
    abs(graded.loc["nested Clayton", "mass crash"] / graded.loc["truth", "mass crash"] - 1) < 0.2
    and graded.loc["nested Clayton", "mass rally"] < 0.25 * graded.loc["truth", "mass rally"],
)
check(
    "a Gaussian copula under-counts both",
    bool(np.all(graded.loc["factor, Gaussian"] < 0.6 * graded.loc["truth"])),
)
check(
    "the Student-t factor copula has the smallest total miss",
    miss.sum(axis=1).idxmin() == "factor, Student-t",
)
# The one-tree vine sits in between: its Student-t links share crashes with
# the index, but with no sector tree it cannot see that a whole sector can
# fall at once. A second tree would help, at 799 more pair-copulas.

# ---------------------------------------------------------------------------
heading("Step 3. Simulate 50,000 days for all 800 stocks")
# ---------------------------------------------------------------------------
# 50,000 x 800 is 40 million numbers (320 MB), so simulate in chunks and keep
# only what is needed: the book's P&L and each stock's own P&L in the tail.
# Each uniform becomes a return through that stock's GARCH model, starting
# from tomorrow's forecast volatility.

next_vol = np.array([m.forecast_vol(1)[0] for m in margins])
mu = np.array([m.mu for m in margins])
sorted_resid = np.sort(resid, axis=0)
grid_pos = np.arange(len(sorted_resid))


def to_returns(v):
    """Uniforms -> returns: invert each stock's residual distribution, then scale."""
    position = v * (len(grid_pos) - 1)
    innov = np.column_stack(
        [np.interp(position[:, j], grid_pos, sorted_resid[:, j]) for j in range(N_STOCKS)]
    )
    return mu + next_vol * innov


def simulate_book(copula, n=50_000, chunk=10_000, truth=False):
    pnl, stock_pnl = [], []
    for _ in range(n // chunk):
        v = copula.rvs(chunk, random_state=rng)  # O(n x 800): no 800 x 800 matrix
        if truth:  # the true margins: t(5) shocks at the true next-day volatility
            r = TRUE_NEXT_VOL * special.stdtrit(5.0, v) / np.sqrt(5.0 / 3.0)
        else:
            r = to_returns(v)
        pnl.append(r @ np.full(N_STOCKS, POSITION))
        stock_pnl.append((r * POSITION).astype(np.float32))
    return np.concatenate(pnl), np.concatenate(stock_pnl)


t0 = time.time()
book = {
    "truth": simulate_book(truth, truth=True),
    "factor, Student-t": simulate_book(factor_t),
    "factor, Gaussian": simulate_book(factor_g),
}
show("seconds to simulate 3 x 50,000 days x 800 stocks", time.time() - t0)

# ---------------------------------------------------------------------------
heading("Step 4. VaR and expected shortfall for the $100m book")
# ---------------------------------------------------------------------------
# VaR 99%: the loss exceeded on 1 day in 100. Expected shortfall 99%: the
# average loss on those days.

rows = {}
for name, (pnl, stock_pnl) in book.items():
    var = value_at_risk(-pnl, 0.99)
    es = expected_shortfall(-pnl, 0.99)
    standalone = sum(expected_shortfall(-stock_pnl[:, j], 0.99) for j in range(N_STOCKS))
    rows[name] = {
        "VaR 99% $m": var / 1e6,
        "ES 99% $m": es / 1e6,
        "diversification": 1 - es / standalone,
    }
risk = pd.DataFrame(rows).T
print(risk.round(3).to_string())
check(
    "the Student-t factor copula's ES is within 15% of the truth",
    abs(risk.loc["factor, Student-t", "ES 99% $m"] / risk.loc["truth", "ES 99% $m"] - 1) < 0.15,
)
check(
    "the Gaussian copula understates the bad-day loss",
    risk.loc["factor, Gaussian", "ES 99% $m"] < risk.loc["factor, Student-t", "ES 99% $m"],
)
check(
    "and promises diversification that is not there",
    risk.loc["factor, Gaussian", "diversification"]
    > risk.loc["factor, Student-t", "diversification"],
)

# Diversification evaporates in a panic. On the book's worst 1% of days, count
# how many of the 800 stocks are having their own worst-5% day. If they failed
# independently it would be about 40; it is far more, and the Gaussian copula
# says far fewer than the Student-t.


def stocks_in_trouble(name):
    pnl, stock_pnl = book[name]
    worst = pnl < np.quantile(pnl, 0.01)
    own_bad = stock_pnl < np.quantile(stock_pnl, 0.05, axis=0)
    return float(own_bad[worst].sum(axis=1).mean())


trouble = {name: stocks_in_trouble(name) for name in book}
for name, count in trouble.items():
    show(f"stocks having their own bad day, on the book's worst days: {name}", f"{count:.0f}")
check("far more than the 40 that independent failures would give", trouble["truth"] > 5 * 40)
check(
    "the Student-t factor copula gets the count right, the Gaussian does not",
    abs(trouble["factor, Student-t"] / trouble["truth"] - 1)
    < abs(trouble["factor, Gaussian"] / trouble["truth"] - 1),
)

show("total runtime, seconds", time.time() - START)
