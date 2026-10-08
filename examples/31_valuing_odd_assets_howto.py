"""How to value odd assets whose price depends on things moving together.

A plain-language walk-through. Some assets cannot be priced one underlying at
a time: a note that pays off the *worst* of three stocks, a put on a whole
basket, insurance on the *first* default in a group of bonds, a catastrophe
bond hit when two regions flood at once, a private stake with no market price.
For all of them the answer depends on how the pieces move *together* -- and in
particular on whether they fail together, which is exactly what a correlation
number does not say.

    1. One idea: same correlation, different crashes
    2. A worst-of note on three stocks       5. A first-to-default credit basket
    3. A put on a basket                     6. A catastrophe bond on two regions
    4. How accurate is the price?            7. Marking a private stake to a crash

Every step prices the same asset under copulas that agree on Kendall's tau --
the usual rank correlation -- and differ only in their tails. Where the price
moves, the gap is what a correlation-only model gets wrong. Market inputs are
assumed, so the script runs offline. Rates are zero to keep the arithmetic
visible.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc
from rcopula.credit import nth_to_default_probability
from rcopula.derivatives import lognormal_terminal
from rcopula.insurance import catastrophe_bond
from rcopula.sampling import quasi_rvs, variance_ratio

TAU = 0.5  # every model below shares this rank correlation
N = 400_000

# ---------------------------------------------------------------------------
heading("Step 1. One idea: same correlation, different crashes")
# ---------------------------------------------------------------------------
# Four ways for three stocks to move together, all calibrated to tau = 0.5.
# They differ only in the tails: the chance two stocks have an extreme day
# *together*. Gaussian says never; Student-t says both ways; Clayton says
# crashes only; Gumbel says rallies only.


def models(dim: int) -> dict[str, rc.Copula]:
    return {
        "Gaussian": rc.GaussianCopula.from_tau(TAU, dim=dim),
        "Student-t": rc.StudentCopula.from_tau(TAU, dim=dim, df=4.0),
        "Clayton": rc.ClaytonCopula.from_tau(TAU, dim=dim),
        "Gumbel": rc.GumbelCopula.from_tau(TAU, dim=dim),
    }


for name, cop in models(2).items():
    tail = cop.lambda_()
    show(
        f"{name:<10} tau, crash together, rally together",
        f"{cop.tau():.2f}, {tail.lower:.2f}, {tail.upper:.2f}",
    )
check("all four agree on tau", all(abs(c.tau() - TAU) < 1e-9 for c in models(2).values()))


def mc(payoff: np.ndarray) -> tuple[float, float]:
    """Price and standard error: the average payoff, and how unsure we are of it."""
    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(payoff.size))


def gap_in_errors(a: tuple[float, float], b: tuple[float, float]) -> float:
    """How many standard errors apart two prices are -- above ~4 is a real difference."""
    return abs(a[0] - b[0]) / np.hypot(a[1], b[1])


# ---------------------------------------------------------------------------
heading("Step 2. A worst-of note on three stocks")
# ---------------------------------------------------------------------------
# A popular retail product. You lend 100 for a year and get an 8% coupon. At
# the end you get your 100 back -- unless the WORST of three stocks has fallen
# below 70% of its start, in which case you get 100 times the worst stock's
# performance. The 8% is payment for selling a put on the worst performer.
#
# Recipe: simulate each stock's price in a year (lognormal, its own vol), link
# them with a copula, apply the payoff, average.

vols = [0.25, 0.30, 0.35]
stocks = [lognormal_terminal(1.0, v, 1.0) for v in vols]  # price relative to today


def worst_of_note(u: np.ndarray) -> np.ndarray:
    perf = np.column_stack([m.ppf(u[:, j]) for j, m in enumerate(stocks)])
    worst = perf.min(axis=1)
    return 8.0 + np.where(worst < 0.70, 100.0 * worst, 100.0)


note = {name: mc(worst_of_note(cop.rvs(N, random_state=1))) for name, cop in models(3).items()}
for name, (p, se) in note.items():
    show(f"{name:<10} value of the note", f"{p:.2f} +/- {se:.2f}")
indep = mc(worst_of_note(rc.IndependenceCopula(3).rvs(N, random_state=1)))
show("independent stocks (for scale)", f"{indep[0]:.2f} +/- {indep[1]:.2f}")
check(
    "the more the stocks move together, the safer the note",
    all(p > indep[0] for p, _ in note.values()),
)
spread = max(p for p, _ in note.values()) - min(p for p, _ in note.values())
show("range across copulas at the same tau", f"{spread:.2f}")
check(
    "the copula choice moves the price by more than the simulation noise",
    gap_in_errors(max(note.values()), min(note.values())) > 4,
)
# Two things are going on. Moving together at all protects the holder: a note
# on independent stocks is worth 92, on any linked trio 96-98. And the tail
# shape still moves the price by about 2.5 points at the same tau. The holder
# loses if ANY stock breaks below 70%; when crashes cluster (Clayton) the bad
# outcomes pile onto the same few paths, so there are fewer paths where at
# least one stock breaks. The holder is long lower-tail dependence -- the same
# logic as the first-to-default basket in step 5.

# ---------------------------------------------------------------------------
heading("Step 3. A put on a basket: insurance against everything falling")
# ---------------------------------------------------------------------------
# A one-year put that pays if an equal-weight basket of the three stocks ends
# below 80%. This is a pure bet on a joint crash: one stock falling is diluted
# by the other two, so it only pays when they fall *together*.


def basket_put(u: np.ndarray, strike: float = 0.80) -> np.ndarray:
    basket = np.column_stack([m.ppf(u[:, j]) for j, m in enumerate(stocks)]).mean(axis=1)
    return 100.0 * np.maximum(strike - basket, 0.0)


put = {name: mc(basket_put(cop.rvs(N, random_state=2))) for name, cop in models(3).items()}
for name, (p, se) in put.items():
    show(f"{name:<10} basket put (per 100 notional)", f"{p:.3f} +/- {se:.3f}")
show("Clayton / Gaussian", f"{put['Clayton'][0] / put['Gaussian'][0]:.2f}x")
check(
    "crash-together (Clayton) prices the protection highest",
    put["Clayton"][0] == max(p for p, _ in put.values()),
)
check(
    "rally-together (Gumbel) prices it lowest", put["Gumbel"][0] == min(p for p, _ in put.values())
)
check("and the gap is far outside the noise", gap_in_errors(put["Clayton"], put["Gumbel"]) > 10)
# The buyer is long lower-tail dependence. Price this protection with a
# Gaussian model and the seller is charging for a crash it thinks is rarer
# than it is.

# ---------------------------------------------------------------------------
heading("Step 4. How accurate is the price? Use fewer, smarter draws")
# ---------------------------------------------------------------------------
# Every price above carries a standard error: the price wobbles by about that
# much if you rerun with a different seed. Halve it and you need 4x the draws.
# Quasi-random (Sobol) points fill the space more evenly than random ones, so
# they reach the same accuracy with far fewer draws. variance_ratio measures
# the gain honestly, by repeating both and comparing the spread.

gauss3 = models(3)["Gaussian"]
gain = variance_ratio(gauss3, basket_put, 8192, method="sobol", replicates=20, random_state=0)
show("standard error, plain Monte Carlo", f"{gain['plain_se']:.4f}")
show("standard error, Sobol", f"{gain['reduced_se']:.4f}")
show("plain draws needed to match Sobol", f"{gain['equivalent_sample_factor']:.1f}x")
check("Sobol is more accurate for the same number of draws", gain["ratio"] > 1.5)
sobol_price = mc(basket_put(quasi_rvs(gauss3, 2**16, random_state=0)))[0]
check(
    "and it agrees with the big random run",
    abs(sobol_price - put["Gaussian"][0]) < 4 * put["Gaussian"][1] + 4 * gain["reduced_se"],
)

# ---------------------------------------------------------------------------
heading("Step 5. A first-to-default basket on ten bonds")
# ---------------------------------------------------------------------------
# Credit insurance on a group of ten companies, each with a 2% chance of
# defaulting this year. A first-to-default swap pays out on the FIRST default;
# a fifth-to-default pays only if five go. Fair premium = chance of payout x
# loss given default (60%).

credit = {
    "Gaussian": rc.GaussianCopula.from_tau(0.3, dim=10),
    "Clayton": rc.ClaytonCopula.from_tau(0.3, dim=10),
}
premium = {}
for name, cop in credit.items():
    first = nth_to_default_probability(cop, 0.02, 1, N, random_state=3)
    fifth = nth_to_default_probability(cop, 0.02, 5, N, random_state=3)
    premium[name] = (first * 0.6, fifth * 0.6)
    show(f"{name:<9} 1st-to-default premium", f"{first * 0.6:.2%} a year")
    show(f"{name:<9} 5th-to-default premium", f"{fifth * 0.6:.3%} a year")
check(
    "crash-together makes the FIRST default cheaper to insure",
    premium["Clayton"][0] < premium["Gaussian"][0],
)
check("and the FIFTH far dearer", premium["Clayton"][1] > 2 * premium["Gaussian"][1])
# When defaults come in clusters, there are fewer separate chances for a first
# one, but when one goes, many go. First-to-default sellers are short
# independence; senior (nth) sellers are short tail dependence. This is the
# trade that went wrong in 2008 -- example 08 has the CDO version.

# ---------------------------------------------------------------------------
heading("Step 6. A catastrophe bond exposed to two regions")
# ---------------------------------------------------------------------------
# An insurer issues bonds that lose principal when its combined hurricane
# losses in Florida and Texas pass an attachment point, and are wiped out at an
# exhaustion point. Losses are heavy-tailed (lognormal), nothing like a normal
# distribution. Two bonds: a "near" one (3bn-6bn) and a "remote" one
# (15bn-30bn). The question is whether one season hits both regions hard.

regions = [stats.lognorm(1.2, scale=0.6), stats.lognorm(1.4, scale=0.4)]  # losses in $bn
layers = {"near (3-6bn)": (3.0, 6.0), "remote (15-30bn)": (15.0, 30.0)}
bond = {}
for name, cop in {
    "Gaussian": rc.GaussianCopula.from_tau(0.4),
    "Gumbel": rc.GumbelCopula.from_tau(0.4),  # big-with-big: upper tail
}.items():
    losses = rc.CopulaDistribution(cop, regions).rvs(N, random_state=4).sum(axis=1)
    for layer, (attach, exhaust) in layers.items():
        bond[name, layer] = catastrophe_bond(losses, attach, exhaust, coupon=0.06, risk_free=0.03)
        show(f"{name:<9} {layer:<17} expected loss", f"{bond[name, layer]['expected_loss']:.2%}")
near, remote = "near (3-6bn)", "remote (15-30bn)"
ratio_near = bond["Gumbel", near]["expected_loss"] / bond["Gaussian", near]["expected_loss"]
ratio_remote = bond["Gumbel", remote]["expected_loss"] / bond["Gaussian", remote]["expected_loss"]
show("Gumbel / Gaussian, near bond", f"{ratio_near:.2f}x")
show("Gumbel / Gaussian, remote bond", f"{ratio_remote:.2f}x")
check("joint extremes barely matter for the near bond", abs(ratio_near - 1) < 0.06)
check("but make the remote bond over 10% riskier", ratio_remote > 1.10)
check(
    "so at the same coupon the remote bond pays a worse multiple",
    bond["Gumbel", remote]["multiple"] < bond["Gaussian", remote]["multiple"],
)
# The near bond is hit by any bad season in either region, so it hardly cares
# whether they coincide. The remote bond needs both regions to be hit hard at
# once -- the upper tail -- so Gumbel, not Clayton, is its stress case. Remote
# cat bond investors are short upper-tail dependence between regions, and the
# effect is real but modest: heavy-tailed losses are often dominated by one
# region's catastrophe alone.

# ---------------------------------------------------------------------------
heading("Step 7. Marking a private stake when the market crashes")
# ---------------------------------------------------------------------------
# You own a stake in a private company: no market price, only a quarterly
# appraisal. The stock market just had a 1-in-20 bad quarter. What is the stake
# likely worth now? Link the private return to the public index with a copula,
# then draw the private return *given* the index's move -- the inverse
# Rosenblatt transform does exactly that conditioning.

private = stats.t(4, loc=0.02, scale=0.12)  # quarterly return: fatter tails, more risk
index_move = 0.05  # the index's quantile this quarter: its worst 5%
z = np.column_stack([np.full(100_000, index_move), np.random.default_rng(5).uniform(size=100_000)])
marks = {}
for name, cop in {
    "Gaussian": rc.GaussianCopula.from_tau(0.5),
    "Clayton": rc.ClaytonCopula.from_tau(0.5),
    "Gumbel": rc.GumbelCopula.from_tau(0.5),
}.items():
    r = private.ppf(rc.inverse_rosenblatt(cop, z)[:, 1])
    marks[name] = (float(np.median(r)), float(np.quantile(r, 0.95) - np.quantile(r, 0.05)))
    show(f"{name:<9} median mark / 5-95% range", f"{marks[name][0]:+.1%} / {marks[name][1]:.1%}")
check(
    "crash-together marks the stake down hardest",
    marks["Clayton"][0] < marks["Gaussian"][0] < marks["Gumbel"][0],
)
check("and is more certain about it (a narrower range)", marks["Clayton"][1] < marks["Gaussian"][1])
# Under Clayton a bad market quarter drags the private stake down with it
# almost every time: a deeper median mark with less room for luck. Gaussian
# says the stake might well have escaped. Appraisals lag; this is the
# model-implied mark they will catch up to. A lender against the stake is short
# lower-tail dependence: the collateral falls hardest when everything does.
