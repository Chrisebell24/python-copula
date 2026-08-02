"""Choosing what to trade: the step before the strategy.

Out of 500 names there are 124,750 pairs and 2.6 billion quadruples. Which ones
you look at is not a preamble to a copula strategy -- it largely *is* the
strategy, because the criteria disagree and each one hands you a different book.

This runs all six pair rules and all four partner rules on the same panel, shows
where they part company, and ends with a caveat about the "extremal" rule that
is easier to state than to discover.

References
----------
Gatev, E., Goetzmann, W. N. and Rouwenhorst, K. G. (2006). Pairs trading:
    performance of a relative-value arbitrage rule.
    *Review of Financial Studies* 19(3), 797-827.
Stübinger, J., Mangold, B. and Krauss, C. (2018). Statistical arbitrage with
    vine copulas. *Quantitative Finance* 18(11), 1831-1849.
Schmid, F. and Schmidt, R. (2007). Multivariate extensions of Spearman's rho
    and related statistics. *Statistics and Probability Letters* 77(4), 407-416.
Krauss, C. (2017). Statistical arbitrage pairs trading strategies: review and
    outlook. *J. Economic Surveys* 31(2), 513-545.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from _common import check, heading, show

import rcopula as rc
from rcopula.statarb import (
    PAIR_METHODS,
    PARTNER_METHODS,
    diagonal_distance,
    multivariate_spearman,
    select_pairs,
    select_partners,
    tail_concentration,
)

heading("The rules disagree, which is why there is more than one")

# Two planted pairs. A1/A2 drift together -- their cumulative paths agree --
# while their daily moves are noisy. B1/B2 move together day to day but their
# paths separate. Both are real relationships; they are not the same one.
rng = np.random.default_rng(3)
n = 1500
common = rng.standard_normal(n)
a1 = common + 1.2 * rng.standard_normal(n)
a2 = common + 1.2 * rng.standard_normal(n)
drift = rng.standard_normal(n)
b1 = drift + 0.25 * rng.standard_normal(n) + 0.04
b2 = drift + 0.25 * rng.standard_normal(n) - 0.04
panel = pd.DataFrame({"A1": a1, "A2": a2, "B1": b1, "B2": b2})

print(f"  {'rule':>12}   {'top pair':<10}{'score':>12}")
picks = {}
for method in PAIR_METHODS:
    best = select_pairs(panel, method=method).iloc[0]
    picks[method] = f"{best['first']}/{best['second']}"
    print(f"  {method:>12}   {picks[method]:<10}{best['score']:>12.4f}")

check(
    "the level-based rules pick the pair whose paths track",
    picks["distance"] == picks["qq"] == "A1/A2",
)
check(
    "the rank-based rules pick the pair whose daily moves track",
    picks["kendall"] == picks["spearman"] == picks["pearson"] == "B1/B2",
)
print(
    "\n  Neither answer is wrong. `distance` asks 'have these drifted apart?',\n"
    "  which is what a mean-reversion trade needs. The rank rules ask 'do these\n"
    "  move together?', which is what a copula needs -- a copula never sees the\n"
    "  levels at all. Picking a rule is picking a question."
)

heading("A realistic screen")

# Sixty names on three sectors plus idiosyncratic noise.
rng = np.random.default_rng(0)
days, per_sector = 1200, 20
sectors = {
    "TECH": rng.standard_normal(days),
    "BANK": rng.standard_normal(days),
    "ENRG": rng.standard_normal(days),
}
market = rng.standard_normal(days)
columns = {}
for sector, factor in sectors.items():
    for k in range(per_sector):
        loading = 0.5 + 0.5 * rng.uniform()
        columns[f"{sector}{k:02d}"] = (
            0.3 * market + loading * factor + 0.8 * rng.standard_normal(days)
        )
universe = pd.DataFrame(columns)
show("universe", f"{universe.shape[1]} names, {universe.shape[0]} days")
show("pairs to score", universe.shape[1] * (universe.shape[1] - 1) // 2)

ranked = select_pairs(universe, method="kendall", top=8)
print(f"\n  {'rank':>5}{'pair':>18}{'tau':>10}   same sector?")
for _, row in ranked.iterrows():
    same = row["first"][:4] == row["second"][:4]
    print(
        f"  {row['rank']:>5}{row['first'] + '/' + row['second']:>18}{row['score']:>10.4f}"
        f"   {'yes' if same else 'no'}"
    )

same_sector = sum(r["first"][:4] == r["second"][:4] for _, r in ranked.iterrows())
show("\ntop 8 pairs drawn from one sector", f"{same_sector} of 8")
check("the screen recovers the sector structure without being told it", same_sector >= 6)

heading("Partners: a quadruple, not a pair")

# A vine needs a group. All four approaches, on one target.
target = "TECH00"
print(f"  {'approach':>13}{'partners':>34}{'score':>12}{'searched':>10}")
found = {}
for method in PARTNER_METHODS:
    result = select_partners(universe, target, method=method, n_candidates=20)
    found[method] = result
    print(
        f"  {method:>13}{', '.join(result['partners']):>34}"
        f"{result['score']:>12.4f}{result['considered']:>10}"
    )

all_tech = {m: all(p.startswith("TECH") for p in r["partners"]) for m, r in found.items()}
show("\napproaches whose partners are all same-sector", sum(all_tech.values()))
check("most approaches recover the sector", sum(all_tech.values()) >= 3)
exhaustive = 59 * 58 * 57 // 6
show("combinations searched per approach", found["extended"]["considered"])
show("combinations without pre-screening", exhaustive)
show("   reduction", f"{exhaustive / found['extended']['considered']:.0f}x")
check("pre-screening does the heavy lifting", found["extended"]["considered"] < 0.1 * exhaustive)
print(
    "\n  At 60 names that is a 29-fold saving and merely convenient. At the 500 of\n"
    "  an index it is 20 million combinations per target against 1140 -- the\n"
    "  difference between a screen that runs and one that does not.\n\n"
    "  Three of the four approaches agree exactly here. `extremal` picks a\n"
    "  different trio, which is the whole reason to have it -- and the last\n"
    "  section is about how much to read into that."
)

heading("What the multivariate measures actually say")

block = universe[[target, *found["extended"]["partners"]]]
noise = universe[[target, "BANK00", "ENRG00", "BANK01"]]
print(f"  {'group':>26}{'mv Spearman':>14}{'diag distance':>15}{'tail conc.':>12}")
for label, frame in [("the chosen quadruple", block), ("a mixed-sector quadruple", noise)]:
    u = rc.pseudo_obs(frame)
    print(
        f"  {label:>26}{multivariate_spearman(u):>14.4f}"
        f"{diagonal_distance(u):>15.4f}{tail_concentration(u):>12.4f}"
    )

chosen_u, mixed_u = rc.pseudo_obs(block), rc.pseudo_obs(noise)
check(
    "the chosen group is more dependent",
    multivariate_spearman(chosen_u) > multivariate_spearman(mixed_u),
)
check(
    "and sits closer to the hyper-diagonal",
    diagonal_distance(chosen_u) < diagonal_distance(mixed_u),
)

show(
    "\nmultivariate Spearman under independence",
    multivariate_spearman(rc.IndependenceCopula(4).rvs(20_000, random_state=0)),
)
show(
    "...and under comonotonicity",
    multivariate_spearman(np.tile(rng.uniform(size=(20_000, 1)), (1, 4))),
)
check(
    "it is pinned at 0 and 1 like the bivariate coefficient it generalises",
    abs(multivariate_spearman(rc.IndependenceCopula(4).rvs(20_000, random_state=0))) < 0.03,
)

heading("The caveat on the extremal rule")

# A four-dimensional corner deep enough to isolate the tail is empty at any
# realistic sample size, so the corner has to widen -- and a wide corner
# measures co-movement in general, not tails.
clayton = rc.ClaytonCopula.from_tau(0.5).rvs(200_000, random_state=0)
gaussian = rc.GaussianCopula.from_tau(0.5).rvs(200_000, random_state=0)
print(f"  {'corner':>9}{'Clayton':>11}{'Gaussian':>11}{'ratio':>9}   (d = 2, same tau)")
for q in (0.30, 0.10, 0.05, 0.02, 0.01):
    c, g = tail_concentration(clayton, q), tail_concentration(gaussian, q)
    print(f"  {q:>9}{c:>11.2f}{g:>11.2f}{c / g:>9.2f}")

check(
    "a wide corner cannot tell them apart",
    tail_concentration(clayton, 0.30) / tail_concentration(gaussian, 0.30) < 1.05,
)
check(
    "a deep corner can",
    tail_concentration(clayton, 0.01) / tail_concentration(gaussian, 0.01) > 1.3,
)

print(f"\n  {'sample size':>14}{'deepest corner at d = 4':>26}")
for size in (1_200, 50_000, 1_000_000, 250_000_000):
    print(f"  {size:>14,}{min(0.25, (40 / size) ** 0.25):>26.3f}")
print(
    "\n  Separation needs a corner near 0.01, and at d = 4 that needs a quarter of\n"
    "  a billion observations to keep forty points inside it. With 1200 days the\n"
    "  deepest usable corner is 0.25 -- squarely in the range where Clayton and\n"
    "  Gaussian are indistinguishable.\n\n"
    "  So `extremal` in four dimensions ranks by joint co-movement, not by tail\n"
    "  dependence. That is a fact about the data, not about the implementation,\n"
    "  and it is worth knowing before choosing that rule expecting otherwise."
)

heading("And the output feeds straight into the strategy")

# The point of the module: what it returns is what the next function takes.
best = select_pairs(universe, method="kendall").iloc[0]
u = rc.pseudo_obs(universe[[best["first"], best["second"]]])
family = rc.select_copula(u, criterion="aic")
show("chosen pair", f"{best['first']} / {best['second']}")
show("family selected for it", family.best.describe())
signal = rc.portfolio.pairs_signal(family.best, np.asarray(u))
check("a tradable signal comes out the other end", np.asarray(signal).shape[0] == len(universe))

vine = rc.fit_vine(rc.pseudo_obs(block), structure="D")
show("and the quadruple fits as a vine", f"{vine.structure}-vine, dim {vine.dim}")
show("   tree-1 families", [type(c).__name__.replace("Copula", "") for c in vine.pair_copulas[0]])
check(
    "with a finite likelihood", np.all(np.isfinite(vine.logpdf(np.asarray(rc.pseudo_obs(block)))))
)
