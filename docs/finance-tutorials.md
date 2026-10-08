# Finance tutorials

Five plain-language walk-throughs, each a sequence of short steps from data to an
answer a desk would act on. Every one has a runnable script in
[`examples/`](https://github.com/Chrisebell24/python-copula/tree/main/examples)
that simulates its own market, so it works offline and **asserts every number
quoted here**. To use real data, swap step 1 for your own returns.

| Tutorial | You will learn to |
|---|---|
| [Vine copulas for markets](#vine-copulas-for-markets) | Fit a vine to returns, read it, forecast VaR, stress test, build a crash-aware portfolio |
| [Risk management](#risk-management-for-a-trading-book) | Backtest VaR, allocate risk to desks, measure diversification and contagion, reverse stress test |
| [Trading strategies](#trading-strategies-with-copulas) | Pick pairs, turn a copula into a signal, backtest with costs, trade a basket with a vine, spot a broken pair |
| [Valuing odd assets](#valuing-odd-assets) | Price worst-of notes, basket puts, first-to-default, cat bonds and private stakes under different tails |
| [Scaling to 800 stocks](#scaling-to-800-stocks) | Beat the curse of dimensionality with factor, sector-nested and truncated-vine copulas, then simulate 50,000 days for VaR and ES |

The thread through all five: a correlation matrix is a **Gaussian copula**, and a
Gaussian copula says extreme days happen independently. Each tutorial measures
what that assumption costs.

## Vine copulas for markets

A correlation matrix says how assets move together *on average*. It cannot say
that small caps crash with the S&P far more often than they rally with it, or
that Treasuries rally in a sell-off. A **vine copula** can: it links assets in
pairs, and every pair gets its own shape. Eight steps, from a table of returns to
a crash-aware portfolio. The full runnable version, with the data simulated so it
works offline, is
[`examples/28_vine_markets_howto.py`](https://github.com/Chrisebell24/python-copula/blob/main/examples/28_vine_markets_howto.py).

**1. Get daily returns.** Any `DataFrame` of returns, one column per asset.

```python
import numpy as np
import yfinance as yf  # pip install yfinance -- or use your own data

tickers = ["SPY", "QQQ", "IWM", "HYG", "TLT"]  # large caps, tech, small caps, credit, bonds
prices = yf.download(tickers, start="2015-01-01")["Close"][tickers]
returns = np.log(prices).diff().dropna()
```

**2. Strip out volatility.** Calm and stormy periods hit every asset at once, and
a copula fitted to raw returns mistakes that for dependence. Fit a GARCH to each
asset and keep the *standardised residuals*: how surprising each day was, given
how volatile things already were. Then convert them to ranks in (0, 1), where
0.01 means "a 1-in-100 bad day for this asset".

```python
import pandas as pd
import rcopula as rc
from rcopula.garch import fit_garch

margins = [fit_garch(returns[t], dist="t", name=t) for t in tickers]
u = rc.pseudo_obs(pd.DataFrame({m.name: m.resid for m in margins}))
```

**3. Fit the vine.** One line. A C-vine puts the most connected asset at the
centre and links every other asset to it; higher trees capture what is left once
that asset is accounted for. Each link picks its own family by AIC.

```python
vine = rc.fit_vine(u, structure="C")
print(vine.describe())
# C-vine copula, dim 5, order [0, 1, 2, 3, 4]       <- SPY (column 0) is the centre
#   tree 1  0,1      Student copula, rho=0.877, df=4.06
#   tree 1  0,2      Clayton copula, theta=2.59
#   tree 1  0,3      Clayton copula, theta=1.32
#   tree 1  0,4      Frank copula, theta=-1.94
#   tree 2  1,2|0    Gaussian copula, rho=0.208      <- QQQ-IWM once SPY is known
#   ...
```

**4. Read what it found.** Tree 1 is the part to read. Kendall's τ is the overall
co-movement (−1 to 1); tail dependence is the chance that two assets have an
extreme day *together* (0 to 1).

```python
for i, pair in enumerate(vine.pair_copulas[0]):
    asset = tickers[vine.order[i + 1]]
    tail = pair.lambda_()
    print(
        f"SPY-{asset}  {pair.name:<8} tau={pair.tau():+.2f}  "
        f"crash together={tail.lower:.2f}  rally together={tail.upper:.2f}"
    )
# SPY-QQQ  Student  tau=+0.68  crash together=0.59  rally together=0.59
# SPY-IWM  Clayton  tau=+0.56  crash together=0.77  rally together=0.00
# SPY-HYG  Clayton  tau=+0.40  crash together=0.59  rally together=0.00
# SPY-TLT  Frank    tau=-0.21  crash together=0.00  rally together=0.00
```

Small caps and credit fall with the S&P but do not rally with it; bonds move the
other way. A Gaussian copula, which is what a correlation matrix implies, sets
every one of those tail numbers to zero.

**5. Simulate tomorrow and measure risk.** Put the GARCH margins and the vine
back together. The forecast starts from *today's* volatility, so the risk number
moves with the market.

```python
from rcopula.garch import CopulaGarch

weights = [0.30, 0.15, 0.15, 0.10, 0.30]
model = CopulaGarch(margins, vine, innovations="parametric")
model.forecast_risk(weights, alpha=0.99, n=200_000)
# {'var': 0.0203, 'expected_shortfall': 0.0271, ...}   Gaussian copula: 0.0186 and 0.0240
```

**6. Ask how often they crash together.** Simulate from the vine and count the
days when SPY, QQQ and IWM all have a 1-in-50 bad day at once.

```python
sims = vine.rvs(400_000)
np.mean(np.all(sims[:, :3] < 0.02, axis=1))
# vine 1.13%   data 1.04%   Gaussian copula 0.48%   independent 0.0008%
```

**7. Stress test.** Keep only the simulated days when SPY has a 1-in-100 loss,
and look at everything else on those days.

```python
crash_days = sims[sims[:, 0] < 0.01]
(crash_days < 0.05).mean(axis=0)  # chance each asset has its own worst-5% day
# QQQ 93%   IWM 99%   HYG 92%   TLT 2%
```

**8. Build a crash-aware portfolio.** Feed simulated days to an optimiser that
minimises expected shortfall (the average loss on the worst 1% of days), capped at
40% per fund. Because the scenarios come from the vine, the optimiser knows which
assets fail together.

```python
from rcopula.portfolio import mean_cvar_weights

scenarios = model.forecast(horizon=1, n=5_000)
w = mean_cvar_weights(scenarios, alpha=0.99, bounds=(0.0, 0.40))
```

Optimised on Gaussian-copula scenarios instead, the portfolio holds less of the
TLT hedge, and its real bad-day loss comes out a little higher. That gain is
modest. Most of what a vine adds is in steps 5–7: knowing *before* the bad day
that the risk number is too low, and by how much.

**Which vine?** Use a **C-vine** when one asset or factor drives the rest (an
index, a sector ETF, oil for energy names). Use a **D-vine** (`structure="D"`) for a
chain, such as points on a yield curve or futures expiries. For 20+ assets, add
`truncate=2` to model only the first two trees and treat the rest as independent.
The [vine tutorial](vines.md) has the
theory, and `rc.statarb.select_partners` picks which assets to group together.

## Risk management for a trading book

Five desks, one book, and the questions a risk manager gets every morning. The
dependence model is where most risk systems quietly go wrong: a correlation
matrix (a Gaussian copula) says desks have their worst days independently. Eight
steps; the runnable version, with simulated data so it works offline, is
[`examples/29_risk_management_howto.py`](https://github.com/Chrisebell24/python-copula/blob/main/examples/29_risk_management_howto.py).

**1. Get each desk's daily returns.** One column per desk, and the dollar exposure in each.

```python
import numpy as np
import pandas as pd
import rcopula as rc

desks = ["US equity", "EU equity", "Credit", "Rates", "Commodities"]
exposure = np.array([40.0, 30.0, 25.0, 40.0, 15.0])  # $ millions
returns = ...  # DataFrame of daily desk returns, columns = desks
```

**2. Fit volatility per desk, then dependence between desks.** GARCH answers "how
volatile is each desk right now"; the copula answers "how do desks move together
once that is accounted for". Fit a Gaussian copula and a Student-t, which lets
desks crash together. Its degrees of freedom say how often: the lower, the more.

```python
from rcopula.garch import fit_garch

margins = [fit_garch(returns[d], dist="t", name=d) for d in desks]
u = rc.pseudo_obs(pd.DataFrame({m.name: m.resid for m in margins}))

gauss = rc.fit(rc.GaussianCopula(dim=5, dispstr="un"), u, method="itau").copula
student = rc.fit(rc.StudentCopula(dim=5, dispstr="un"), u, method="itau.mpl").copula
# df = 3.05; log-likelihood 616 points better than the Gaussian
```

**3. Today's VaR and expected shortfall, in dollars.** VaR 99% is the loss you
should exceed on 1 day in 100; expected shortfall (ES) is the average loss on
those days. The forecast starts from today's volatility.

```python
from rcopula.garch import CopulaGarch

model = CopulaGarch(margins, student, innovations="parametric")
model.forecast_risk(exposure, alpha=0.99, n=200_000)
# Student-t:  VaR $1.84m   ES $2.43m
# Gaussian:   VaR $1.76m   ES $2.23m    <- understates the bad-day loss by ~8%
```

**4. Backtest: was each day's VaR right?** Walk forward through days the model
never saw. Each morning compute VaR, each evening check whether the loss broke
through it. A 99% VaR should break on about 1% of days, and Kupiec's test asks
whether the count is too far off to be bad luck.

```python
breaks = int(np.sum(-pnl > var_each_day))  # see the example for the walk-forward loop
# 4,000 days, expect 40:   Gaussian 51 (p = 0.09)   Student-t 45 (p = 0.44)
```

The Gaussian breaks more often, but only by a handful of days, and on other
simulated histories either model can fail the test. Daily VaR is driven mostly by
volatility, which GARCH handles for both. The copula matters most deeper in the
tail, in the steps below.

**5. Which desk is the risk coming from?** Euler allocation gives each desk's
share of ES: its average loss on the book's worst days. The shares add up to the
total, so they can be used to charge capital.

```python
from rcopula.risk import risk_contributions

risk_contributions(student, loss_margins, exposure, alpha=0.99)
#              exposure %   ES share %
# US equity        26.7        34.1
# EU equity        20.0        40.0
# Credit           16.7         9.3
# Rates            26.7        -0.5     <- a quarter of the capital, none of the risk
# Commodities      10.0        17.1
```

**6. How much does diversification actually buy?** Compare the book's ES with the
worst case, where every desk has its bad day at once.

```python
from rcopula.risk import diversification_benefit

diversification_benefit(student, loss_margins, 0.99, exposure)["benefit_pct"]
# Student-t 27.9%   Gaussian 33.4%   <- the correlation matrix promises benefit that isn't there
```

**7. If one desk is in trouble, what happens to the rest?** Delta-CoVaR measures
how much the rest of the book's VaR rises when one desk has a 1-in-20 bad day. It
measures contagion, not size.

```python
from rcopula.risk import delta_covar

delta_covar(rest_of_book, desk_losses)
#              Student-t   Gaussian      ($m)
# Credit          1.23       0.98
# US equity       1.04       0.87
# Rates           0.38      -0.30    <- the hedge that stops working in a crisis
```

Under a Gaussian copula, a bad day for rates is a good day for the equity-heavy
rest of the book. Under the Student-t it is not: a 1-in-20 move anywhere means a
turbulent day everywhere. A hedge that works on average does not necessarily work
in a crisis.

**8. Reverse stress test: what does a disaster look like?** Rather than guessing a
scenario, ask the model what typically happens on days worse than the
1-in-1,000-day loss (about once every four years).

```python
book = sims.sum(axis=1)
worst = sims[book > rc.risk.value_at_risk(book, 0.999)]
worst.mean(axis=0)
# threshold $3.22m, average loss $4.01m:  EU equity 1.55  US equity 1.33
#   Commodities 0.75  Credit 0.38  Rates 0.00
# How often is a loss that big?   Student-t: 1 day in 1,000   Gaussian: 1 in 1,794
```

## Trading strategies with copulas

A copula pairs trade asks one question every day: *given what stock B just did,
how unusual was stock A's move?* If B jumped and A didn't, A has fallen behind and
looks cheap. Eight steps, from a universe of stocks to knowing when to stop
trading. The runnable version uses a simulated market with hidden pairs, so it
runs offline and checks itself:
[`examples/30_trading_strategies_howto.py`](https://github.com/Chrisebell24/python-copula/blob/main/examples/30_trading_strategies_howto.py).

**1. Get a universe of returns, and split it.** Learn on the first three years and
trade the next two. The model never sees the trading window while it is being
fitted.

```python
import numpy as np
import yfinance as yf

prices = yf.download(tickers, start="2019-01-01")["Close"]
returns = np.log(prices).diff().dropna()
formation, trading = returns.iloc[:750], returns.iloc[750:]
```

**2. Pick the pair.** Rank every pair by Kendall's τ on the formation window. τ
works on ranks, so a few huge days can't dominate it.

```python
from rcopula.statarb import select_pairs

select_pairs(formation, method="kendall", top=3)
#  first second  score
#   OIL1   OIL2  0.696     <- the true pair, found
#  BANK3  BANK1  0.609
#  BANK2  BANK1  0.607
```

**3. Fit a copula to the pair.** Let AIC choose the family.

```python
import rcopula as rc

u = rc.pseudo_obs(formation[["OIL1", "OIL2"]])
copula = rc.select_copula(u, families=["gaussian", "student", "clayton", "gumbel", "frank"]).best
# Student copula, rho=0.888, df=2.76   tau=0.70, chance of crashing together=0.66
```

**4. Turn it into a signal.** The *h-function* `h1 = P(A this low | B's move)` is
the "how unusual?" number. It's uniform under the copula, so 0.05 means the same
thing for any pair.

```python
from rcopula.portfolio import mispricing_index

h1, h2 = mispricing_index(copula, [[0.50, 0.95]])  # A flat, B has a top-5% day
# h1 = 0.020  -> A has lagged its partner: buy A, sell B
```

One day of this is mostly noise. Add the daily `h1 - 0.5` into a running *flag*
that forgets at the rate the pair's gap closes. Enter when the flag passes a
threshold and exit when it returns to zero.

**5. Backtest out of sample, with costs.** Tune the threshold on the formation
window, freeze it, and charge 5 bp per leg on every trade.

```
                 total return  Sharpe  trades
copula flag            +8.0%    0.69      12
one-day copula         -7.0%   -1.37      19    <- trading single days: costs eat it
```

**6. Compare with the classic z-score trade.** Watch the price gap itself, and
enter at ±k standard deviations.

```
z-score               +21.4%    1.99       7
```

The z-score wins here, and that's the honest result. These two stocks differ by
exactly one straight-line price gap, and the z-score watches that gap directly.
When the relationship is a simple linear spread, use the simple tool. A copula
earns its place with more than two assets, and in knowing when to stop.

**7. Trade one stock against three (a vine).** BANK1 has no single twin: it tracks
the average of three other banks. Find them, fit a vine with BANK1 last, and the
vine's Rosenblatt transform gives `P(BANK1 this low | all three partners)`.

```python
from rcopula.statarb import select_partners

partners = select_partners(formation, "BANK1")["partners"]  # ['BANK2', 'BANK3', 'BANK4']
vine = rc.fit_vine(formation[[*partners, "BANK1"]], structure="D", order=[0, 1, 2, 3])
h = vine.rosenblatt(u_today)[:, -1]  # feed into the same flag rule
# vine, BANK1 vs basket: +8.6%, Sharpe 0.82  |  gap closes ~6%/day vs ~1.5%/day for one partner
```

BANK1's gap to the basket comes back. Its gap to any single bank doesn't, because
each bank has news of its own.

**8. Know when to stop.** Pairs break, for example after a takeover. Track
Kendall's τ over the last 60 days and stop once it falls below half the formation
value.

```python
from scipy import stats

tau_now = stats.kendalltau(last60["TECH1"], last60["TECH2"]).statistic
# formation tau 0.61 -> 0.06; the alarm fires on day 286, 36 days after the takeover
```

```
               total return  Sharpe
keep trading         +39.6%    0.93
stop on alarm        +29.3%    1.98
```

After the break, the gap swings three times harder and never comes back. Carrying
on is a bigger coin flip that happened to pay off this time. The alarm doesn't
promise more money; it stops you betting on a relationship that no longer exists.

## Valuing odd assets

Some assets can't be priced one underlying at a time. A note paying off the
*worst* of three stocks, a put on a whole basket, insurance on the *first* default
in a group of bonds, a cat bond hit when two regions flood at once, a private
stake with no market price. Their value depends on whether the pieces fail
*together*, which a correlation number does not say. Each step prices one asset
under copulas that agree on Kendall's τ and differ only in their tails. The
runnable version is
[`examples/31_valuing_odd_assets_howto.py`](https://github.com/Chrisebell24/python-copula/blob/main/examples/31_valuing_odd_assets_howto.py).

**1. Same correlation, different crashes.** Four copulas, all at τ = 0.5:

```python
import rcopula as rc

for cop in (
    rc.GaussianCopula.from_tau(0.5),
    rc.StudentCopula.from_tau(0.5, df=4.0),
    rc.ClaytonCopula.from_tau(0.5),
    rc.GumbelCopula.from_tau(0.5),
):
    print(cop.name, cop.lambda_())
# Gaussian   crash together 0.00   rally together 0.00
# Student-t  crash together 0.40   rally together 0.40
# Clayton    crash together 0.71   rally together 0.00
# Gumbel     crash together 0.00   rally together 0.59
```

**2. A worst-of note on three stocks.** Lend 100 for a year and earn an 8% coupon.
If the *worst* stock ends below 70% of its start, you get back 100 × its
performance instead of 100. To price it, simulate each stock with its own
volatility, link them with a copula, apply the payoff and average.

```python
import numpy as np
from rcopula.derivatives import lognormal_terminal

stocks = [lognormal_terminal(1.0, v, 1.0) for v in (0.25, 0.30, 0.35)]


def worst_of_note(u):
    worst = np.column_stack([m.ppf(u[:, j]) for j, m in enumerate(stocks)]).min(axis=1)
    return 8.0 + np.where(worst < 0.70, 100.0 * worst, 100.0)


worst_of_note(rc.ClaytonCopula.from_tau(0.5, dim=3).rvs(400_000)).mean()
# Gaussian 96.74   Student-t 97.06   Clayton 98.49   Gumbel 96.02   independent 92.30  (+/- 0.03)
```

The holder loses if *any* stock breaks below 70%. When crashes cluster, the bad
outcomes land on the same few paths, so fewer paths have a break. The holder
benefits when stocks crash together.

**3. A put on a basket.** It pays if an equal-weight basket of the three stocks
ends below 80%. One stock falling is diluted by the other two, so it only pays
when they fall *together*.

```python
# per 100 notional:  Gaussian 2.73   Student-t 2.67   Clayton 3.04   Gumbel 2.49   (+/- 0.01)
```

Clayton prices this protection 11% above Gaussian at the same correlation. The
buyer is the one exposed to joint crashes.

**4. How accurate is the price?** Every price above has a standard error, and
halving it costs 4x the draws. Quasi-random (Sobol) points cover the space more
evenly, so they reach the same accuracy with far fewer draws:

```python
from rcopula.sampling import quasi_rvs, variance_ratio

variance_ratio(cop, basket_put, 8192, method="sobol")
# standard error 0.0647 -> 0.0014; plain draws would need ~2000x as many to match
price = basket_put(quasi_rvs(cop, 2**16)).mean()
```

That gain is unusually large: this payoff is smooth and depends on only three
assets. Most payoffs gain far less.

**5. A first-to-default basket on ten bonds.** Each bond has a 2% chance of
default. A first-to-default swap pays on the first default; a fifth-to-default
pays only if five go.

```python
from rcopula.credit import nth_to_default_probability

0.6 * nth_to_default_probability(rc.ClaytonCopula.from_tau(0.3, dim=10), 0.02, n_th=1)
#                 1st-to-default   5th-to-default
# Gaussian            7.59%            0.22%      premium a year
# Clayton             3.82%            0.93%
```

Clustered defaults make the first one *cheaper* to insure and the fifth four times
dearer. This is the correlation trade that broke in 2008.

**6. A catastrophe bond exposed to two regions.** Florida and Texas hurricane
losses are heavy-tailed (lognormal). There are two bonds: a near one (3–6bn) and a
remote one (15–30bn).

```python
from rcopula.insurance import catastrophe_bond

losses = rc.CopulaDistribution(rc.GumbelCopula.from_tau(0.4), regions).rvs(400_000).sum(axis=1)
catastrophe_bond(losses, 15.0, 30.0, coupon=0.06)["expected_loss"]
#            near bond   remote bond
# Gaussian     12.79%       0.65%
# Gumbel       12.25%       0.75%     (0.96x and 1.16x)
```

Big losses arriving together barely matter for the near bond, which any bad season
in either region hits. The remote bond needs both regions hit hard at once, so its
stress case is Gumbel, not Clayton.

**7. Marking a private stake after a market crash.** The stake has no market
price, and the index just had its worst-5% quarter. Draw the private return
*given* that move with the inverse Rosenblatt transform:

```python
z = np.column_stack([np.full(100_000, 0.05), rng.uniform(size=100_000)])
r = private.ppf(rc.inverse_rosenblatt(rc.ClaytonCopula.from_tau(0.5), z)[:, 1])
#            median mark   5-95% range
# Gaussian     -14.3%         45.2%
# Clayton      -20.8%         27.7%
# Gumbel       -11.9%         47.5%
```

Under Clayton the stake falls further and more predictably. Gaussian says it may
well have escaped. A lender against the stake has collateral that falls hardest
when everything falls.

## Scaling to 800 stocks

Copulas that work beautifully for five assets break at 800. One Clayton or Gumbel
copula forces all 319,600 pairs to share a single tail number; an unstructured
800-stock Gaussian or Student-t copula needs 319,600 correlations, far more than
a few years of data can pin down. The fix is structure: assume a few factors, or
sectors, drive the co-movement. Four steps, from 800 return series to a VaR you
can trust. The runnable version simulates a market whose true dependence is
known, so every model can be graded against it:
[`examples/32_high_dimensional_basket_howto.py`](https://github.com/Chrisebell24/python-copula/blob/main/examples/32_high_dimensional_basket_howto.py).

**0. See why the obvious copulas fail.** A single Archimedean copula can't tell a
same-sector pair from a cross-sector one. A full correlation matrix estimated
from four years of data is nearly singular: it "finds" portfolios that look
almost riskless.

```python
import numpy as np

sample_corr = np.corrcoef(z_train, rowvar=False)  # z_train: 1,000 days x 800 stocks
np.linalg.eigvalsh(sample_corr)[0]
# smallest eigenvalue: true matrix 0.342, estimated 0.004
# Kendall's tau: same sector 0.31, different sectors 0.19; one Clayton forces 0.21 on all
```

**1. Filter each stock with GARCH.** Strip out each stock's own volatility
clustering, then turn its residuals into uniforms with `pseudo_obs`. 800 fits take
about 7 seconds. rcopula fits GARCH(1,1); it has no asymmetric (GJR) variant.

```python
import pandas as pd
import rcopula as rc
from rcopula.garch import fit_garch

margins = [fit_garch(returns[c], dist="t", name=c) for c in returns.columns]
u = rc.pseudo_obs(np.column_stack([m.resid for m in margins]))
```

**2. Fit a structured copula.** Three architectures, each fitted on four years and
graded on two more:

- **Factor copula.** Each stock is a market loading times a market factor, plus a
  sector loading times its sector factor, plus noise of its own; given the
  factors, stocks are independent. 800 + 800 loadings + 1 tail number = **1,601
  parameters** instead of 319,600. The example fits the loadings by matching
  sample correlations (rcopula has no factor-copula class) and profiles the
  Student-t degrees of freedom.

  ```
  held-out log-likelihood per day
  full 319,600-entry matrix   -1,763
  factor, Gaussian               233
  factor, Student-t              325     df = 4, as in the truth
  ```

- **Nested copula by sector.** A Clayton copula inside each sector, a weaker one
  linking the ten sectors: 11 parameters.

  ```python
  from rcopula.structural import NestedArchimedean

  sectors = [NestedArchimedean(rc.ClaytonCopula(theta[s]), members[s]) for s in range(10)]
  nested = NestedArchimedean(rc.ClaytonCopula(theta_root), children=sectors)
  # root theta 0.48, sectors 0.88 on average
  ```

- **Truncated vine.** An equal-weight index at the root of a C-vine, cut after the
  first tree: every stock links to the index with its own family, and the stocks
  are independent given the index. 800 pair-copulas, fitted in about 2 minutes.

  ```python
  vine = rc.fit_vine(
      np.column_stack([index_u, u]),
      structure="C",
      order=list(range(801)),
      families=("gaussian", "student"),
      truncate=1,
  )
  # all 800 links choose the Student-t
  ```

Grade them on days when 80 or more of the 800 stocks have their own worst-1% day
(a mass crash), or best-1% day (a mass rally):

```
                   mass crash   mass rally
truth                 2.97%        2.81%
factor, Student-t     2.64%        2.78%    <- closest overall
factor, Gaussian      1.24%        1.36%
nested Clayton        2.84%        0.00%    <- right on crashes, blind to rallies
vine (index root)     1.83%        1.96%    <- cannot see sectors after one tree
```

The factor copula's loadings can also move over time: `rcopula.dynamic.fit_dynamic`
with `driver="gas"` lets one stock's link to the market drift day by day. Read its
gain with care. On this market, whose loadings are constant, it found gains from
0.5 to 20 log-likelihood units depending on the simulated history, because a
Gaussian recursion mistakes fat-tailed panic days for moving correlation.

**3. Simulate 50,000 days for all 800 stocks.** Draw uniforms from the factor
copula, turn each into a return through its stock's residual distribution and
tomorrow's GARCH volatility, and sum the P&L. 50,000 × 800 is 320 MB, so simulate
in chunks of 10,000 and keep only the P&L.

```python
for _ in range(5):
    v = simulate_factor_t(10_000, market_loadings, sector_loadings, df=4.0)
    pnl.append(to_returns(v) @ positions)  # $125,000 in each of 800 stocks
```

**4. Read off VaR and expected shortfall.**

```
                     VaR 99%   ES 99%   diversification
truth                $2.51m    $3.33m       42.9%
factor, Student-t    $2.44m    $3.19m       42.5%
factor, Gaussian     $2.23m    $2.62m       52.2%   <- 21% too little ES
```

Diversification is how far the book's ES sits below the sum of 800 standalone
ESs. On the book's worst 1% of days, 487 of the 800 stocks are having their own
worst-5% day; independent failures would give 40. The Student-t factor copula
says 487, the Gaussian 339. That gap is the diversification a correlation matrix
promises and a panic takes away.
