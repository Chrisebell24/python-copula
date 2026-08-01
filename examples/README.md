# Examples

Every script here **runs and asserts**. There is no prose claiming a result that
the code does not check: run `python examples/01_...py` and it either prints its
output and exits 0, or it fails loudly. `pytest tests/test_examples.py` runs all
of them.

Where a script ports something from R's `copula` package, the original R lines
sit in a comment block beside the Python so the two can be compared directly.

## Foundations

| | |
|---|---|
| [01](01_construction_and_sampling.py) | Build a copula, evaluate it, sample it, look at it |
| [02](02_sklar_and_frechet_bounds.py) | Sklar's theorem in both directions; the bounds that constrain every copula |
| [03](03_dependence_measures.py) | Why rank correlation beats Pearson, and what each measure misses |
| [04](04_tail_dependence.py) | The one property that separates families calibrated to the same τ |

## Inference

| | |
|---|---|
| [05](05_fitting.py) | Five estimators, and the standard errors no other Python package reports |
| [06](06_goodness_of_fit.py) | Testing a family against the data, and choosing among families |

## Applications

| | |
|---|---|
| [07](07_risk_aggregation.py) | Portfolio VaR and expected shortfall — and why "Clayton for losses" is backwards |
| [08](08_cdo_tranches.py) | Tranche pricing, the correlation skew, and the 2008 failure mode |
| [09](09_basket_options.py) | Basket and spread options that respect each underlying's own smile |
| [10](10_pairs_trading.py) | Conditional copula probabilities as a trading signal |
| [11](11_copula_garch.py) | Volatility in time, dependence in the cross-section |
| [12](12_operational_risk.py) | Loss-distribution approach and the Basel 99.9% capital number |

## Beyond finance

| | |
|---|---|
| [13](13_hydrology_return_periods.py) | Kendall return periods, and why the univariate one is the wrong number |
| [14](14_asymmetric_dependence.py) | Breaking exchangeability, and getting both tails at once |
