# Examples

Fourteen scripts in [`examples/`](https://github.com/Chrisebell24/python-copula/tree/main/examples),
each of which **runs and asserts its own claims** — so they cannot drift out of
date without failing.

```bash
python examples/04_tail_dependence.py
pytest tests/test_examples.py     # run them all
```

## Foundations

| | |
|---|---|
| 01 | Build a copula, evaluate it, sample it |
| 02 | Sklar's theorem both ways; the Fréchet bounds |
| 03 | What each dependence measure sees, and what it misses |
| 04 | Tail dependence: what τ does not tell you |

## Inference

| | |
|---|---|
| 05 | Five estimators, and standard errors that are calibrated |
| 06 | Goodness of fit, family selection, cross-validation |

## Applications

| | |
|---|---|
| 07 | Portfolio VaR, expected shortfall, CoVaR, stress testing |
| 08 | CDO tranches, the correlation skew, the 2008 failure mode |
| 09 | Basket, rainbow, spread and CMS spread options |
| 10 | Pairs trading on conditional copula probabilities |
| 11 | Copula-GARCH: volatility in time, dependence in the cross-section |
| 12 | Operational-risk capital, reinsurance layers, cat bonds |

## Beyond finance

| | |
|---|---|
| 13 | Kendall return periods for flood frequency |
| 14 | Asymmetric dependence and both-tail mixtures |

## Numbers these produce

All at **identical Kendall's τ**, so none of it is visible to a rank
correlation:

- Six families, two-line loss portfolio: 99% ES spans **29%**, and plain
  Clayton lands *below* the Gaussian.
- CDO senior tranche, t copula vs Gaussian: **3× worse**.
- Operational risk across seven cells: Gumbel costs **20% more capital**.
- Flood frequency, five families: Kendall return periods span a factor of **33**.
