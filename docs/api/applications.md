# Applications

Reference implementations for analysis, **not production pricing or capital
libraries**. Each module states its modelling assumptions.

::: rcopula.risk
::: rcopula.credit
::: rcopula.derivatives
::: rcopula.portfolio
::: rcopula.statarb

## Copula-GARCH margins

`fit_garch` and `CopulaGarch.fit` take two model switches. `vol="gjr"` adds the
GJR leverage term (volatility rises more after a fall than after a rally of the
same size); `mean="zero"`, `"ar1"` or `"arma11"` replaces the constant mean.
The defaults (`vol="garch"`, `mean="constant"`) are the GARCH(1,1) of earlier
releases, with identical results. Fits are checked against R's `rugarch`
(`tools/rgolden/10_garch.R`).

```python
from rcopula.garch import fit_garch

res = fit_garch(returns, dist="t", vol="gjr", mean="ar1")
res.gamma, res.phi  # leverage and AR(1) coefficients
res.forecast_vol(10)  # accounts for gamma: persistence = alpha + gamma/2 + beta
res.forecast_mean(10)  # AR(1) mean decaying to mu / (1 - phi)
```

::: rcopula.garch
::: rcopula.factor
::: rcopula.dynamic
::: rcopula.discrete
::: rcopula.insurance
