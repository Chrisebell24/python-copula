# Coming from R

The API is Pythonic (scipy/statsmodels verbs), but R's option strings are
preserved so code ports across with the strings intact.

## Construction and evaluation

| R | `rcopula` |
|---|---|
| `claytonCopula(2, dim = 3)` | `rc.ClaytonCopula(theta=2, dim=3)` |
| `normalCopula(0.5, dim = 3, dispstr = "un")` | `rc.GaussianCopula([...], dim=3, dispstr="un")` |
| `tCopula(0.5, df = 4, df.fixed = TRUE)` | `rc.StudentCopula(0.5, df=4, df_fixed=True)` |
| `rCopula(n, cop)` | `cop.rvs(n)` |
| `dCopula(u, cop)` / `pCopula(u, cop)` | `cop.pdf(u)` / `cop.cdf(u)` |
| `prob(cop, l, u)` | `cop.prob(l, u)` |
| `getSigma(cop)` / `p2P` / `P2p` | `cop.sigma()` / `rc.p2P` / `rc.P2p` |
| `fixParam(cop, c(FALSE))` | `cop.fix_params([False])` |
| `describeCop(cop)` | `cop.describe()` |

## Dependence measures

| R | `rcopula` |
|---|---|
| `tau(cop)` / `rho(cop)` / `lambda(cop)` | `cop.tau()` / `cop.rho()` / `cop.lambda_()` |
| `beta(cop)` | `cop.beta()` |
| `iTau(claytonCopula(), 0.5)` | `rc.ClaytonCopula.from_tau(0.5)` |
| `iRho(normalCopula(), 0.5)` | `rc.GaussianCopula.from_rho(0.5)` |
| `cor(x, method = "kendall")` | `rc.cor_kendall(x)` |
| `pobs(x)` | `rc.pseudo_obs(x)` |

## Inference

| R | `rcopula` |
|---|---|
| `fitCopula(cop, u, method = "mpl")` | `rc.fit(cop, u, method="mpl")` |
| `fitMvdc(x, mvdc, start)` | `rc.fit_joint(mv, x, method="ifm")` |
| `coef(fit)` / `vcov(fit)` / `logLik(fit)` | `res.params` / `res.cov_params` / `res.loglik` |
| `xvCopula(cop, x, k = 10)` | `rc.cross_validate(cop, x, k=10)` |
| `loglikCopula(par, u, cop)` | `rc.loglik_copula(par, u, cop)` |
| *(loop by hand)* | `rc.select_copula(u)` |

## Testing

| R | `rcopula` |
|---|---|
| `gofCopula(cop, x, simulation = "pb")` | `rc.gof_test(cop, x, simulation="pb")` |
| `gofCopula(cop, x, simulation = "mult")` | `rc.gof_test(cop, x, simulation="mult")` |
| `exchTest(x)` / `radSymTest(x)` | `rc.exch_test(x)` / `rc.rad_sym_test(x)` |
| `indepTest(x, d)` / `evTestC(x)` | `rc.indep_test(x)` / `rc.ev_test(x)` |

## Structural constructions

| R | `rcopula` |
|---|---|
| `rotCopula(cop, flip)` | `rc.RotatedCopula(cop, flip)`, `rc.survival(cop)` |
| `khoudrajiCopula(c1, c2, shapes)` | `rc.KhoudrajiCopula(c1, c2, shapes)` |
| `mixCopula(list(c1, c2), w)` | `rc.MixtureCopula([c1, c2], w)` |
| `margCopula(cop, keep)` | `rc.marginal_copula(cop, indices)` |
| `fitLambda(x, method = ...)` | `rc.fit_lambda(x)` |
| `onacopula("G", C(1.5, , list(C(4, 1:3))))` | `rc.NestedArchimedean(rc.GumbelCopula(1.5), children=[...])` |
| `enacopula(u, cop, method = "etau")` | `rc.fit_nested(structure, u)` |

## Transforms, Kendall function, plots

| R | `rcopula` |
|---|---|
| `cCopula(u, cop)` | `rc.rosenblatt(cop, u)` |
| `mvdc(cop, margins, paramMargins)` | `rc.CopulaDistribution(cop, margins=[...])` |
| `pK` / `qK` / `dK` / `rK` | `rc.kendall_cdf` / `_ppf` / `_pdf` / `_rvs` |
| `Kn(u, x)` | `rc.kendall_empirical(x, u)` |
| `contour(cop, dCopula)` / `persp(...)` | `rc.plots.contour(cop)` / `rc.plots.surface(cop)` |
| `retstable(n, V0, h, alpha)` | `rcopula.special.stable.retstable(...)` |
| `htrafo(u, cop)` | `rc.htrafo(cop, u)` |
| `dependogram(x, d)` | `rc.dependogram(x)` + `rc.plots.dependogram_plot(result)` |
| *(no equivalent)* | `rc.radial_simplex(cop, u)` — the McNeil-Neslehova split |
| inverse `cCopula` | `rc.inverse_rosenblatt(cop, z)` |
| `rAntitheticVariates(n, d)` | `rc.sampling.antithetic_rvs(cop, n)` |
| `rLatinHypercube(n, d)` | `rc.sampling.latin_hypercube_rvs(cop, n)` |

## Things R has no equivalent for

Not translations -- there is nothing on the left-hand side to translate.

| | `rcopula` |
|---|---|
| Rank every admissible family at once | `rc.select_copula(u, criterion="bic")` |
| Dependence that moves over time | `rc.dynamic.fit_dynamic(u, cop, driver="patton")` |
| ...score-driven instead | `rc.dynamic.fit_dynamic(u, cop, driver="gas")` |
| A whole correlation matrix that moves | `rc.dynamic.fit_dcc(u)` |
| Genuinely discrete margins | `rc.discrete.fit_discrete(x, cop, margins)` |
| ...mixed with a continuous one | `rc.discrete.mixed_pdf(cop, x, margins, discrete)` |
| The ceiling ties impose on tau | `rc.discrete.tau_upper_bound(margins)` |
| Bootstrap intervals, in parallel | `rc.bootstrap.bootstrap_measure(u, "tau", n_jobs=4)` |
| ...for a fitted parameter | `rc.bootstrap.bootstrap_fit(x, cop)` |
| Save a model to a readable file | `rc.serialize.to_json(cop)` / `from_json(text)` |
| Vines (R/C/D) | `rc.fit_vine(u, structure="D")` |
| Screen a universe for tradable pairs | `rc.statarb.select_pairs(returns, method=...)` |
| Pick vine partners for a target | `rc.statarb.select_partners(returns, target)` |
| Sobol/Halton draws from a copula | `rc.sampling.quasi_rvs(cop, n)` |
| Measure what variance reduction bought | `rc.sampling.variance_ratio(cop, payoff, n)` |

## Differences worth knowing

- **`fit()` returns a results object, not a modified copula.** Copulas here are
  immutable; `with_params` returns a new one.
- **`iTau`/`iRho` are classmethods** (`from_tau`, `from_rho`) because they
  construct rather than mutate.
- **Margins are scipy frozen distributions**, not name strings with parameter
  lists — anything with `cdf`/`pdf`/`ppf` works.
- **Non-integer degrees of freedom are supported.** `mvtnorm::pmvt` refuses
  them, even though `fitCopula` happily produces them.
- **AMH works in `d > 2`.** R's `amhCopula` is bivariate-only, though its
  `copAMH` generator object is not.
- Places where R is the less accurate side are listed under
  [How parity is verified](parity.md).
