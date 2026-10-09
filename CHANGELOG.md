# Changelog

Dates are ISO. Pre-1.0 the API may change; breaking changes are listed first in
each release.

## 0.5.0 — 2026-10-09

### Added

- **Regular vines**: `VineCopula(..., structure="R", matrix=...)` (R
  `VineCopula` matrix convention, 0-based) with density, sampling, Rosenblatt
  transform, `describe`, truncation and serialization; `.matrix` and
  `.to_rvine()` on every vine. `fit_vine(structure="R")` selects the trees with
  Dissmann's algorithm and supports `truncate=`. On R's own test data it picks
  the same matrix and all ten families as `RVineStructureSelect`.
- **More pair-copula families**: `BB1Copula`, `BB7Copula`, and Joe plus the
  90/180/270 rotations for vines (`EXTENDED_FAMILIES`; `select_copula` groups
  `"rotated"`, `"bb"`, `"vine"`). Checked against R's VineCopula.
- `VineCopula.rosenblatt` now works for C-vines.

### Changed

- Full vines sample with closed-form inverse h-functions: 15–40× faster at
  d = 10. Seeded draws agree with 0.4.0 to about 1e-13.
- `mpl` fits with more than two parameters converge to a tighter tolerance;
  their log-likelihood now matches or beats R's. One- and two-parameter fits
  keep the default, which already matches R and keeps rolling refits fast.

### Fixed

- `fit(..., "itau"/"irho")` for `dispstr="toep"`/`"ar1"` above two dimensions
  fits the structure as R does (Toeplitz used to come back with every lag
  equal; AR(1) estimated 0.40 for a true 0.6).
- `"irho"` for unstructured t copulas uses the t copula's Spearman relation,
  not the Gaussian one (0.677 for a true 0.7 at df = 2.5).
- C-vines with pair-copulas whose arguments cannot be swapped (90/270
  rotations) used inconsistent argument order between density, sampling and
  fitting.
- Near independence: Frank tau/rho/CDF/generators, Plackett CDF, and Clayton's
  generators lost precision or broke; `from_tau`/`from_rho` failed for tiny
  targets (Frank, Gumbel, Joe); `rosenblatt` failed at theta = 0; subnormal
  theta is treated as independence.
- Debye functions accurate to ~1e-16 (exact Bernoulli numbers); Student-t
  factor copula CDF accurate at low df.
- Saving and reloading a `MixtureCopula` is exact (the document now carries
  its parameter vector; older documents still load).

### Tests

- R fixtures in 4 and 5 dimensions (`tools/rgolden/11_highdim.R`): elliptical
  with un/toep/ar1 structures, fits and standard errors, Archimedean, nested,
  marginal and Rosenblatt.
- R parity for regular vines (density, Rosenblatt, structure selection) and
  BB/rotated pair-copulas.
- Property-based tests with hypothesis (`hypothesis` added to the dev extra).

## 0.4.0 — 2026-10-08

### Added

- **Factor copulas** (`rcopula.factor`): `FactorCopula` and `fit_factor`, a
  Gaussian or Student-t copula driven by one market factor plus optional group
  (e.g. sector) factors. Densities use the Woodbury identity, so 800 stocks
  need an 11 × 11 solve rather than an 800 × 800 one; fitting 800 stocks takes
  seconds. Works with `CopulaDistribution`, `CopulaGarch`, `fit` and
  serialization. Example 32 now uses it instead of hand-rolled numpy.
- **GJR-GARCH and ARMA means**: `fit_garch(..., vol="gjr", mean="ar1" |
  "arma11" | "zero")`, carried through `CopulaGarch` fitting, simulation and
  forecasting. Defaults are unchanged and give identical results. Checked
  against R's `rugarch` (`tools/rgolden/10_garch.R`).

### Packaging

- **matplotlib is optional**: `pip install "rcopula[plots]"`. `import rcopula`
  no longer needs it; plot functions raise an ImportError explaining how to
  install it.
- Ships a `py.typed` marker, so type checkers use rcopula's annotations.
- Python 3.14 supported and tested in CI.
- The unused `garch` extra (`arch`) is removed; `rcopula.garch` is pure NumPy.
- ruff and mypy are pinned in the `dev` extra so local and CI linting agree.
- Tagged releases also publish a GitHub Release with that version's changelog
  section; releases for 0.1.0–0.3.0 were backfilled.

## 0.3.0 — 2026-10-08

### Breaking

- **`fit()` returns only the estimated parameters for every method.** `params`
  and `param_names` (now plain `str`) hold the free parameters for `itau`,
  `irho` and `itau.mpl` as they already did for `mpl`/`ml`, matching R;
  `result.copula.params` is still the full vector. Inversion now respects
  parameters pinned with `fix_params`, and a bivariate t keeps its `df` and
  `dispstr` instead of coming back at `df=4`.
- **`VineCopula.rosenblatt` returns columns in the original variable order**,
  consistent with `rvs` and `logpdf`, not in the vine's internal order.
- **`StudentCopula.lambda_` raises for non-exchangeable `dim > 2`** instead of
  silently reporting the (1, 2) pair; use `marginal_copula(cop, [i, j]).lambda_()`.
- **`gof_statistic(..., "Tn")` is `sqrt(n) * max|C_n - C|`** (Genest et al.
  2009); it was `n * max`. P-values are unchanged.
- **`backtest_pairs`** computes Sharpe over all live periods (flat periods earn
  zero) and counts entries, not every position change. Earlier Sharpe ratios
  were inflated — example 10's fell from +14–55 to +0.6–0.8.
- **`credit.tranche_spread`** discounts both legs, so spreads change whenever
  `discount_rate` is non-zero; `catastrophe_bond` annualises the expected loss
  for `multiple` and `expected_return` (new key `annual_expected_loss`).
- Parameterless copulas and `EmpiricalCopula` reject unknown keyword arguments.

### Fixed

- Copula families: `from_tau`/`from_rho` fill every correlation for `"toep"`
  and `"un"`; `StudentCopula.from_rho` no longer returns the edge of a
  too-narrow search; Frank reaches tau near ±1; Clayton/Frank reject negative
  tau above two dimensions with a clear message; `from_tau(0)` works for Joe,
  Gumbel and Tawn; `EmpiricalCopula` smoothings handle ties exactly;
  `CopulaDistribution` and `fit_joint` read DataFrames by column name;
  MarshallOlkin, Plackett and FGM argument handling.
- Inference: `itau.mpl` honours `start`, `optim_method` and
  `estimate_variance`; `fit_joint` keeps names and stops counting pinned margin
  parameters; `select_copula` gives every family identical folds; GoF argument
  checks happen before fitting; clearer errors in `ev_test`,
  `return_period_level`, `conditional_ppf`, `kendall_return_period`,
  `radial_simplex`.
- Finance: `pairs_signal` honours `exit_band`; `mean_cvar_weights` uses sparse
  matrices (20,000 scenarios in under 200 MB); `SmileMargin` extrapolates
  lognormal tails; `spread_option` prices puts; `implied_correlation` uses
  common random numbers over a wider range; input validation across `credit`,
  `derivatives`, `garch` and `risk`.
- Modelling: `bootstrap` supports matrix-valued statistics; random tie-breaking
  in `pseudo_obs` is truly random; the GAS forecast's first step is the
  filter's next value; `fit_discrete`'s likelihood-ratio statistic; `mixed_pdf`
  works above two dimensions; `select_pairs(top=0)`; `select_partners` label
  handling; cached datasets are checked against their digest; `to_json`
  accepts fit results.
- Structural and numerics: `fit_nested` clips to each family's range (no more
  Gumbel crash on weak dependence); `marginal_copula` keeps every Archimedean
  parameter; `plots.vine_trees` labels higher trees correctly; the t-copula CDF
  is finite and correct at tiny `df`.

### Performance

- `VineCopula.rvs`, `logpdf` and `fit_vine` stop at a truncated vine's last
  dependent tree: an 801-variable one-tree vine samples 50,000 draws in ~20 s
  (previously impractical). New `VineCopula.truncation_level`.
- `cor_kendall` is computed in blocks, bit-identical to scipy: 800 columns in
  ~5 s instead of ~4 minutes.

### Documentation

- Every public function and class has a plain-English docstring with explicit
  parameter types, shapes and defaults.
- New tutorial and example 32: copulas for an 800-stock basket.

## 0.2.0 — 2026-10-08

### Breaking

- **Unstructured correlation vectors now follow R's order from four dimensions
  up.** `p2P`, `P2p` and the `rho.ij` parameter names filled the lower triangle
  row by row; R fills it column by column, so for `dim >= 4` the same vector
  meant a different matrix. Code that passes an explicit parameter list to a
  `dispstr="un"` copula with `dim >= 4` must reorder it to
  `(rho_12, rho_13, ..., rho_1d, rho_23, ...)`. Up to `dim = 3` nothing changes,
  which is why the R golden fixtures (all `dim <= 3`) never caught it.
  Saved models still load correctly: the serialization schema moves to 2, and
  schema-1 documents are reordered on read.

### Fixed

- `fit(..., method="itau")`, `"irho"` and `"itau.mpl"` returned scrambled
  correlation matrices for unstructured elliptical copulas with `dim >= 4` —
  the pairwise statistics were in R's order and the matrix builder was not.
- The Clayton density lost all precision near independence: at `theta = 1e-17`
  its log-density was +4 per observation instead of 0, enough for `fit_vine` to
  pick a "Clayton" edge that was really independence and report a
  log-likelihood ~2,000 units too high.
- `fit_garch` could return `alpha + beta > 1` (a non-stationary model) when the
  optimiser stopped early near the boundary; it now refits with persistence as
  a bounded parameter so the constraint cannot be broken.

### Documentation

- Four step-by-step finance tutorials — vine copulas for markets, risk
  management, trading strategies, and valuing odd assets — in the README and on
  a new documentation page, each backed by a runnable example (28–31) that
  asserts every number quoted.
- README links are absolute, so they work on PyPI; the documentation site is
  published.

## 0.1.0

The first release. All families, `CopulaDistribution`, all five fitting methods
**with standard errors**, goodness-of-fit and hypothesis tests, the empirical
copula and `pseudo_obs` — plus everything below, which R's `copula` has no
equivalent for.

### Beyond R's `copula`

- **Time-varying copulas** (`rcopula.dynamic`). Patton (2006) forcing-term and
  score-driven (GAS) recursions for a bivariate parameter, and Engle's DCC for a
  correlation matrix in higher dimensions. Filtering, simulation, a forecast that
  reports the parameter's *distribution* rather than a point, and a likelihood
  ratio against constancy — documented as the diagnostic it is, since the null
  sits on a boundary.
- **Discrete and mixed margins** (`rcopula.discrete`). The exact
  inclusion–exclusion mass function, mixed densities, maximum likelihood, the
  distributional transform, and `tau_upper_bound`. Rank-based estimation is
  deliberately *not* offered: with ties the sample τ does not estimate the
  copula's τ, and example 20 measures the resulting bias.
- **JSON serialization** (`rcopula.serialize`). Exact round trips — a reloaded
  copula returns bit-identical densities. Structural constructions and vines
  nest. `EmpiricalCopula` is refused, because it is its data.
- **Bootstrap confidence intervals** (`rcopula.bootstrap`). Percentile, basic and
  BCa, resampling whole rows, with `n_jobs`. Coverage measured at 94.5–95.5%
  against nominal 95%.
- **`htrafo` and `radial_simplex`** (`rcopula.transforms`). The Hering–Hofert
  transform needs no high-order generator derivatives, so it works at *d* = 100
  where the Rosenblatt transform degrades.
- **Pair and partner selection for statistical arbitrage** (`rcopula.statarb`).
  Six criteria for pairs and the four vine partner-selection approaches of
  Stübinger, Mangold and Krauss, including Schmid and Schmidt's multivariate
  Spearman's rho.
- **Quasi-random and variance-reduced sampling** (`rcopula.sampling`), on top of
  a new `inverse_rosenblatt`. Sobol and Halton point sets pushed through the
  copula (Cambou–Hofert–Lemieux 2017), plus antithetic pairing and Latin
  hypercube designs, and `variance_ratio` to measure what each actually bought
  rather than assuming the theoretical rate.
- **Nonparametric tail dependence** (`rcopula.fit_lambda`, R's `fitLambda`).
  Two estimators with standard errors, and the whole threshold path, because
  there is no threshold-free estimator of a tail coefficient and reading one
  number off a plateau is the only defensible way to use it.
- **The outer power transformation** (`rcopula.opower`, R's `opower`). A second
  parameter on any Archimedean generator, moving Kendall's tau by
  `1 - (1 - tau)/alpha` and creating upper tail dependence in families that have
  none.
- **Lower-dimensional margins** (`rcopula.marginal_copula`, R's `margCopula`).
  R supports the elliptical and Archimedean classes only; the structural
  constructions come too, and a nested Archimedean's bivariate margin is read
  straight off the tree as its lowest common ancestor's generator.
- **`radial_cdf` / `radial_ppf`** (R's `pacR` / `qacR`). The law of an
  Archimedean copula's radial part, which is where all the family-specific
  information lives once the angular half is split off.
- **`gof_two_sample`** (R's `gofT2stat`). Compares two samples' copulas to each
  other with no model between them -- for "did the dependence change after the
  crisis?" -- and is blind to the margins by construction.
- **`serial_indep_test`** (R's `serialIndepTest`) and **`to_emp_margins`**.
  The first embeds a series in its own lags and runs the dependogram on it, so
  it names which lag structure carries the dependence; the second maps uniforms
  back onto a sample's empirical margins.
- **`pairs_rosenblatt`** (R's `pairsRosenblatt`). A goodness-of-fit test says
  whether a copula fits; this says where it does not, panel by panel.
- **Dependogram** (`rcopula.dependogram`, `rcopula.plots.dependogram_plot`).
  Independence decomposed over every subset of the coordinates via the Mobius
  transform, so it locates the dependence rather than only detecting it.
- **Joint fitting of margins and copula** (`rcopula.fit_joint`, R's `fitMvdc`),
  by inference functions for margins or full maximum likelihood.
- **Automatic family selection** (`rcopula.select_copula`), **vines**
  (`rcopula.vine`), **nested Archimedean copulas** and the **exponentially tilted
  stable sampler** they need (`rcopula.special.stable.retstable`) — none of which
  had a Python implementation anywhere.

### Validation

- `tests/test_literature.py`: 240 checks that consult no fixture at all, each
  computing a quantity from its published definition by a different route than
  the package uses. Spearman's ρ from 12∬C − 3 agrees to 1e-11 for every
  bivariate family; Kendall's τ from Nelsen's identity to 1e-9; Blomqvist's β
  exactly; tail dependence as converging sequences. Plus the structural
  identities — gamma frailty gives Clayton, positive-stable gives Gumbel, the
  Nataf transform is a Gaussian copula, mutual information is copula entropy.
- Two constants pinned against `mpmath` at 25 digits rather than against
  anything in this package.

### Data

- Five sources, none vendored. Two live endpoints (USGS peak flows, NOAA
  GHCN-Daily) and three static UCI tables (abalone, red wine quality, NASA
  aerofoil self-noise) whose SHA-256 is verified on every fetch.

### Documentation

A [vine copula tutorial](docs/vines.md) that reproduces Aas, Czado, Frigessi and
Bakken (2009) rather than asserting agreement with it: their equation (11) and
the canonical-vine factorisation of section 2.3, typed out by hand and compared
against `VineCopula.logpdf` (2e-15), the h-function definition against a finite
difference, and their count of 12 D-vines and 12 C-vines in four dimensions
reproduced by enumeration.

### Examples

Twenty-seven scripts, each of which runs and asserts its own claims. New in this
cycle: vines, the nine diagnostic plots, science and engineering domains,
statistics and machine learning, time-varying dependence, count data, and
quasi-random sampling.
