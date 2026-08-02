# Changelog

Dates are ISO. Pre-1.0 the API may change; breaking changes are listed first in
each release.

## Unreleased

Everything below is in `main` and not yet published to PyPI.

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

### Examples

Twenty scripts, each of which runs and asserts its own claims. New in this
cycle: vines, the nine diagnostic plots, science and engineering domains,
statistics and machine learning, time-varying dependence, and count data.

## 0.1.0 — unreleased

The first published release will cover all families, `CopulaDistribution`, all
five fitting methods **with standard errors**, goodness-of-fit and hypothesis
tests, the empirical copula and `pseudo_obs`.
