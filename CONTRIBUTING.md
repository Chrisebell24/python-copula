# Contributing to `rcopula`

## The clean-room rule (read this first)

`rcopula` replicates the functionality of the R package **`copula`**, which is licensed
**GPL (>= 3)**. `rcopula` is **MIT**. To keep that legitimate:

> **Do not read R's `copula` source code in order to write Python code for this project.**
> Implement every algorithm from its originating published paper. Validate only against
> R's *outputs*.

Concretely:

| Allowed | Not allowed |
|---|---|
| Reading R's **documentation** (man pages, vignettes) for behaviour and signatures | Reading R's `.R` / `.c` sources to see *how* something is computed |
| Reading the **papers** the algorithms come from | Transliterating R code into Python |
| Running R to record numeric outputs into `tests/golden/` | Copying R comments, variable names, or code structure |
| Matching R's argument names and option strings | Copying R's internal helper implementations |

**Every module that implements a nontrivial algorithm must cite its source paper in the
module docstring.** A PR that adds an algorithm without a citation will be asked for one.
See `NOTICE` for the running reference list.

If you are unsure whether something crosses the line, ask in an issue before writing code.

## Development setup

```bash
git clone https://github.com/Chrisebell24/python-copula
cd python-copula

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"

pytest -q
ruff check . && ruff format --check .
mypy rcopula
```

## Regenerating golden fixtures (needs R)

Fixtures under `tests/golden/` are committed, so **CI never needs R**. Regenerate them only
when the R `copula` version changes or you add new parity coverage:

```bash
brew install r          # or your platform's equivalent
Rscript -e 'install.packages(c("copula","lcopula","qrmtools","jsonlite"), repos="https://cloud.r-project.org")'
make golden
```

Fixtures are written with 17 significant digits so they round-trip exactly through
`float64`. Commit the regenerated files together with the code change that motivated them,
and note the R and `copula` versions in the PR description.

## Testing conventions

- `pytest -m golden` — parity against R fixtures.
- `pytest -m slow` — bootstrap and Monte-Carlo level studies. Not run on every commit.
- Numerics get **two** independent checks: an `mpmath` high-precision reference *and* an R
  golden fixture, exercised at extreme arguments as well as typical ones.
- Deterministic quantities (pdf, cdf, tau, rho, lambda, iTau/iRho, Pickands A, K function,
  GoF statistics on fixed data) must match R to ~1e-10. See the "Bucket A / Bucket B"
  section in the docs for why sampling is validated differently.

## Style

- Public API follows scipy/statsmodels conventions: `.pdf`/`.cdf`/`.logpdf`/`.rvs`,
  `random_state` accepting a `numpy.random.Generator`, `fit()` returning a **results
  object** rather than `self`.
- Keep R's exact option strings (`method="mpl"`, `"itau.mpl"`, `dispstr="ar1"`, …) so R
  users can paste them across.
- Type-annotate public functions. `disallow_untyped_defs` is off for now but new public
  code should be annotated anyway.
