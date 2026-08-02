# rcopula

Copula modelling in Python, replicating R's `copula` package and verified
against it — every deterministic quantity is checked against fixtures generated
by running the real R package.

```bash
pip install rcopula
```

```python
import rcopula as rc

cop = rc.ClaytonCopula(theta=2.0, dim=3)
u = cop.rvs(1000, random_state=42)

cop.pdf(u), cop.cdf(u)
cop.tau(), cop.rho(), cop.lambda_()

res = rc.fit(rc.GumbelCopula(), u[:, :2], method="mpl")
print(res.summary())  # estimate, standard error, z, p-value, AIC, BIC
```

## What this has that Python did not

- **Standard errors for a fitted copula.** No other Python package returns one.
- **The multiplier-bootstrap goodness-of-fit test**, which is orders of
  magnitude faster than the parametric bootstrap.
- **Nested Archimedean copulas** — dependence that varies by branch of a tree.
  These need an exponentially tilted stable sampler that exists nowhere else in
  the Python ecosystem.
- **Khoudraji's device, mixtures and rotations**, so asymmetry and two-tailed
  dependence do not require a new family.
- **The Kendall distribution function** and Kendall return periods.
- **Automatic family selection**: `rc.select_copula(u)` fits every admissible
  family and ranks them, reporting the tail dependence next to each score.
- Families missing elsewhere under one API: Joe, AMH, Plackett, FGM,
  Marshall–Olkin, Galambos, Hüsler–Reiss, Tawn, t-EV, Fréchet bounds.

## The one number to take away

Two copulas fitted to the same data, with the **same Kendall's τ**, disagree
about the things you care about:

| At τ = 0.5 | Clayton | Gumbel |
|---|---|---|
| 99% expected shortfall, two-line loss portfolio | 8.57 | 10.95 |
| Kendall return period at the 99% critical layer | 6689 years | 199 years |
| Operational-risk capital, seven cells | baseline | +20% |

A rank correlation, however carefully estimated, does not determine the answer.
Choosing the family is the modelling decision.

## Licence

MIT. R's `copula` is GPL-3 and is used **solely as a behavioural test oracle** —
no R source was read to write this package. See `NOTICE` and `CONTRIBUTING.md`.
