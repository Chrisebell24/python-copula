# Choosing a family

| If the data look like this | Reach for |
|---|---|
| Symmetric, no joint extremes | `GaussianCopula` |
| Symmetric, joint extremes in **both** tails | `StudentCopula` |
| Joint crashes but not joint booms | `ClaytonCopula` |
| Joint booms but not joint crashes | `GumbelCopula` |
| Flexible middle, negative dependence allowed, no tail dependence | `FrankCopula` |
| Heavier upper tail than Gumbel | `JoeCopula` |
| Only weak dependence (τ ∈ [−0.18, 1/3]) | `AMHCopula` |
| Joint maxima (floods, wind, temperature) | `GalambosCopula`, `HuslerReissCopula`, `TawnCopula`, `TEVCopula` |
| Full dependence range, no tail dependence, closed-form ρ | `PlackettCopula` |
| **Tail in the other corner** | `rc.survival(cop)`, `rc.RotatedCopula(cop, 90)` |
| **Asymmetric** — the pair does not look the same from both sides | `rc.KhoudrajiCopula(c1, c2, shapes)` |
| **Both tails at once** | `rc.MixtureCopula([clayton, gumbel], w)` |
| Blocks or a hierarchy | `rc.NestedArchimedean(...)` |

Bivariate only: Galambos, Hüsler–Reiss, Tawn, t-EV, Plackett, Marshall–Olkin,
and the Fréchet lower bound.

## Let the data choose

```python
ranking = rc.select_copula(u)  # every admissible family, ranked
print(ranking)
best = ranking.best  # fitted, ready to use
```

The table reports λ_L, λ_U and τ next to each score, so the ranking reads as a
statement about which asymmetries the data insists on rather than an opaque
leaderboard.

!!! note "Two cautions, both stated in the table rather than hidden"

    **AIC and BIC compare fit against complexity, not against the data.** The
    best of a bad set is still bad — pass `gof=True` (or `gof="mult"` for the
    fast version) to test the winner against the data itself.

    **Selection is itself an estimate.** With a few hundred observations the top
    few families are usually within noise; a difference in AIC of less than
    about 2 is not evidence. `rc.cross_validate` is the more honest comparison,
    and avoids the known bias of AIC for copulas fitted to pseudo-observations.

## A worked warning

"We used Clayton because we care about the tail" is, for **loss aggregation**,
exactly backwards. Aggregate loss risk lives in the *upper* tail; Clayton binds
in the lower one. At τ = 0.5 on two exponential margins, the 99% expected
shortfall comes out:

```
Clayton           8.57      <- lower than the Gaussian
Frank             9.00
Gaussian         10.20
Student(4)       10.56
Gumbel           10.95
survival Clayton 11.02
```

It is the **survival** Clayton that belongs there. Kendall's τ is 0.5 in every
row and reports none of this.
