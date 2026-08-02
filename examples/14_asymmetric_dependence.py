"""Breaking exchangeability, and getting both tails at once.

Every Archimedean copula is exchangeable and every equicorrelated elliptical one
is too. Plenty of data is not. These are the two constructions that escape,
neither of which requires inventing a new family.
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show

import rcopula as rc

heading("Rotations: moving a family's tail to the other corner")

base = rc.ClaytonCopula.from_tau(0.5)
print(f"  {'copula':<26}{'tau':>9}{'lambda_L':>11}{'lambda_U':>11}")
for label, cop in [
    ("Clayton", base),
    ("survival Clayton (180)", rc.survival(base)),
    ("rotated 90", rc.RotatedCopula(base, 90)),
    ("rotated 270", rc.RotatedCopula(base, 270)),
]:
    lam = cop.lambda_()
    print(f"  {label:<26}{cop.tau():>9.4f}{lam.lower:>11.4f}{lam.upper:>11.4f}")

check(
    "reflecting both coordinates swaps the tails",
    rc.survival(base).lambda_().upper == base.lambda_().lower,
)
check(
    "reflecting one coordinate reverses the dependence",
    rc.RotatedCopula(base, 90).tau() == -base.tau(),
)
check(
    "Clayton and Gumbel admit no negative dependence at all",
    rc.ClaytonCopula.from_tau(0.5).tau() > 0,
)

# Rotations compose as REFLECTIONS, not as rotations: two 90s cancel.
check(
    "90 then 90 returns to the original",
    rc.RotatedCopula(rc.RotatedCopula(base, 90), 90).degrees == 0,
)
check("90 then 270 gives 180", rc.RotatedCopula(rc.RotatedCopula(base, 90), 270).degrees == 180)

heading("Khoudraji's device: genuine asymmetry")

asymmetric = rc.KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(3.0), [0.4, 0.95])
show("C(0.3, 0.7)", float(asymmetric.cdf([[0.3, 0.7]])[0]))
show("C(0.7, 0.3)", float(asymmetric.cdf([[0.7, 0.3]])[0]))
check(
    "swapping the arguments changes the value",
    abs(asymmetric.cdf([[0.3, 0.7]])[0] - asymmetric.cdf([[0.7, 0.3]])[0]) > 0.01,
)
check(
    "no Archimedean copula can do that",
    abs(base.cdf([[0.3, 0.7]])[0] - base.cdf([[0.7, 0.3]])[0]) < 1e-15,
)

show("Kendall's tau", asymmetric.tau())
show("upper tail dependence", asymmetric.lambda_().upper)

# Both components extreme-value means the result is too, so the tail
# dependence is exact rather than estimated.
check("the construction stays in the extreme-value class", asymmetric.is_extreme_value)
q = 1 - 1e-9
limit = (1 - 2 * q + float(asymmetric.cdf([[q, q]])[0])) / (1 - q)
check(
    f"lambda_U = 2(1 - A(1/2)) matches the diagonal limit {limit:.6f}",
    abs(asymmetric.lambda_().upper - limit) < 1e-6,
)

# And the test for exchangeability sees it.
sample = asymmetric.rvs(800, random_state=0)
result = rc.exch_test(sample, n_rep=400, random_state=0)
show("exchangeability test p-value", float(result.pvalue))
check("the asymmetry is detected", result.pvalue < 0.05)

symmetric = rc.exch_test(base.rvs(800, random_state=0), n_rep=400, random_state=0)
check("and Clayton is not falsely flagged", symmetric.pvalue > 0.05)

heading("Mixtures: both tails at once")

mixture = rc.MixtureCopula([rc.ClaytonCopula(4.0), rc.GumbelCopula(3.0)], [0.5, 0.5])
lam = mixture.lambda_()
show("lower tail dependence", lam.lower)
show("upper tail dependence", lam.upper)
check("both tails are dependent", lam.lower > 0.3 and lam.upper > 0.3)
for component in mixture.copulas:
    check(f"but {component.name} alone has only one", min(component.lambda_()) == 0.0)

# Three measures mix exactly; Kendall's tau does not.
weights = np.array([0.5, 0.5])
show("Spearman rho, mixture", mixture.rho())
show("Spearman rho, weighted average", float(np.dot(weights, [c.rho() for c in mixture.copulas])))
check(
    "rho is exactly the weighted average",
    abs(mixture.rho() - np.dot(weights, [c.rho() for c in mixture.copulas])) < 1e-12,
)

half = rc.MixtureCopula([rc.FrechetUpperCopula(2), rc.IndependenceCopula(2)], [0.5, 0.5])
show("half comonotone, half independent: rho", half.rho())
show("half comonotone, half independent: tau", half.tau())
check("rho is exactly 0.5", abs(half.rho() - 0.5) < 1e-9)
check("tau is NOT -- it is quadratic in the copula", abs(half.tau() - 0.5) > 0.08)

heading("Fitting the structures, not just naming them")

truth = rc.MixtureCopula([rc.ClaytonCopula(4.0), rc.GumbelCopula(3.0)], [0.3, 0.7])
data = truth.rvs(4000, random_state=0)
fitted = rc.fit(
    rc.MixtureCopula([rc.ClaytonCopula(), rc.GumbelCopula()], [0.5, 0.5]), data, method="mpl"
)
show("recovered weight on Clayton", float(fitted.copula.weights[0]))
show("recovered Clayton theta", float(fitted.copula.copulas[0].theta))
show("recovered Gumbel theta", float(fitted.copula.copulas[1].theta))
check("the weights are recovered", abs(fitted.copula.weights[0] - 0.3) < 0.12)

truth_k = rc.KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(3.0), [0.4, 0.95])
fitted_k = rc.fit(
    rc.KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(), [0.5, 0.5]),
    truth_k.rvs(4000, random_state=0),
    method="mpl",
)
show("recovered shape 1", float(fitted_k.copula.shapes[0]))
show("recovered shape 2", float(fitted_k.copula.shapes[1]))
check("the shapes are recovered", np.allclose(fitted_k.copula.shapes, [0.4, 0.95], atol=0.12))

heading("Where the structures matter: a loss portfolio")


def shortfall(cop: rc.Copula) -> float:
    losses = -np.log1p(-cop.rvs(300_000, random_state=0)).sum(axis=1)
    return float(rc.risk.expected_shortfall(losses, 0.995))


tau = 0.5
candidates = {
    "Clayton": rc.ClaytonCopula.from_tau(tau),
    "survival Clayton": rc.survival(rc.ClaytonCopula.from_tau(tau)),
    "Gumbel": rc.GumbelCopula.from_tau(tau),
    "Clayton+Gumbel mix": rc.MixtureCopula(
        [rc.ClaytonCopula.from_tau(tau), rc.GumbelCopula.from_tau(tau)], [0.5, 0.5]
    ),
}
for name, cop in candidates.items():
    show(f"99.5% expected shortfall, {name}", shortfall(cop))

check(
    "the mixture sits between its components",
    shortfall(candidates["Clayton"])
    < shortfall(candidates["Clayton+Gumbel mix"])
    < shortfall(candidates["Gumbel"]),
)
