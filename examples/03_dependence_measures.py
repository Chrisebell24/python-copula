"""What each dependence measure sees, and what it misses.

Ported from *Elements of Copula Modeling with R*, chapter 2.6.

    ## R
    ## tau(claytonCopula(2)); rho(claytonCopula(2))
    ## iTau(gumbelCopula(), 0.5); iRho(normalCopula(), 0.5)
    ## cor(x, y, method = "kendall")
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show

import rcopula as rc

heading("Pearson correlation is not a property of the dependence")

# The textbook demonstration: y = x^2 on a symmetric x. The two are perfectly
# dependent -- knowing x determines y -- and Pearson reports nothing.
rng = np.random.default_rng(0)
x = rng.normal(size=5000)
y = x**2
show("Pearson correlation", float(np.corrcoef(x, y)[0, 1]))
show("Kendall's tau", float(rc.cor_kendall(np.column_stack([x, y]))[0, 1]))
check("Pearson sees nothing", abs(np.corrcoef(x, y)[0, 1]) < 0.05)

# Rank measures do not see it either -- the relationship is not monotone. The
# lesson is narrower than "use rank correlation": no scalar summarises
# dependence. What rank measures do guarantee is invariance.
heading("Closed forms, and calibrating to a target")

for cop in (
    rc.ClaytonCopula(2.0),
    rc.GumbelCopula(2.0),
    rc.FrankCopula(5.0),
    rc.JoeCopula(2.0),
    rc.GaussianCopula(0.7),
    rc.PlackettCopula(4.0),
):
    print(f"  {cop.name:12s} tau = {cop.tau():+.6f}   rho = {cop.rho():+.6f}")

heading("Inverting them (R's iTau / iRho)")

for target in (0.25, 0.5, 0.75):
    calibrated = {
        name: ctor.from_tau(target)
        for name, ctor in [
            ("Clayton", rc.ClaytonCopula),
            ("Gumbel", rc.GumbelCopula),
            ("Frank", rc.FrankCopula),
            ("Gaussian", rc.GaussianCopula),
            ("Plackett", rc.PlackettCopula),
        ]
    }
    shown = "  ".join(f"{n} {c.params[0]:7.4f}" for n, c in calibrated.items())
    print(f"  tau = {target}:  {shown}")
    for name, cop in calibrated.items():
        check(f"{name}.from_tau({target}) round-trips", abs(cop.tau() - target) < 1e-9)

heading("Gumbel's iTau is exactly 1/(1 - tau)")

for target in (0.25, 0.5, 0.75, 0.05, 0.45, 0.7):
    check(
        f"iTau(Gumbel, {target}) = {1 / (1 - target):.6f}",
        abs(rc.GumbelCopula.from_tau(target).theta - 1 / (1 - target)) < 1e-12,
    )

heading("Elliptical copulas share tau but not rho")

# tau = (2/pi) arcsin(rho) holds for every elliptical copula. Spearman's rho
# does NOT transfer -- (6/pi) arcsin(rho/2) is specific to the Gaussian, and R
# declines to answer for the t copula at all.
for correlation in (0.3, 0.6, 0.9):
    gaussian = rc.GaussianCopula(correlation)
    student = rc.StudentCopula(correlation, df=4.0)
    check(
        f"tau agrees at rho = {correlation}",
        abs(gaussian.tau() - student.tau()) < 1e-12,
    )
    show(f"  Spearman rho at {correlation}: Gaussian", gaussian.rho())
    show(f"  Spearman rho at {correlation}: Student(4)", student.rho())
    check("Spearman's rho does not", student.rho() < gaussian.rho())

heading("Blomqvist's beta depends on the copula only at the centre")

for cop in (rc.ClaytonCopula(2.0), rc.GumbelCopula(2.0), rc.FrankCopula(5.0)):
    show(f"{cop.name} beta", cop.beta())
