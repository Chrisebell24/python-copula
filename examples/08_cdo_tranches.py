"""CDO tranche pricing, the correlation skew, and the 2008 failure mode.

Validated against the Vasicek large-homogeneous-pool closed form, which the
Monte-Carlo model must reproduce as the pool grows.

.. warning::
   A reference implementation for understanding the mechanics, not a production
   pricing library.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from _common import check, heading, show

import rcopula as rc

N_NAMES = 125
DEFAULT_PROB = 0.05  # five-year cumulative
LGD = 0.6
TRANCHES = [(0.00, 0.03), (0.03, 0.07), (0.07, 0.10), (0.10, 0.15), (0.15, 1.00)]

heading("The one-factor Gaussian copula model")

rho = 0.3
cop = rc.GaussianCopula(rho, dim=N_NAMES)
loss = rc.credit.portfolio_loss(cop, DEFAULT_PROB, lgd=LGD, n=100_000, random_state=0)
show("expected pool loss", float(loss.mean()))
show("analytic expected loss (p x LGD)", DEFAULT_PROB * LGD)
check(
    "the simulated pool loss matches the analytic mean",
    abs(loss.mean() - DEFAULT_PROB * LGD) < 0.002,
)

print(f"\n  {'tranche':<14}{'expected loss':>15}{'spread (bp)':>14}")
for attach, detach in TRANCHES:
    el = rc.credit.tranche_expected_loss(loss, attach, detach)
    spread = rc.credit.tranche_spread(loss, attach, detach, maturity=5.0)
    print(f"  {attach:>5.0%}-{detach:<7.0%}{el:>15.4f}{1e4 * spread:>14.1f}")

heading("Correlation is not a neutral assumption")

print(f"  {'rho':<8}" + "".join(f"{f'{a:.0%}-{d:.0%}':>12}" for a, d in TRANCHES))
for rho in (0.0, 0.1, 0.3, 0.6, 0.9):
    cop = rc.GaussianCopula(rho, dim=N_NAMES)
    loss = rc.credit.portfolio_loss(cop, DEFAULT_PROB, lgd=LGD, n=60_000, random_state=0)
    row = [rc.credit.tranche_expected_loss(loss, a, d) for a, d in TRANCHES]
    print(f"  {rho:<8.2f}" + "".join(f"{v:>12.4f}" for v in row))

print(
    "\n  Read down the columns. Correlation moves the equity and senior tranches\n"
    "  in OPPOSITE directions: higher correlation makes 'everyone survives' and\n"
    "  'everyone defaults' both more likely, which protects the equity tranche\n"
    "  and exposes the senior one. A single number cannot be conservative for\n"
    "  the whole structure."
)

equity_low = rc.credit.tranche_expected_loss(
    rc.credit.portfolio_loss(
        rc.GaussianCopula(0.1, dim=N_NAMES), DEFAULT_PROB, lgd=LGD, n=60_000, random_state=0
    ),
    0.0,
    0.03,
)
equity_high = rc.credit.tranche_expected_loss(
    rc.credit.portfolio_loss(
        rc.GaussianCopula(0.9, dim=N_NAMES), DEFAULT_PROB, lgd=LGD, n=60_000, random_state=0
    ),
    0.0,
    0.03,
)
senior_low = rc.credit.tranche_expected_loss(
    rc.credit.portfolio_loss(
        rc.GaussianCopula(0.1, dim=N_NAMES), DEFAULT_PROB, lgd=LGD, n=60_000, random_state=0
    ),
    0.15,
    1.0,
)
senior_high = rc.credit.tranche_expected_loss(
    rc.credit.portfolio_loss(
        rc.GaussianCopula(0.9, dim=N_NAMES), DEFAULT_PROB, lgd=LGD, n=60_000, random_state=0
    ),
    0.15,
    1.0,
)
check("equity loss FALLS as correlation rises", equity_high < equity_low)
check("senior loss RISES as correlation rises", senior_high > senior_low)

heading("Validation: the Vasicek large-pool limit")

# As the pool grows, the loss distribution converges to a closed form. The
# simulation must reproduce it -- this is the check that the model is right.
for rho in (0.1, 0.3, 0.6):
    big = rc.GaussianCopula(rho, dim=800)
    simulated = rc.credit.portfolio_loss(big, DEFAULT_PROB, lgd=1.0, n=100_000, random_state=0)
    for q in (0.5, 0.9, 0.99):
        level = float(np.quantile(simulated, q))
        analytic = float(rc.credit.vasicek_loss_cdf(level, DEFAULT_PROB, rho))
        check(
            f"rho={rho}: Vasicek CDF at the simulated {q:.0%} quantile is {analytic:.4f}",
            abs(analytic - q) < 0.02,
        )

heading("The 2008 failure mode: the same tranches under a t copula")

# Same marginal default probabilities, same rank correlation. Only the tail
# behaviour of the dependence differs.
gaussian = rc.GaussianCopula.from_tau(rc.GaussianCopula(0.3).tau(), dim=N_NAMES)
student = rc.StudentCopula.from_tau(rc.GaussianCopula(0.3).tau(), dim=N_NAMES, df=4.0)

print(f"  {'tranche':<14}{'Gaussian':>12}{'Student(4)':>13}{'ratio':>9}")
for attach, detach in TRANCHES:
    g = rc.credit.tranche_expected_loss(
        rc.credit.portfolio_loss(gaussian, DEFAULT_PROB, lgd=LGD, n=60_000, random_state=0),
        attach,
        detach,
    )
    s = rc.credit.tranche_expected_loss(
        rc.credit.portfolio_loss(student, DEFAULT_PROB, lgd=LGD, n=60_000, random_state=0),
        attach,
        detach,
    )
    ratio = s / g if g > 1e-9 else float("inf")
    print(f"  {attach:>5.0%}-{detach:<7.0%}{g:>12.5f}{s:>13.5f}{ratio:>9.2f}")

print(
    "\n  The senior tranche is where the two models disagree most, by a large\n"
    "  multiple. Senior tranches were the ones rated AAA, and the ones that\n"
    "  the Gaussian assumption said were nearly riskless."
)

heading("Basket default swaps")

small = rc.GaussianCopula(0.3, dim=10)
for k in (1, 2, 5, 10):
    p = rc.credit.nth_to_default_probability(small, DEFAULT_PROB, n_th=k, n=100_000, random_state=0)
    show(f"P(at least {k} of 10 default)", p)

probabilities = [
    rc.credit.nth_to_default_probability(small, DEFAULT_PROB, n_th=k, n=100_000, random_state=0)
    for k in (1, 2, 5, 10)
]
check("higher k is less likely", all(a >= b for a, b in pairwise(probabilities)))
