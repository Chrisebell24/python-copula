"""Vine copulas: a joint distribution built from bivariate pieces.

An Archimedean copula gives every pair the same dependence; an elliptical one
gives every pair the same shape. A vine gives up neither flexibility nor
tractability.

    ## R (VineCopula package)
    ## RVineStructureSelect(u, familyset = c(1, 3, 4, 5))
    ## RVineLogLik(u, fit)$loglik
"""

from __future__ import annotations

import numpy as np
from _common import check, heading, show
from scipy import stats

import rcopula as rc

heading("A vine mixing three families")

truth = rc.VineCopula(
    [[rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(4.0)]],
    structure="D",
)
print(truth.describe())

u = truth.rvs(150_000, random_state=0)
show("sample shape", u.shape)
for j in range(3):
    p = stats.kstest(u[:, j], "uniform").pvalue
    check(f"margin {j} is uniform (KS p = {p:.3f})", p > 0.01)

heading("Tree 1 governs the pairs it joins")

# A D-vine's first tree pairs adjacent variables, and their dependence is
# exactly the pair-copula's -- so a vine lets a lower-tail pair and an
# upper-tail pair coexist, which no single family allows.
for (i, j), copula in zip([(0, 1), (1, 2)], truth.pair_copulas[0], strict=True):
    observed = stats.kendalltau(u[:, i], u[:, j]).statistic
    show(f"tau({i},{j}) from the sample", float(observed))
    show(f"   {copula.name} pair-copula's tau", copula.tau())
    check(f"they agree for the ({i},{j}) pair", abs(observed - copula.tau()) < 0.01)

lower = rc.ClaytonCopula(3.0).lambda_()
upper = rc.GumbelCopula(2.5).lambda_()
show("pair (0,1) lower tail dependence", lower.lower)
show("pair (1,2) upper tail dependence", upper.upper)
check("one pair binds below and the other above", lower.lower > 0.5 and upper.upper > 0.5)

heading("Validation: an all-Gaussian vine IS a Gaussian copula")

# The sharpest check available, because it is an identity rather than a
# tolerance. A vine's tree-k parameters are partial correlations, so the
# ordinary correlation matrix can be recovered and the two densities compared.
rng = np.random.default_rng(0)
for d in (3, 4, 5, 6):
    a = rng.normal(size=(d, d + 4))
    scatter = a @ a.T
    scale = np.sqrt(np.diag(scatter))
    sigma = scatter / np.outer(scale, scale)

    blank = rc.VineCopula([[rc.GaussianCopula(0.0)] * (d - 1 - k) for k in range(d - 1)])
    from rcopula.vine import _partial

    vine = rc.VineCopula(
        [
            [
                rc.GaussianCopula(float(_partial(sigma, *blank._edge_indices(k, i))))
                for i in range(d - 1 - k)
            ]
            for k in range(d - 1)
        ]
    )
    points = rng.uniform(0.05, 0.95, size=(200, d))
    gap = float(np.max(np.abs(vine.logpdf(points) - vine.to_gaussian().logpdf(points))))
    check(f"d={d}: the vine density equals the Gaussian's (max gap {gap:.1e})", gap < 1e-9)
    check(
        f"d={d}: the correlation matrix round-trips",
        np.allclose(vine.to_gaussian().sigma(), sigma, atol=1e-12),
    )

heading("The Rosenblatt transform ties the sampler to the density")

# Under the true vine the transform gives independent uniforms. Under a wrong
# one it visibly does not -- which is what makes the check worth running.
z = truth.rosenblatt(truth.rvs(20_000, random_state=1))
for j in range(3):
    p = stats.kstest(z[:, j], "uniform").pvalue
    check(f"column {j} is uniform (KS p = {p:.3f})", p > 0.01)
worst = max(
    abs(stats.kendalltau(z[:, i], z[:, j]).statistic) for i in range(3) for j in range(i + 1, 3)
)
check(f"and the columns are independent (largest |tau| = {worst:.4f})", worst < 0.03)

wrong = rc.VineCopula(
    [[rc.GumbelCopula(4.0), rc.ClaytonCopula(4.0)], [rc.FrankCopula(-6.0)]], structure="D"
)
bad = wrong.rosenblatt(truth.rvs(20_000, random_state=1))
check(
    "a misspecified vine fails the same check",
    min(stats.kstest(bad[:, j], "uniform").pvalue for j in range(3)) < 1e-6,
)

heading("Fitting: each edge's family chosen from what that edge sees")

fitted = rc.fit_vine(
    truth.rvs(4000, random_state=0),
    structure="D",
    order=[0, 1, 2],
    families=["clayton", "gumbel", "frank", "gaussian", "student"],
)
print(fitted.describe())
check(
    "tree 1's families are recovered",
    [c.name for c in fitted.pair_copulas[0]] == ["Clayton", "Gumbel"],
)

data = truth.rvs(4000, random_state=1)
show("log-likelihood, fitted vine", fitted.loglik(data))
show("log-likelihood, misspecified vine", wrong.loglik(data))
check("the fit beats the misspecified vine", fitted.loglik(data) > wrong.loglik(data))

heading("Truncation: not spending parameters on noise")

# Higher trees usually carry little. Truncating sets them to independence.
truncated = rc.fit_vine(data, structure="D", order=[0, 1, 2], truncate=1)
show("tree 2 family after truncating at 1", truncated.pair_copulas[1][0].name)
check("the higher tree is independence", truncated.pair_copulas[1][0].name == "Independence")
show("log-likelihood lost by truncating", fitted.loglik(data) - truncated.loglik(data))

heading("A five-dimensional vine, fitted from scratch")

wide = rc.VineCopula(
    [
        [rc.ClaytonCopula(4.0), rc.GumbelCopula(3.0), rc.FrankCopula(6.0), rc.GaussianCopula(0.5)],
        [rc.GaussianCopula(0.4), rc.ClaytonCopula(1.5), rc.FrankCopula(2.0)],
        [rc.GaussianCopula(0.2), rc.FrankCopula(1.0)],
        [rc.GaussianCopula(0.1)],
    ],
    structure="D",
)
sample = wide.rvs(4000, random_state=0)
selected = rc.fit_vine(sample, structure="D", order=list(range(5)))
show("pair-copulas in the model", selected.n_pairs)
print("  tree 1 families:", [c.name for c in selected.pair_copulas[0]])
print("  tree 4 family:  ", [c.name for c in selected.pair_copulas[3]])
check("the fitted vine has 10 pair-copulas", selected.n_pairs == 10)
check("it beats independence by a wide margin", selected.loglik(sample) > 500)
