"""Diagnostic plots: where a family fits, not merely whether.

A fitted parameter and a goodness-of-fit p-value say *whether* a family fits.
These say **where** it does not, which is what decides whether the failure
matters. Every figure is written to disk so the script can run headless.

    ## R
    ## contour(claytonCopula(3), dCopula); persp(cop, pCopula)
    ## pairs2(u); K.plot(x)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from _common import check, heading, show

import rcopula as rc
from rcopula.plots import (
    contour,
    dependence_heatmap,
    kendall_plot,
    nested_tree,
    pickands_plot,
    scatter_matrix,
    surface,
    tail_concentration,
    vine_trees,
)
from rcopula.structural import NestedArchimedean

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def save(handle, name: str) -> Path:
    """Find the figure behind whatever the plot returned.

    Some return a single axes, some a 1-d array of them (vine trees), some a
    2-d grid (the scatter matrix).
    """
    while not hasattr(handle, "figure"):
        handle = handle[0]
    figure = handle.figure
    path = OUT / f"{name}.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return path


heading("The density, seen two ways")

# A tail-dependent density spans many orders of magnitude, so its linear
# contours all crowd into one corner -- log is the readable one.
_, axes = plt.subplots(1, 2, figsize=(9, 4))
contour(rc.ClaytonCopula(3.0), kind="pdf", ax=axes[0])
contour(rc.ClaytonCopula(3.0), kind="logpdf", ax=axes[1])
show("saved", save(axes[0], "01_contour_clayton"))

show("saved", save(surface(rc.FrankCopula(5.0), kind="cdf"), "02_surface_frank"))

heading("The plot to reach for after fitting")

# Two families at the same tau agree almost everywhere and separate exactly
# where the risk is.
u = rc.ClaytonCopula.from_tau(0.5).rvs(3000, random_state=0)
ax = tail_concentration(
    u,
    [
        rc.ClaytonCopula.from_tau(0.5),
        rc.GumbelCopula.from_tau(0.5),
        rc.GaussianCopula.from_tau(0.5),
    ],
)
show("saved", save(ax, "03_tail_concentration"))

from rcopula.plots import _concentration  # noqa: E402

levels = np.array([0.01, 0.99])
clayton = _concentration(levels, rc.ClaytonCopula.from_tau(0.5).cdf(np.column_stack([levels] * 2)))
gumbel = _concentration(levels, rc.GumbelCopula.from_tau(0.5).cdf(np.column_stack([levels] * 2)))
show("lower-tail concentration at q=0.01: Clayton", float(clayton[0]))
show("lower-tail concentration at q=0.01: Gumbel", float(gumbel[0]))
check("Clayton dominates below", clayton[0] > gumbel[0] + 0.4)
check("Gumbel dominates above", gumbel[1] > clayton[1] + 0.4)

# The trap the plot exists to expose.
gaussian = rc.GaussianCopula.from_tau(0.5)
deep = np.array([1e-2, 1e-4, 1e-8])
curve = _concentration(deep, gaussian.cdf(np.column_stack([deep] * 2)))
show("Gaussian lambda_L (exactly zero)", gaussian.lambda_().lower)
for level, value in zip(deep, curve, strict=True):
    show(f"   but its concentration at q={level:.0e}", float(value))
check("still 0.27 at the 1-in-100 level", abs(curve[0] - 0.273) < 0.01)
check("and 0.10 at 1-in-10,000", abs(curve[1] - 0.103) < 0.01)

heading("Dependence without a fitted model")

# The Kendall plot needs no margins and no family: it is a function of ranks.
_, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (label, cop) in zip(
    axes,
    [
        ("independent", rc.IndependenceCopula(2)),
        ("Clayton(4)", rc.ClaytonCopula(4.0)),
        ("Frank(-8)", rc.FrankCopula(-8.0)),
    ],
    strict=True,
):
    kendall_plot(cop.rvs(500, random_state=0), ax=ax)
    ax.set_title(f"Kendall plot: {label}")
show("saved", save(axes[0], "04_kendall_plots"))

from rcopula.kendall import kendall_empirical  # noqa: E402
from rcopula.plots import _independence_order_statistics  # noqa: E402

for label, cop, direction in [
    ("independent", rc.IndependenceCopula(2), 0),
    ("Clayton(4)", rc.ClaytonCopula(4.0), 1),
    ("Frank(-8)", rc.FrankCopula(-8.0), -1),
]:
    w = kendall_empirical(cop.rvs(2000, random_state=0))
    gap = float(np.mean(w - _independence_order_statistics(w.size)))
    show(f"{label}: mean departure from the diagonal", gap)
    if direction == 0:
        check("independence sits on the diagonal", abs(gap) < 0.02)
    else:
        check(f"{label} bows {'above' if direction > 0 else 'below'} it", np.sign(gap) == direction)

heading("Extreme-value families, inside their triangle")

ax = pickands_plot(
    [
        rc.GumbelCopula(2.0),
        rc.GalambosCopula(1.5),
        rc.HuslerReissCopula(2.0),
        rc.KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(3.0), [0.4, 0.95]),
    ]
)
show("saved", save(ax, "05_pickands"))

asymmetric = rc.KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(3.0), [0.4, 0.95])
grid = np.linspace(0.01, 0.99, 99)
curve = asymmetric.pickands(grid)
check(
    "the Khoudraji curve is visibly asymmetric about t = 1/2",
    abs(curve[24] - curve[-25]) > 0.01,
)

heading("Structures whose whole point is that dependence is not one number")

vine = rc.VineCopula(
    [[rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(4.0)]], structure="D"
)
show("saved", save(vine_trees(vine), "06_vine_trees"))

nested = NestedArchimedean(
    rc.GumbelCopula(1.5),
    children=[
        NestedArchimedean(rc.GumbelCopula(4.0), [0, 1, 2]),
        NestedArchimedean(rc.GumbelCopula(3.0), [3, 4]),
    ],
)
show("saved", save(nested_tree(nested), "07_nested_tree"))
show("saved", save(dependence_heatmap(nested.tau_matrix()), "08_tau_heatmap"))

tau = nested.tau_matrix()
show("tau within the first block", float(tau[0, 1]))
show("tau within the second block", float(tau[3, 4]))
show("tau across the blocks", float(tau[0, 3]))
check(
    "three distinct values, which no flat Archimedean copula can produce",
    len({round(tau[0, 1], 6), round(tau[3, 4], 6), round(tau[0, 3], 6)}) == 3,
)

heading("Every pair at once, on the copula scale")

sample = nested.rvs(1500, random_state=0)
show("saved", save(scatter_matrix(sample, names=list("ABCDE")), "09_scatter_matrix"))

print(f"\n  {len(list(OUT.glob('*.png')))} figures written to {OUT}")
check("every figure was written", len(list(OUT.glob("*.png"))) >= 9)
