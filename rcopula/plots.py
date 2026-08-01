r"""Diagnostic plots for copulas.

Numbers alone are a poor way to judge a dependence model. A fitted parameter and
a goodness-of-fit p-value tell you *whether* a family fits; these tell you
**where** it does not, which is the part that decides whether the failure
matters. A copula that misses the middle of the distribution is usually
harmless; one that misses the corner is not.

The five here are chosen because each answers a question the others cannot.

=============================  ===============================================
:func:`contour`                What the density or CDF actually looks like.
:func:`surface`                The same in three dimensions (R's ``persp``).
:func:`scatter_matrix`         Every pair at once, on the copula scale.
:func:`tail_concentration`     **Fitted against empirical, in the corners.**
:func:`kendall_plot`           A dependence diagnostic needing no fitted model.
:func:`pickands_plot`          The Pickands function inside its bounds.
:func:`vine_trees`             A vine's trees, edge by edge.
:func:`nested_tree`            A nested copula's hierarchy.
:func:`dependence_heatmap`     A pairwise matrix, when one number will not do.
=============================  ===============================================

:func:`tail_concentration` is the one to reach for after fitting. Two families
calibrated to the same Kendall's tau agree almost everywhere and separate
exactly where the risk is, so a plot that puts the empirical corner behaviour
next to the fitted one shows the disagreement that a single summary statistic
averages away.

Every function takes an optional ``ax`` and returns the axes it drew on, so the
plots compose into whatever figure you are building. Nothing is shown or saved
automatically.

References
----------
Genest, C. and Boies, J.-C. (2003). Detecting dependence with Kendall plots.
    *The American Statistician* 57(4), 275-284.
Venter, G. (2002). Tails of copulas. *Proceedings of the Casualty Actuarial
    Society* 89, 68-113.
    The tail concentration function.
Fisher, N. I. and Switzer, P. (2001). Graphical assessment of dependence: is a
    picture worth 100 tests? *The American Statistician* 55(3), 233-239.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.base import Copula
from rcopula.core.extreme_value import ExtremeValueCopula
from rcopula.dependence import cor_kendall, pseudo_obs
from rcopula.kendall import kendall_empirical

__all__ = [
    "contour",
    "dependence_heatmap",
    "kendall_plot",
    "nested_tree",
    "pickands_plot",
    "scatter_matrix",
    "surface",
    "tail_concentration",
    "vine_trees",
]

#: Grid points per axis for the surface and contour plots. The default keeps a
#: contour smooth without making a t-copula CDF (which is quadrature) slow.
_GRID = 60


def _axes(ax: Any, **kwargs: Any) -> Any:
    """Return ``ax``, or a fresh one. Imported lazily so that importing
    ``rcopula`` does not drag in pyplot and pick a backend."""
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    return plt.subplots(**kwargs)[1]


def _as_list(copula: Copula | list[Copula] | None) -> list[Copula]:
    """Accept one copula or several, so callers need not wrap a single one."""
    if copula is None:
        return []
    return [copula] if isinstance(copula, Copula) else list(copula)


def _unit_grid(margin: float, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.linspace(margin, 1.0 - margin, n)
    return np.meshgrid(x, x, indexing="ij")


def _evaluate(copula: Copula, kind: str, uu: NDArray, vv: NDArray) -> NDArray[np.float64]:
    points = np.column_stack([uu.ravel(), vv.ravel()])
    if kind == "pdf":
        values = copula.pdf(points)
    elif kind == "logpdf":
        values = copula.logpdf(points)
    elif kind == "cdf":
        values = copula.cdf(points)
    else:
        raise ValueError(f"kind must be 'pdf', 'logpdf' or 'cdf', got {kind!r}")
    return np.asarray(values).reshape(uu.shape)


def contour(
    copula: Copula,
    kind: str = "pdf",
    n: int = _GRID,
    margin: float = 0.01,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Contour plot of a bivariate copula's density or distribution function.

    Parameters
    ----------
    copula : Copula
        Bivariate, with parameters specified.
    kind : {"pdf", "logpdf", "cdf"}
        ``"logpdf"`` is usually the readable one for a tail-dependent family,
        whose density spans many orders of magnitude and whose linear contours
        therefore all crowd into one corner.
    n : int
        Grid points per axis.
    margin : float
        How far to stay off the boundary, where a tail-dependent density
        diverges.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.plots import contour
    >>> ax = contour(ClaytonCopula(3.0), kind="logpdf")
    >>> ax.get_xlabel()
    'u1'
    """
    if copula.dim != 2:
        raise ValueError(f"contour plots are bivariate; got dim={copula.dim}")
    uu, vv = _unit_grid(margin, n)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        z = _evaluate(copula, kind, uu, vv)

    ax = _axes(ax)
    kwargs.setdefault("levels", 14)
    ax.contour(uu, vv, np.nan_to_num(z, neginf=np.nanmin(z[np.isfinite(z)])), **kwargs)
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_title(f"{copula.name} copula: {kind}")
    ax.set_aspect("equal")
    return ax


def surface(
    copula: Copula,
    kind: str = "pdf",
    n: int = _GRID,
    margin: float = 0.02,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Three-dimensional surface of the density or CDF (R's ``persp``).

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from rcopula import FrankCopula
    >>> from rcopula.plots import surface
    >>> ax = surface(FrankCopula(5.0), kind="cdf")
    >>> ax.get_zlabel()
    'cdf'
    """
    if copula.dim != 2:
        raise ValueError(f"surface plots are bivariate; got dim={copula.dim}")
    uu, vv = _unit_grid(margin, n)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        z = _evaluate(copula, kind, uu, vv)

    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.figure().add_subplot(projection="3d")
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("linewidth", 0)
    ax.plot_surface(uu, vv, np.nan_to_num(z), **kwargs)
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
    ax.set_zlabel(kind)
    ax.set_title(f"{copula.name} copula")
    return ax


def scatter_matrix(
    data: ArrayLike,
    names: list[str] | None = None,
    axes: Any = None,
    **kwargs: Any,
) -> Any:
    """Pairwise scatter plots on the copula scale, annotated with Kendall's tau.

    Ranks are taken first, so the panels show *dependence* and nothing else --
    the marginal shapes that dominate a raw scatter plot are removed. Two
    variables whose raw plot is an uninformative smear can show a clean pattern
    here.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.plots import scatter_matrix
    >>> u = ClaytonCopula(3.0, dim=3).rvs(300, random_state=0)
    >>> grid = scatter_matrix(u, names=["a", "b", "c"])
    >>> grid.shape
    (3, 3)
    """
    u = pseudo_obs(data)
    d = u.shape[1]
    labels = names or [f"u{j + 1}" for j in range(d)]
    if len(labels) != d:
        raise ValueError(f"got {len(labels)} names for {d} columns")

    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(d, d, figsize=(2.2 * d, 2.2 * d), squeeze=False)
    tau = cor_kendall(u)
    kwargs.setdefault("s", 4)
    kwargs.setdefault("alpha", 0.4)

    for i in range(d):
        for j in range(d):
            ax = axes[i][j]
            if i == j:
                ax.hist(u[:, i], bins=20, color="0.7")
                ax.set_yticks([])
            elif i > j:
                ax.scatter(u[:, j], u[:, i], **kwargs)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                # The upper triangle repeats the lower one, so use it for the
                # number instead of a mirrored picture.
                ax.text(0.5, 0.5, f"tau\n{tau[i, j]:+.3f}", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            if i == d - 1:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])
    return axes


def tail_concentration(
    data: ArrayLike | None = None,
    copula: Copula | list[Copula] | None = None,
    n: int = 99,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    r"""Tail concentration functions, empirical against fitted.

    The two branches are

    .. math::
        L(q) = \frac{C(q, q)}{q}\ \ (q \le \tfrac12), \qquad
        R(q) = \frac{1 - 2q + C(q, q)}{1 - q}\ \ (q > \tfrac12),

    which tend to :math:`\lambda_L` and :math:`\lambda_U` at the ends. Plotting
    them together makes visible the thing a single dependence measure hides:
    families calibrated to the same Kendall's tau agree in the middle and
    separate in the corners, and the corners are where the money is.

    Parameters
    ----------
    data : array_like, optional
        Observations, plotted as the empirical curve.
    copula : Copula or list of Copula, optional
        One or more fitted copulas, plotted as reference curves.

    Examples
    --------
    Two families at the same tau, and the picture that distinguishes them:

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from rcopula import ClaytonCopula, GumbelCopula
    >>> from rcopula.plots import tail_concentration
    >>> u = ClaytonCopula.from_tau(0.5).rvs(2000, random_state=0)
    >>> ax = tail_concentration(u, [ClaytonCopula.from_tau(0.5), GumbelCopula.from_tau(0.5)])
    >>> ax.get_ylabel()
    'concentration'
    """
    if data is None and copula is None:
        raise ValueError("give data, a copula, or both")

    q = np.linspace(1.0 / (n + 1), n / (n + 1), n)
    ax = _axes(ax)

    if data is not None:
        u = pseudo_obs(data)
        if u.shape[1] != 2:
            raise ValueError(f"tail concentration is bivariate; got {u.shape[1]} columns")
        joint = np.array([np.mean((u[:, 0] <= t) & (u[:, 1] <= t)) for t in q])
        ax.plot(q, _concentration(q, joint), label="empirical", color="black", **kwargs)

    for cop in _as_list(copula):
        if cop.dim != 2:
            raise ValueError(f"tail concentration is bivariate; got dim={cop.dim}")
        joint = cop.cdf(np.column_stack([q, q]))
        ax.plot(q, _concentration(q, joint), label=cop.name, linestyle="--")

    ax.axvline(0.5, color="0.8", linewidth=0.8)
    ax.set_xlabel("q")
    ax.set_ylabel("concentration")
    ax.set_title("Tail concentration: lower tail left, upper tail right")
    ax.legend()
    return ax


def _concentration(q: NDArray[np.float64], joint: NDArray[np.float64]) -> NDArray[np.float64]:
    """``C(q,q)/q`` below the centre and the survival ratio above it."""
    return np.where(q <= 0.5, joint / q, (1.0 - 2.0 * q + joint) / (1.0 - q))


def kendall_plot(
    data: ArrayLike,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    r"""Kendall plot (K-plot), a dependence diagnostic with no fitted model.

    Genest & Boies (2003) plot the ordered statistics :math:`W_{i}` of the
    empirical Kendall function against their expected values *under
    independence*. Points on the diagonal mean independence; bowing above it
    means positive dependence, below means negative -- and the shape of the
    departure carries information a scalar correlation does not.

    Its appeal is that it needs neither margins nor a fitted family: it is a
    function of the ranks alone, so it can be looked at before any modelling
    decision has been made.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from rcopula import ClaytonCopula
    >>> from rcopula.plots import kendall_plot
    >>> u = ClaytonCopula(3.0).rvs(300, random_state=0)
    >>> ax = kendall_plot(u)
    >>> ax.get_xlabel()
    'expected under independence'
    """
    w = kendall_empirical(data)
    n = w.size
    expected = _independence_order_statistics(n)

    ax = _axes(ax)
    kwargs.setdefault("s", 8)
    ax.scatter(expected, w, **kwargs)
    ax.plot([0, 1], [0, 1], color="0.5", linewidth=0.8)
    ax.set_xlabel("expected under independence")
    ax.set_ylabel("ordered W")
    ax.set_title("Kendall plot")
    ax.set_aspect("equal")
    return ax


def _independence_order_statistics(n: int, nodes: int = 400) -> NDArray[np.float64]:
    r""":math:`E[W_{i:n}]` under independence, for the Kendall plot.

    With :math:`K_0(w) = w - w\log w` the independence Kendall function and
    :math:`k_0(w) = -\log w` its density,

    .. math::
        W_{i:n} = n\binom{n-1}{i-1}\int_0^1 w\,k_0(w)\,K_0(w)^{i-1}
                  \bigl(1 - K_0(w)\bigr)^{n-i}\,\mathrm{d}w,

    the usual order-statistic expectation. Computed in logs, since the binomial
    coefficient overflows well before the sample sizes people actually use.
    """
    from scipy.special import gammaln

    x, weights = np.polynomial.legendre.leggauss(nodes)
    w = 0.5 * (x + 1.0)
    wt = 0.5 * weights

    with np.errstate(divide="ignore", invalid="ignore"):
        k0 = w - w * np.log(w)
        density = -np.log(w)
        log_k0 = np.log(np.clip(k0, 1e-300, 1.0))
        log_1mk0 = np.log(np.clip(1.0 - k0, 1e-300, 1.0))

    i = np.arange(1, n + 1)
    log_binomial = np.log(n) + gammaln(n) - gammaln(i) - gammaln(n - i + 1.0)
    integrand = w * density * wt
    log_shape = (i[:, None] - 1.0) * log_k0[None, :] + (n - i[:, None]) * log_1mk0[None, :]

    # The binomial coefficient and the integral diverge in opposite directions:
    # at n = 2000 the coefficient alone is e^1403 while the integrand is e^-1400,
    # so forming either separately gives inf * 0 = nan. Factor the row maximum
    # out of the integral and recombine in logs, where both are ordinary.
    peak = log_shape.max(axis=1)
    inner = np.exp(log_shape - peak[:, None]) @ integrand
    return np.asarray(np.exp(log_binomial + peak) * inner)


def pickands_plot(
    copula: Copula | list[Copula],
    n: int = 201,
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    r"""The Pickands dependence function inside its admissible triangle.

    Every extreme-value copula is determined by a convex :math:`A` with
    :math:`\max(t, 1-t) \le A(t) \le 1`. The upper edge is independence and the
    lower one comonotonicity, so the plot places a family between those two
    extremes and shows at a glance how much dependence it carries and whether it
    is symmetric -- which is the property Khoudraji's device exists to break.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from rcopula import GalambosCopula, GumbelCopula
    >>> from rcopula.plots import pickands_plot
    >>> ax = pickands_plot([GumbelCopula(2.0), GalambosCopula(1.5)])
    >>> ax.get_ylabel()
    'A(t)'
    """
    from rcopula.core.archimedean import GumbelCopula
    from rcopula.core.extreme_value import gumbel_pickands
    from rcopula.structural.khoudraji import KhoudrajiCopula

    t = np.linspace(0.0, 1.0, n)
    ax = _axes(ax)
    ax.plot(t, np.maximum(t, 1.0 - t), color="0.6", linewidth=0.8)
    ax.plot(t, np.ones_like(t), color="0.6", linewidth=0.8)

    for cop in _as_list(copula):
        if isinstance(cop, ExtremeValueCopula):
            a = cop.A(t)
        elif isinstance(cop, GumbelCopula):
            a = gumbel_pickands(t, cop.theta)
        elif isinstance(cop, KhoudrajiCopula) and cop.is_extreme_value:
            a = cop.pickands(t)
        else:
            raise ValueError(
                f"{cop.name} is not an extreme-value copula, so it has no "
                "Pickands dependence function"
            )
        ax.plot(t, a, label=cop.name, **kwargs)

    ax.set_xlabel("t")
    ax.set_ylabel("A(t)")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Pickands dependence function")
    ax.legend()
    return ax


def vine_trees(
    vine: Any,
    max_trees: int | None = None,
    axes: Any = None,
    **kwargs: Any,
) -> Any:
    """Draw a vine's trees, edge labels annotated with each pair-copula.

    A vine is a sequence of trees, and the thing a reader needs to see is which
    pairs each tree joins, what it conditions on, and which family was selected
    there. Tree 1 carries most of the dependence and is where a misspecified
    family costs most; the higher trees are usually where a fit is spending
    parameters on noise, which this makes visible at a glance.

    Parameters
    ----------
    vine : VineCopula
        A fitted or specified vine.
    max_trees : int, optional
        Draw only the first few trees. Defaults to all of them.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import rcopula as rc
    >>> from rcopula.plots import vine_trees
    >>> vine = rc.VineCopula(
    ...     [[rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(4.0)]],
    ...     structure="D",
    ... )
    >>> grid = vine_trees(vine)
    >>> len(grid)
    2
    """
    depth = len(vine.pair_copulas) if max_trees is None else min(max_trees, len(vine.pair_copulas))
    if axes is None:
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(1, depth, figsize=(4.2 * depth, 3.8), squeeze=False)
        axes = axes[0]

    for k in range(depth):
        ax = axes[k]
        # Lay the tree's nodes on a circle: readable for a star and for a path.
        nodes = sorted({idx for i in range(len(vine.pair_copulas[k]))
                        for idx in vine._edge_indices(k, i)[:2]})
        angle = {node: 2 * np.pi * j / len(nodes) for j, node in enumerate(nodes)}
        position = {node: (np.cos(a), np.sin(a)) for node, a in angle.items()}

        for i, copula in enumerate(vine.pair_copulas[k]):
            a, b, conditioning = vine._edge_indices(k, i)
            (x0, y0), (x1, y1) = position[a], position[b]
            ax.plot([x0, x1], [y0, y1], color="0.6", linewidth=1.0, zorder=1, **kwargs)
            ax.text(
                0.5 * (x0 + x1), 0.5 * (y0 + y1),
                f"{copula.name}\n{_short_params(copula)}",
                ha="center", va="center", fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.2"},
                zorder=3,
            )
        for node, (x, y) in position.items():
            label = str(vine.order[node])
            conditioning = vine._edge_indices(k, 0)[2]
            if k > 0 and conditioning:
                label = f"{vine.order[node]}|·"
            ax.scatter([x], [y], s=280, color="0.9", edgecolor="0.4", zorder=2)
            ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=4)

        ax.set_title(f"tree {k + 1}")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
    return axes


def _short_params(copula: Copula) -> str:
    """A compact parameter string for an edge label."""
    if not len(copula.params):
        return ""
    return ", ".join(f"{v:.2f}" for v in copula.params)


def nested_tree(node: Any, ax: Any = None, **kwargs: Any) -> Any:
    """Draw a nested Archimedean copula's hierarchy.

    The whole point of nesting is that dependence varies by branch, so the
    picture worth drawing is the tree with each node's parameter and Kendall's
    tau on it -- from which the pairwise tau of any two leaves can be read off
    directly, since two variables meet at exactly one node.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import rcopula as rc
    >>> from rcopula.plots import nested_tree
    >>> from rcopula.structural import NestedArchimedean
    >>> cop = NestedArchimedean(
    ...     rc.GumbelCopula(1.5),
    ...     children=[
    ...         NestedArchimedean(rc.GumbelCopula(4.0), [0, 1, 2]),
    ...         NestedArchimedean(rc.GumbelCopula(3.0), [3, 4]),
    ...     ],
    ... )
    >>> ax = nested_tree(cop)
    >>> ax.get_title()
    'Nested Gumbel copula'
    """
    ax = _axes(ax)
    leaf_x = [0.0]

    def draw(current: Any, depth: int) -> float:
        """Place children first, then centre the parent over them."""
        spots = []
        for leaf in current.components:
            x = leaf_x[0]
            leaf_x[0] += 1.0
            ax.scatter([x], [-depth - 1.0], s=220, color="white", edgecolor="0.4", zorder=3)
            ax.text(x, -depth - 1.0, str(leaf), ha="center", va="center", fontsize=8, zorder=4)
            spots.append(x)
        for child in current.children:
            spots.append(draw(child, depth + 1))

        centre = float(np.mean(spots)) if spots else 0.0
        for x in spots:
            ax.plot([centre, x], [-depth, -depth - 1.0], color="0.6", linewidth=1.0, zorder=1)
        ax.scatter([centre], [-depth], s=520, color="0.92", edgecolor="0.35", zorder=3, **kwargs)
        ax.text(
            centre, -depth,
            f"{current.theta:.2f}\ntau {current.generator_copula.tau():.2f}",
            ha="center", va="center", fontsize=7, zorder=4,
        )
        return centre

    draw(node, 0)
    ax.set_title(f"Nested {node.generator_copula.name} copula")
    ax.axis("off")
    ax.margins(0.15)
    return ax


def dependence_heatmap(
    values: ArrayLike,
    names: list[str] | None = None,
    label: str = "Kendall's tau",
    ax: Any = None,
    **kwargs: Any,
) -> Any:
    """Heat map of a pairwise dependence matrix, annotated with the numbers.

    Useful precisely for the constructions whose whole point is that dependence
    is *not* one number: a nested copula's :meth:`tau_matrix`, or the empirical
    matrix from a sample.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import rcopula as rc
    >>> from rcopula.plots import dependence_heatmap
    >>> from rcopula.structural import NestedArchimedean
    >>> cop = NestedArchimedean(
    ...     rc.GumbelCopula(1.5),
    ...     children=[
    ...         NestedArchimedean(rc.GumbelCopula(4.0), [0, 1, 2]),
    ...         NestedArchimedean(rc.GumbelCopula(3.0), [3, 4]),
    ...     ],
    ... )
    >>> ax = dependence_heatmap(cop.tau_matrix())
    >>> ax.get_title()
    "Kendall's tau"
    """
    matrix = np.atleast_2d(np.asarray(values, dtype=np.float64))
    d = matrix.shape[0]
    if matrix.shape != (d, d):
        raise ValueError(f"expected a square matrix, got {matrix.shape}")
    labels = names or [str(j) for j in range(d)]

    ax = _axes(ax)
    kwargs.setdefault("cmap", "RdBu_r")
    kwargs.setdefault("vmin", -1.0)
    kwargs.setdefault("vmax", 1.0)
    ax.imshow(matrix, **kwargs)
    for i in range(d):
        for j in range(d):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(d), labels)
    ax.set_yticks(range(d), labels)
    ax.set_title(label)
    return ax
