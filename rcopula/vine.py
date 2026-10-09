r"""Vine copulas: a joint distribution built from bivariate pieces.

Archimedean copulas give every pair the same dependence; elliptical ones give
every pair the same *shape*. A vine gives up neither flexibility nor
tractability: it factorises a :math:`d`-dimensional density into
:math:`d(d-1)/2` **bivariate** copulas, each of which may be a different family
with a different parameter.

.. math::

    c(u_1,\dots,u_d) = \prod_{k=1}^{d-1}\prod_{i}
        c_{\,e_{k,i}}\bigl(F(u_a \mid \mathbf u_D),\, F(u_b \mid \mathbf u_D)\bigr)

The conditioning arguments are **h-functions** -- conditional distribution
functions -- and every one of them is available in closed form for the
Archimedean and elliptical families, which is what makes the construction
practical rather than merely decomposable.

Three structures are implemented here:

* a **C-vine** (canonical) puts one variable at the centre of each tree, which
  suits a market factor plus its satellites;
* a **D-vine** (drawable) lays each tree out as a path, which suits an ordering
  -- a term structure, a spatial transect, a time series of maturities;
* a **regular vine** (R-vine) allows any sequence of trees that satisfies the
  *proximity condition* -- two edges may be joined in the next tree only if
  they share a node. C- and D-vines are the two extreme special cases. An
  R-vine is written down as an *R-vine matrix* (Dissmann et al. 2013), the
  same convention as R's ``VineCopula`` package.

============================  ================================================
:class:`VineCopula`           The construction: density, sampler, Rosenblatt.
:func:`fit_vine`              Sequential estimation, selecting each pair (and,
                              for an R-vine, each tree).
:data:`EXTENDED_FAMILIES`     Pair-copula families with rotations and BB1/BB7.
============================  ================================================

Correctness has an unusually sharp check available. A vine whose pair-copulas
are **all Gaussian is itself a Gaussian copula**, with a correlation matrix
determined by the pair parameters through the partial-correlation recursion. So
:meth:`VineCopula.to_gaussian` reconstructs that matrix and the vine's density
can be compared against :class:`~rcopula.core.elliptical.GaussianCopula`
directly -- an exact identity, not a tolerance.

References
----------
Joe, H. (1996). Families of m-variate distributions with given margins and
    m(m-1)/2 bivariate dependence parameters. In *Distributions with Fixed
    Marginals and Related Topics*, IMS Lecture Notes 28, 120-141.
Bedford, T. and Cooke, R. M. (2002). Vines -- a new graphical model for
    dependent random variables. *Annals of Statistics* 30(4), 1031-1068.
Aas, K., Czado, C., Frigessi, A. and Bakken, H. (2009). Pair-copula
    constructions of multiple dependence.
    *Insurance: Mathematics and Economics* 44(2), 182-198.
    The likelihood and simulation algorithms implemented here.
Dissmann, J., Brechmann, E. C., Czado, C. and Kurowicka, D. (2013). Selecting
    and estimating regular vine copulae and application to financial returns.
    *Computational Statistics & Data Analysis* 59, 52-69.
    The sequential selection used by :func:`fit_vine`, and the R-vine matrix.
Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall/CRC,
    section 6.17 -- the inverse h-functions used by the sampler.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.base import Copula, TailDependence
from rcopula.core.elliptical import GaussianCopula, P2p
from rcopula.dependence import pseudo_obs
from rcopula.transforms import conditional_cdf, conditional_ppf

__all__ = ["EXTENDED_FAMILIES", "VineCopula", "fit_vine"]

#: Structures this module builds.
STRUCTURES = ("C", "D", "R")

#: Families tried by :func:`fit_vine` when none are named. Deliberately spans
#: the tail-dependence possibilities -- none, lower only, upper only, both --
#: since that is what a pair-copula choice is really deciding.
DEFAULT_FAMILIES = ("independence", "gaussian", "student", "clayton", "gumbel", "frank")

#: A wider candidate set for :func:`fit_vine`: the defaults plus Joe, the
#: 90/180/270-degree rotations of Clayton, Gumbel and Joe, and the
#: two-parameter BB1 and BB7 families with their rotations. Rotations by 90 and
#: 270 degrees are what let a positive-only family describe *negative*
#: dependence; the 180-degree (survival) versions move the tail dependence to
#: the other corner. Names are keys of :data:`rcopula.select.FAMILIES`.
EXTENDED_FAMILIES = (
    "independence",
    "gaussian",
    "student",
    "frank",
    "clayton",
    "clayton90",
    "clayton180",
    "clayton270",
    "gumbel",
    "gumbel90",
    "gumbel180",
    "gumbel270",
    "joe",
    "joe90",
    "joe180",
    "joe270",
    "bb1",
    "bb1_90",
    "bb1_180",
    "bb1_270",
    "bb7",
    "bb7_90",
    "bb7_180",
    "bb7_270",
)

#: Clipping applied to every conditional value, as everywhere in this module.
_LO, _HI = 1e-12, 1.0 - 1e-12


def _is_independence(copula: Copula) -> bool:
    from rcopula.core.other import IndependenceCopula

    return isinstance(copula, IndependenceCopula)


def _pair_h(copula: Copula, points: NDArray[np.float64], given: int) -> NDArray[np.float64]:
    """Unclipped h-function of a bivariate copula, using every closed form available.

    Rotated copulas are reduced to their base -- a reflection of the
    conditioning coordinate only relabels it, a reflection of the other one
    turns ``P(U <= x)`` into ``1 - P(U <= 1 - x)`` -- so a rotated Clayton gets
    Clayton's exact h-function instead of a numerical derivative of the
    inclusion-exclusion CDF. Families with their own ``hfunc`` (BB1, BB7) use
    it; everything else goes through :func:`~rcopula.transforms.conditional_cdf`.
    """
    from rcopula.structural.rotated import RotatedCopula

    if isinstance(copula, RotatedCopula) and copula.dim == 2:
        flip = copula.flip
        reflected = points.copy()
        reflected[:, flip] = 1.0 - reflected[:, flip]
        inner = _pair_h(copula.base, reflected, given)
        return 1.0 - inner if flip[1 - given] else inner
    hfunc = getattr(copula, "hfunc", None)
    if callable(hfunc):
        return np.asarray(hfunc(points, given), dtype=np.float64)
    return conditional_cdf(copula, points, given=given)


def _h(copula: Copula, first: NDArray, second: NDArray, given: int) -> NDArray[np.float64]:
    r"""An h-function: :math:`\partial C/\partial u_{\text{given}}`.

    ``given=1`` returns the conditional distribution of the *first* argument
    given the second, and ``given=0`` the other way round.
    """
    points = np.column_stack([first, second])
    return np.clip(_pair_h(copula, points, given), _LO, _HI)


def _h_inverse(
    copula: Copula,
    target: NDArray,
    given: NDArray,
    side: int,
    closed_form: bool = True,
    cache: dict[Any, Any] | None = None,
) -> NDArray[np.float64]:
    """Invert :func:`_h` in its first (``side=1``) or second (``side=0``) slot.

    ``side`` is the slot the *conditioning* value ``given`` occupies, as in
    :func:`~rcopula.transforms.conditional_ppf`. Wherever an exact inverse
    exists it is used -- Gaussian, Student t, Clayton (``theta > 0``), Frank,
    every rotation of those, and BB1/BB7 through their own Newton solver -- and
    otherwise (Gumbel, Joe, the extreme-value families, ...) the generic
    60-step bisection on :func:`_h` is the fallback. ``closed_form=False``
    forces the bisection everywhere, which is how rcopula 0.3.0 and earlier
    sampled a full vine. ``cache`` remembers a conditioning variable's
    quantiles between elliptical calls that share it, as every tree-1 edge of
    a C-vine does.
    """
    if closed_form:
        exact = _closed_h_inverse(copula, target, given, side, cache)
        if exact is not None:
            return np.clip(exact, _LO, _HI)
    return np.clip(conditional_ppf(copula, target, given, given=side), _LO, _HI)


def _closed_h_inverse(
    copula: Copula,
    target: NDArray,
    given: NDArray,
    side: int,
    cache: dict[Any, Any] | None = None,
) -> NDArray[np.float64] | None:
    """The exact inverse h-function, or ``None`` when the family has none."""
    from rcopula.core.archimedean import ClaytonCopula, FrankCopula
    from rcopula.core.elliptical import StudentCopula
    from rcopula.structural.rotated import RotatedCopula

    if copula.dim != 2:
        return None
    if isinstance(copula, RotatedCopula):
        flip = copula.flip
        cond_slot, free_slot = side, 1 - side
        w = np.asarray(target, dtype=np.float64)
        c = np.asarray(given, dtype=np.float64)
        if flip[cond_slot]:
            c = 1.0 - c
        if flip[free_slot]:
            w = 1.0 - w
        inner = _h_inverse(copula.base, np.clip(w, _LO, _HI), np.clip(c, _LO, _HI), side)
        return 1.0 - inner if flip[free_slot] else inner
    if isinstance(copula, GaussianCopula | StudentCopula):
        return _elliptical_h_inverse(copula, target, given, cache)
    w = np.clip(np.asarray(target, dtype=np.float64), _LO, _HI)
    c = np.clip(np.asarray(given, dtype=np.float64), _LO, _HI)
    if isinstance(copula, ClaytonCopula) and copula.theta > 0.0:
        return _clayton_h_inverse(w, c, copula.theta)
    if isinstance(copula, FrankCopula):
        return _frank_h_inverse(w, c, copula.theta)
    hinv = getattr(copula, "hinv", None)
    if callable(hinv):
        return np.asarray(hinv(w, c, given=side), dtype=np.float64)
    return None


def _clayton_h_inverse(w: NDArray, c: NDArray, theta: float) -> NDArray[np.float64]:
    r"""Clayton: :math:`x = \bigl(1 + c^{-\theta}(w^{-\theta/(1+\theta)} - 1)\bigr)^{-1/\theta}`."""
    inner = np.exp(-theta * np.log(c)) * np.expm1(-theta / (1.0 + theta) * np.log(w))
    return np.exp(-np.log1p(inner) / theta)


def _frank_h_inverse(w: NDArray, c: NDArray, theta: float) -> NDArray[np.float64]:
    r"""Frank, solved for :math:`e^{-\theta x} - 1`; the upper half via radial symmetry.

    :math:`x = -\log\bigl(1 + w(e^{-\theta}-1)/(w + (1-w)e^{-\theta c})\bigr)/\theta`.
    Near ``x = 1`` that cancels, so there the symmetry
    :math:`h(1-x \mid 1-c) = 1 - h(x \mid c)` is used to solve for ``1 - x``
    instead.
    """
    if theta == 0.0:
        return np.asarray(w, dtype=np.float64)

    def lower(ww: NDArray, cc: NDArray) -> NDArray[np.float64]:
        ratio = ww * np.expm1(-theta) / (ww + (1.0 - ww) * np.exp(-theta * cc))
        return -np.log1p(ratio) / theta

    x = lower(w, c)
    upper = x > 0.5
    if np.any(upper):
        x = np.where(upper, 1.0 - lower(1.0 - w, 1.0 - c), x)
    return x


def _elliptical_h_inverse(
    copula: Copula,
    target: NDArray,
    given: NDArray,
    cache: dict[Any, Any] | None = None,
) -> NDArray[np.float64]:
    """Closed-form inverse h-function of a bivariate Gaussian or t copula.

    Both h-functions are symmetric in the two slots, so ``side`` does not enter:
    ``x = F(rho * b + s * G^{-1}(w))`` with ``b = F^{-1}(given)``, ``F`` the
    margin's CDF and ``G``/``s`` the conditional's distribution and scale.
    """
    from scipy.special import ndtr, ndtri, stdtr, stdtrit

    from rcopula.core.elliptical import StudentCopula

    w = np.clip(np.asarray(target, dtype=np.float64), _LO, _HI)
    rho = float(copula.sigma()[0, 1])  # type: ignore[attr-defined]
    nu = float(copula.df) if isinstance(copula, StudentCopula) else np.inf
    key = (id(given), nu)
    entry = None if cache is None else cache.get(key)
    # The array is kept alongside its quantiles, so a recycled id() can never
    # hand back another array's values.
    b = entry[1] if entry is not None and entry[0] is given else None
    if b is None:
        c = np.clip(np.asarray(given, dtype=np.float64), _LO, _HI)
        b = stdtrit(nu, c) if np.isfinite(nu) else ndtri(c)
        if cache is not None:
            cache[key] = (given, b)
    if np.isfinite(nu):
        scale = np.sqrt((nu + b**2) * (1.0 - rho**2) / (nu + 1.0))
        x = stdtr(nu, rho * b + scale * stdtrit(nu + 1.0, w))
    else:
        x = ndtr(rho * b + np.sqrt(1.0 - rho**2) * ndtri(w))
    return np.clip(np.asarray(x, dtype=np.float64), _LO, _HI)


def _swap_arguments(copula: Copula) -> Copula:
    """The same bivariate copula with its two arguments exchanged.

    Exchangeable families are returned as they are; a 90/270-degree rotation
    becomes the opposite rotation. Anything else that is not exchangeable is
    refused, since silently mis-orienting an edge would change the model.
    """
    from rcopula.structural.rotated import RotatedCopula

    if isinstance(copula, RotatedCopula) and copula.dim == 2:
        return RotatedCopula(_swap_arguments(copula.base), copula.flip[::-1].copy())
    if _is_exchangeable(copula):
        return copula
    raise ValueError(
        f"cannot re-orient the non-exchangeable {copula.name} pair-copula onto this "
        "R-vine edge; use an exchangeable family or a rotation"
    )


def _is_exchangeable(copula: Copula) -> bool:
    """``C(u, v) == C(v, u)``, from the class where known, else checked numerically."""
    from rcopula.core.archimedean import ArchimedeanCopula
    from rcopula.core.bb import BB1Copula, BB7Copula
    from rcopula.core.elliptical import EllipticalCopula
    from rcopula.core.other import IndependenceCopula

    if isinstance(
        copula, ArchimedeanCopula | EllipticalCopula | IndependenceCopula | BB1Copula | BB7Copula
    ):
        return True
    if np.isnan(copula.params).any():
        return False
    probe = np.random.default_rng(0).uniform(0.05, 0.95, size=(16, 2))
    return bool(np.allclose(copula.cdf(probe), copula.cdf(probe[:, ::-1]), rtol=1e-10, atol=1e-12))


# --------------------------------------------------------------------------
# R-vine matrices
# --------------------------------------------------------------------------


class _Edge(NamedTuple):
    """One edge of an R-vine, read off the matrix.

    The pair-copula is evaluated at ``(u_{first|D}, u_{second|D})`` where
    ``D = conditioning``; ``second`` is the column's diagonal variable.
    """

    tree: int
    column: int
    first: int
    second: int
    conditioning: tuple[int, ...]


def _normalise_matrix(matrix: ArrayLike) -> NDArray[np.int64]:
    """Square integer array with the upper triangle zeroed; shape checks only."""
    arr = np.asarray(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        raise ValueError(f"an R-vine matrix must be square with d >= 2, got shape {arr.shape}")
    if not np.all(np.isfinite(arr.astype(float))) or np.any(arr != np.round(arr)):
        raise ValueError("an R-vine matrix must hold integer variable labels")
    out = np.tril(arr.astype(np.int64))
    out.flags.writeable = False
    return out


def _matrix_edges(m: NDArray[np.int64], trees: int | None = None) -> list[list[_Edge]]:
    """The edges of the first ``trees`` trees (all by default), tree by tree, by column."""
    d = m.shape[0]
    depth = d - 1 if trees is None else min(trees, d - 1)
    out: list[list[_Edge]] = []
    for t in range(depth):
        row = d - 1 - t
        out.append(
            [
                _Edge(t, i, int(m[row, i]), int(m[i, i]), tuple(int(v) for v in m[row + 1 :, i]))
                for i in range(d - 1 - t)
            ]
        )
    return out


def _check_matrix(m: NDArray[np.int64], proximity_trees: int | None = None) -> None:
    """Raise ``ValueError`` unless ``m`` is an R-vine matrix (0-based labels).

    Checks that the diagonal is a permutation of ``0..d-1``; that every column
    below the diagonal holds exactly the variables further down the diagonal
    (so that each variable is conditioned only on ones sampled before it); and,
    for the first ``proximity_trees`` trees (all by default), that tree 1 is a
    spanning tree and every later edge joins two edges of the tree before that
    share a node -- the proximity condition.
    """
    d = m.shape[0]
    diag = [int(v) for v in np.diagonal(m)]
    if sorted(diag) != list(range(d)):
        raise ValueError(
            f"the diagonal of an R-vine matrix must be a permutation of 0..{d - 1}, got {diag}"
        )
    for i in range(d - 1):
        column = [int(v) for v in m[i:, i]]
        if sorted(column) != sorted(diag[i:]):
            raise ValueError(
                f"column {i} of the R-vine matrix must hold the variables {sorted(diag[i:])} "
                f"(those on the diagonal from row {i} down), got {column}"
            )

    edges = _matrix_edges(m, proximity_trees)
    # Tree 1: d - 1 edges on d nodes with no cycle is a spanning tree.
    parent = list(range(d))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in edges[0] if edges else []:
        ra, rb = find(edge.first), find(edge.second)
        if ra == rb:
            raise ValueError(
                f"tree 1 of the R-vine matrix has a cycle through {edge.first}, {edge.second}"
            )
        parent[ra] = rb

    previous = (
        {frozenset((e.first, e.second, *e.conditioning)): k for k, e in enumerate(edges[0])}
        if edges
        else {}
    )
    for t in range(1, len(edges)):
        links = list(range(len(previous)))

        def root(x: int, links: list[int] = links) -> int:
            while links[x] != x:
                links[x] = links[links[x]]
                x = links[x]
            return x

        current: dict[frozenset[int], int] = {}
        for k, edge in enumerate(edges[t]):
            rest = set(edge.conditioning)
            left = previous.get(frozenset(rest | {edge.first}))
            right = previous.get(frozenset(rest | {edge.second}))
            if left is None or right is None:
                raise ValueError(
                    f"edge {edge.first},{edge.second}|{','.join(map(str, edge.conditioning))} "
                    f"of tree {t + 1} does not join two edges of tree {t} that share a node: "
                    "the matrix violates the proximity condition"
                )
            ra, rb = root(left), root(right)
            if ra == rb:
                raise ValueError(f"tree {t + 1} of the R-vine matrix has a cycle")
            links[ra] = rb
            current[frozenset((edge.first, edge.second, *edge.conditioning))] = k
        previous = current


def _c_matrix(order: Sequence[int]) -> NDArray[np.int64]:
    """The R-vine matrix of a C-vine with the given order (root first)."""
    d = len(order)
    m = np.zeros((d, d), dtype=np.int64)
    for i in range(d):
        m[i, i] = order[d - 1 - i]
        for r in range(i + 1, d):
            m[r, i] = order[d - 1 - r]
    return m


def _d_matrix(order: Sequence[int]) -> NDArray[np.int64]:
    """The R-vine matrix of a D-vine with the given path order."""
    d = len(order)
    m = np.zeros((d, d), dtype=np.int64)
    for i in range(d):
        m[i, i] = order[d - 1 - i]
        for r in range(i + 1, d):
            m[r, i] = order[r - i - 1]
    return m


class _Conditionals:
    """Conditional distribution values keyed by ``(variable, conditioning set)``.

    Values can be *deferred*: stored as the h-function call that would produce
    them and evaluated only if something asks. An R-vine sampler or Rosenblatt
    transform needs only some of the two conditionals each edge could produce,
    and this skips the rest without having to know in advance which they are.
    """

    def __init__(self) -> None:
        self._values: dict[tuple[int, frozenset[int]], NDArray[np.float64]] = {}
        self._pending: dict[tuple[int, frozenset[int]], tuple[Any, ...]] = {}

    def set(self, key: tuple[int, frozenset[int]], value: NDArray[np.float64]) -> None:
        self._values[key] = value
        self._pending.pop(key, None)

    def defer(
        self,
        key: tuple[int, frozenset[int]],
        copula: Copula,
        first: NDArray,
        second: NDArray,
        given: int,
    ) -> None:
        if key not in self._values:
            self._pending[key] = (copula, first, second, given)

    def get(self, key: tuple[int, frozenset[int]]) -> NDArray[np.float64]:
        value = self._values.get(key)
        if value is None:
            copula, first, second, given = self._pending.pop(key)
            value = _h(copula, first, second, given)
            self._values[key] = value
        return value


class VineCopula(Copula):
    """A many-variable copula built by chaining together two-variable copulas.

    Technically a *pair-copula construction* (vine copula). Each pair of
    variables -- and, in later "trees", each pair conditional on the variables
    between them -- gets its own bivariate copula, which may be a different
    family with a different parameter. Use it when one family cannot describe
    every pair: for example, crash-together (Clayton) dependence between two
    assets but symmetric (Frank) dependence elsewhere. To estimate one from
    data, use :func:`fit_vine` rather than building it by hand.

    Parameters
    ----------
    pair_copulas : sequence of sequence of Copula
        The bivariate copulas, grouped by tree. ``pair_copulas[k]`` holds tree
        ``k``'s copulas, so for ``d`` variables there are ``d - 1`` trees and
        tree ``k`` has ``d - 1 - k`` entries (``d(d-1)/2`` copulas in all).
        Every entry must have ``dim == 2``. The dimension ``d`` is inferred as
        ``len(pair_copulas) + 1``. For an R-vine, ``pair_copulas[k][i]`` is
        the copula of the edge in column ``i`` of ``matrix`` (see below).
    structure : {"C", "D", "R"}, default "D"
        ``"C"`` (canonical): each tree is a star around one central variable,
        which suits one market factor plus its satellites. ``"D"`` (drawable):
        each tree is a path, which suits naturally ordered variables such as
        maturities. ``"R"`` (regular): any trees allowed by the proximity
        condition, given by ``matrix``.
    order : sequence of int or None, default None
        A permutation of ``0, 1, ..., d-1`` saying which data column takes
        which position in the structure. For a C-vine the first entry is the
        root of tree 1; for a D-vine the sequence is the path. ``None`` means
        ``0, 1, ..., d-1``. Must be ``None`` for an R-vine, whose matrix
        already says where every variable sits.
    matrix : array_like of int, shape (d, d), or None, default None, keyword-only
        The R-vine matrix, required when ``structure="R"`` and refused
        otherwise. It follows R ``VineCopula``'s ``RVineMatrix`` convention
        with **0-based** variable labels (subtract 1 from an R matrix): a
        lower-triangular matrix ``M`` whose diagonal is a permutation of the
        variables; column ``i``, row ``r > i`` is the edge joining ``M[r, i]``
        and ``M[i, i]`` given ``M[r+1:, i]``, in tree ``d - r`` (row ``d - 1``
        is tree 1). Its pair-copula is ``pair_copulas[d - 1 - r][i]``,
        evaluated at ``(u_{M[r,i] | given}, u_{M[i,i] | given})`` -- the
        off-diagonal variable first, as in ``VineCopula``. Entries above the
        diagonal are ignored.
    free : array_like of bool or None, default None, keyword-only
        Free/fixed mask passed to :class:`~rcopula.core.base.Copula`. A vine
        has no top-level parameters of its own (its parameters live in the
        pair-copulas), so this is accepted for interface compatibility and
        normally left as ``None``.

    Attributes
    ----------
    pair_copulas : list of list of Copula
        The bivariate copulas by tree, as given.
    structure : str
        ``"C"``, ``"D"`` or ``"R"``.
    order : tuple of int
        The variable ordering, length ``d``: the order in which :meth:`rvs`
        draws the variables and :meth:`rosenblatt` conditions them. For an
        R-vine it is the matrix diagonal read from the bottom up.
    matrix : numpy.ndarray of int, shape (d, d)
        The R-vine matrix (0-based labels). Computed for a C- or D-vine, so
        any vine can be passed to R or rebuilt as ``structure="R"``.
    dim : int
        Number of variables ``d``.
    n_pairs : int
        Number of bivariate copulas, ``d(d-1)/2``.
    is_gaussian : bool
        Whether every pair-copula is Gaussian.
    truncation_level : int
        Number of leading trees that carry dependence; every tree after it is
        all independence copulas.

    Raises
    ------
    ValueError
        If ``structure`` is not ``"C"``, ``"D"`` or ``"R"``, if a tree has the
        wrong number of copulas for the dimension, if any pair-copula is not
        bivariate, if ``order`` is not a permutation of ``0..d-1``, if
        ``matrix`` is missing for an R-vine (or given for a C- or D-vine), or
        if ``matrix`` is not a valid R-vine matrix.

    Notes
    -----
    The density, the sampler (:meth:`rvs`) and the Rosenblatt transform are
    exact. All three stop at :attr:`truncation_level`: trees made entirely of
    independence copulas after it are skipped, so a vine truncated after one
    tree samples in time linear in ``d`` (an 801-variable C-vine with one
    Student-t tree draws 50,000 rows in well under a minute).

    The sampler inverts each pair-copula's h-function in closed form where an
    exact inverse exists (Gaussian, Student t, Clayton, Frank and their
    rotations; BB1 and BB7 by a safeguarded Newton solve) and by bisection
    otherwise (Gumbel, Joe, ...). rcopula 0.3.0 used bisection everywhere on a
    full vine, so seeded draws from such a vine differ from 0.3.0 in about the
    eighth decimal place while being several times faster.

    An R-vine matrix is checked for the proximity condition in every tree that
    carries dependence; trees past :attr:`truncation_level` hold only
    independence copulas, do not affect the distribution, and are checked only
    for the column condition (which keeps a 500-variable truncated R-vine
    cheap to build).

    There is no closed-form distribution function, so :meth:`cdf` raises
    :class:`NotImplementedError`; neither are there single-number
    ``tau``/``rho``/``lambda_`` summaries, since every pair has its own.

    Examples
    --------
    A three-dimensional D-vine mixing three families -- something no single
    parametric copula can express:

    >>> import rcopula as rc
    >>> from rcopula.vine import VineCopula
    >>> vine = VineCopula(
    ...     [[rc.ClaytonCopula(2.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(3.0)]],
    ...     structure="D",
    ... )
    >>> vine.dim
    3
    >>> u = vine.rvs(2000, random_state=0)
    >>> u.shape
    (2000, 3)

    Tree 1 pairs adjacent variables, so their dependence is the pair-copula's:

    >>> from scipy import stats
    >>> tau = stats.kendalltau(u[:, 0], u[:, 1]).statistic
    >>> bool(abs(tau - rc.ClaytonCopula(2.0).tau()) < 0.03)
    True

    An all-Gaussian vine **is** a Gaussian copula, which makes it checkable
    against one exactly:

    >>> gaussian = VineCopula(
    ...     [[rc.GaussianCopula(0.6), rc.GaussianCopula(0.5)], [rc.GaussianCopula(0.3)]],
    ...     structure="D",
    ... )
    >>> equivalent = gaussian.to_gaussian()
    >>> import numpy as np
    >>> pts = np.array([[0.3, 0.5, 0.7], [0.8, 0.2, 0.4]])
    >>> bool(np.allclose(gaussian.logpdf(pts), equivalent.logpdf(pts), atol=1e-9))
    True

    A regular vine on five variables whose first tree is neither a star nor a
    path: variable 0 is joined to 1, 2 and 3, and variable 4 hangs off 3.
    Column ``i`` of the matrix lists, from the bottom up, the variables its
    diagonal entry is paired with in trees 1, 2, ... (this is the example
    matrix of R ``VineCopula``'s ``RVineMatrix`` help page, minus 1):

    >>> m = np.array(
    ...     [[4, 0, 0, 0, 0], [1, 1, 0, 0, 0], [2, 2, 2, 0, 0], [0, 3, 3, 3, 0], [3, 0, 0, 0, 0]]
    ... )
    >>> rvine = VineCopula(
    ...     [
    ...         [
    ...             rc.ClaytonCopula(2.0),
    ...             rc.GumbelCopula(1.5),
    ...             rc.FrankCopula(4.0),
    ...             rc.GaussianCopula(0.5),
    ...         ],
    ...         [
    ...             rc.GaussianCopula(0.3),
    ...             rc.RotatedCopula(rc.ClaytonCopula(1.0), 90),
    ...             rc.FrankCopula(-2.0),
    ...         ],
    ...         [rc.StudentCopula(0.2, df=6.0), rc.GumbelCopula(1.2)],
    ...         [rc.IndependenceCopula(2)],
    ...     ],
    ...     structure="R",
    ...     matrix=m,
    ... )
    >>> print(rvine.describe())
    R-vine copula, dim 5, order [0, 3, 2, 1, 4]
      tree 1  3,4            Clayton copula, dim 2, theta=2
      tree 1  0,1            Gumbel copula, dim 2, theta=1.5
      tree 1  0,2            Frank copula, dim 2, theta=4
      tree 1  0,3            Gaussian copula, dim 2, rho.1=0.5
      tree 2  0,4|3          Gaussian copula, dim 2, rho.1=0.3
      tree 2  3,1|0          90-degree rotated Clayton copula, dim 2, theta=1
      tree 2  3,2|0          Frank copula, dim 2, theta=-2
      tree 3  2,4|0,3        t copula, dim 2, rho.1=0.2, df=6
      tree 3  2,1|3,0        Gumbel copula, dim 2, theta=1.2
      tree 4  1,4|2,0,3      Independence copula, dim 2
    >>> rvine.truncation_level
    3
    >>> u = rvine.rvs(4000, random_state=0)
    >>> w = rvine.rosenblatt(u)
    >>> bool(np.all(np.abs(np.corrcoef(w, rowvar=False) - np.eye(5)) < 0.06))
    True
    """

    name = "Vine"
    param_names: tuple[str, ...] = ()

    def __init__(
        self,
        pair_copulas: Sequence[Sequence[Copula]],
        structure: str = "D",
        order: Sequence[int] | None = None,
        *,
        matrix: ArrayLike | None = None,
        free: ArrayLike | None = None,
    ) -> None:
        if structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {STRUCTURES}, got {structure!r}")
        if structure == "R" and matrix is None:
            raise ValueError("structure='R' needs the R-vine matrix: pass matrix=...")
        if structure != "R" and matrix is not None:
            raise ValueError(
                f"matrix= describes an R-vine; a {structure}-vine is set by its order. "
                "Pass structure='R' to use the matrix."
            )
        if structure == "R" and order is not None:
            raise ValueError("an R-vine's order is read from its matrix; leave order=None")
        trees = [list(level) for level in pair_copulas]
        dim = len(trees) + 1
        for k, level in enumerate(trees):
            if len(level) != dim - 1 - k:
                raise ValueError(
                    f"tree {k} needs {dim - 1 - k} pair-copulas for dim={dim}, got {len(level)}"
                )
            for cop in level:
                if cop.dim != 2:
                    raise ValueError(f"pair-copulas must be bivariate, got dim={cop.dim}")

        self.pair_copulas = trees
        self.structure = structure
        self._matrix: NDArray[np.int64] | None = None
        if structure == "R":
            assert matrix is not None
            m = _normalise_matrix(matrix)
            if m.shape[0] != dim:
                raise ValueError(
                    f"the R-vine matrix is {m.shape[0]}x{m.shape[0]} but the pair-copulas "
                    f"describe a {dim}-dimensional vine"
                )
            _check_matrix(m, proximity_trees=max(self.truncation_level, 1))
            self._matrix = m
            self.order = tuple(int(m[dim - 1 - s, dim - 1 - s]) for s in range(dim))
        else:
            self.order = tuple(range(dim)) if order is None else tuple(int(j) for j in order)
        if sorted(self.order) != list(range(dim)):
            raise ValueError(f"order must be a permutation of 0..{dim - 1}, got {self.order}")

        super().__init__(np.empty(0), dim, free=free)

    # -- plumbing ------------------------------------------------------

    @property
    def n_pairs(self) -> int:
        """How many bivariate copulas the vine contains.

        Returns
        -------
        int
            ``d(d-1)/2`` for a ``d``-dimensional vine.
        """
        return self.dim * (self.dim - 1) // 2

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        """Bounds on the vine's own top-level parameters: always empty.

        A vine's parameters belong to its pair-copulas, so there is nothing at
        this level to bound.

        Returns
        -------
        list of tuple of (float, float)
            Always ``[]``.
        """
        return []

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> VineCopula:
        if self.structure == "R":
            return VineCopula(self.pair_copulas, "R", matrix=self._matrix)
        return VineCopula(self.pair_copulas, self.structure, self.order)

    def _reorder(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return u[:, list(self.order)]

    # -- R-vine view -----------------------------------------------------

    @property
    def matrix(self) -> NDArray[np.int64]:
        """The vine's R-vine matrix, with 0-based variable labels.

        For an R-vine this is the matrix it was built from (upper triangle
        zeroed); for a C- or D-vine it is computed, so every vine has one. Add
        1 to every entry on and below the diagonal to get R ``VineCopula``'s
        ``RVineMatrix``. The edge in column ``i``, row ``r`` has pair-copula
        ``pair_copulas[d - 1 - r][i]`` -- for a C- or D-vine that is the
        tree's copula list *reversed* (see :meth:`to_rvine`).

        Returns
        -------
        numpy.ndarray of int, shape (d, d)
            Lower-triangular, read-only.

        Examples
        --------
        >>> import rcopula as rc
        >>> from rcopula.vine import VineCopula
        >>> vine = VineCopula(
        ...     [[rc.ClaytonCopula(2.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(3.0)]],
        ...     structure="D",
        ... )
        >>> vine.matrix.tolist()
        [[2, 0, 0], [0, 1, 0], [1, 0, 0]]
        """
        if self._matrix is not None:
            return self._matrix
        build = _c_matrix if self.structure == "C" else _d_matrix
        m = build(self.order)
        m.flags.writeable = False
        return m

    def to_rvine(self) -> VineCopula:
        """Re-express the vine as an R-vine (``structure="R"``) with the same distribution.

        A C- or D-vine is a special case of a regular vine, so this only
        rewrites the bookkeeping: the result has :attr:`matrix` as its matrix
        and each tree's pair-copulas in column order. Its density is the same
        function; its sampler draws the variables in the same :attr:`order`.
        An R-vine is returned unchanged.

        Returns
        -------
        VineCopula
            A vine with ``structure == "R"``.

        Examples
        --------
        >>> import numpy as np
        >>> import rcopula as rc
        >>> from rcopula.vine import VineCopula
        >>> cvine = VineCopula(
        ...     [[rc.ClaytonCopula(2.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(3.0)]],
        ...     structure="C",
        ... )
        >>> rvine = cvine.to_rvine()
        >>> rvine.structure
        'R'
        >>> pts = np.array([[0.3, 0.5, 0.7], [0.8, 0.2, 0.4]])
        >>> bool(np.allclose(cvine.logpdf(pts), rvine.logpdf(pts), atol=1e-12))
        True
        """
        if self.structure == "R":
            return self
        # In both C- and D-vine layouts, tree k edge e sits in matrix column
        # d - 2 - k - e, and both put the off-diagonal variable first, as the
        # matrix convention does -- so a reversal is all it takes.
        trees = [list(reversed(level)) for level in self.pair_copulas]
        return VineCopula(trees, "R", matrix=self.matrix)

    @property
    def edges(self) -> list[tuple[int, int, int, tuple[int, ...], Copula]]:
        """Every edge of the vine as ``(tree, first, second, conditioning, copula)``.

        ``tree`` is 0-based (it indexes :attr:`pair_copulas`); ``first`` and
        ``second`` are the conditioned variables in the order the copula takes
        them, and ``conditioning`` the variables they are conditioned on, all
        in original column indices.

        Returns
        -------
        list of tuple of (int, int, int, tuple of int, Copula)
            ``d(d-1)/2`` entries, tree by tree.

        Examples
        --------
        >>> import rcopula as rc
        >>> from rcopula.vine import VineCopula
        >>> vine = VineCopula(
        ...     [[rc.ClaytonCopula(2.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(3.0)]],
        ...     structure="D",
        ... )
        >>> [(t, a, b, given) for t, a, b, given, _ in vine.edges]
        [(0, 0, 1, ()), (0, 1, 2, ()), (1, 0, 2, (1,))]
        """
        out = []
        for k, level in enumerate(self.pair_copulas):
            for i, cop in enumerate(level):
                a, b, conditioning = self._edge_indices(k, i)
                out.append(
                    (
                        k,
                        self.order[a],
                        self.order[b],
                        tuple(self.order[c] for c in conditioning),
                        cop,
                    )
                )
        return out

    def _rvine_edges(self, depth: int) -> list[list[tuple[_Edge, Copula]]]:
        """The first ``depth`` trees of an R-vine, each edge with its copula."""
        assert self._matrix is not None
        return [
            [(edge, self.pair_copulas[t][edge.column]) for edge in level]
            for t, level in enumerate(_matrix_edges(self._matrix, depth))
        ]

    def _rvine_forward(self, u: NDArray[np.float64], density: bool):
        """One pass up the trees of an R-vine: the log-density and the conditionals.

        Every edge evaluates its copula at the two conditionals it joins and
        *defers* the two conditionals it produces, so only those some later
        edge (or the Rosenblatt read-out) asks for are ever computed.
        """
        depth = self.truncation_level
        store = _Conditionals()
        empty: frozenset[int] = frozenset()
        for j in range(self.dim):
            store.set((j, empty), u[:, j])
        total = np.zeros(u.shape[0])
        for level in self._rvine_edges(depth):
            for edge, copula in level:
                given = frozenset(edge.conditioning)
                a = store.get((edge.first, given))
                b = store.get((edge.second, given))
                up_a, up_b = (
                    (edge.first, given | {edge.second}),
                    (edge.second, given | {edge.first}),
                )
                if _is_independence(copula):
                    store.set(up_a, a)
                    store.set(up_b, b)
                    continue
                if density:
                    total = total + copula.logpdf(np.column_stack([a, b]))
                store.defer(up_a, copula, a, b, given=1)
                store.defer(up_b, copula, a, b, given=0)
        return total, store

    def _rvine_rosenblatt(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        assert self._matrix is not None
        m, d = self._matrix, self.dim
        depth = self.truncation_level
        _, store = self._rvine_forward(u, density=False)
        out = np.empty_like(u)
        for i in range(d):
            levels = min(d - 1 - i, depth)
            given = frozenset(int(v) for v in m[d - levels :, i]) if levels else frozenset()
            out[:, int(m[i, i])] = store.get((int(m[i, i]), given))
        return out

    def _simulate_r_vine(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        """Inverse Rosenblatt for an R-vine, one matrix column at a time.

        Variables are drawn bottom-up along the diagonal (the :attr:`order`).
        Column ``i``'s diagonal variable ``x`` is unwound from its deepest tree
        down to tree 1 through the inverse h-functions; on the way the
        conditionals of ``x`` are recorded, and those of its partners given
        ``x`` are deferred for the columns still to come.
        """
        assert self._matrix is not None
        m, d = self._matrix, self.dim
        depth = self.truncation_level
        trees = self._rvine_edges(depth)
        store = _Conditionals()
        cache: dict[Any, Any] = {}
        empty: frozenset[int] = frozenset()
        x_out = np.empty_like(w)
        for step, i in enumerate(range(d - 1, -1, -1)):
            x = int(m[i, i])
            levels = min(d - 1 - i, depth)
            column = [trees[t][i] for t in range(levels)]
            chain: list[NDArray[np.float64]] = [w[:, step]] * (levels + 1)
            value = w[:, step]
            for t in range(levels - 1, -1, -1):
                edge, copula = column[t]
                if not _is_independence(copula):
                    other = store.get((edge.first, frozenset(edge.conditioning)))
                    value = _h_inverse(copula, value, other, side=0, cache=cache)
                chain[t] = value
            x_out[:, x] = value
            store.set((x, empty), value)
            for t in range(levels):
                edge, copula = column[t]
                given = frozenset(edge.conditioning)
                store.set((x, given), chain[t])
                store.set((x, given | {edge.first}), chain[t + 1])
                other = store.get((edge.first, given))
                if _is_independence(copula):
                    store.set((edge.first, given | {x}), other)
                else:
                    store.defer((edge.first, given | {x}), copula, other, chain[t], given=1)
        return x_out

    # -- density -------------------------------------------------------

    def _logpdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.structure == "R":
            return self._rvine_forward(u, density=True)[0]
        arranged = self._reorder(u)
        total = np.zeros(arranged.shape[0])
        for copula, first, second in self._edges(arranged):
            if _is_independence(copula):
                continue  # log-density identically zero
            total = total + copula.logpdf(np.column_stack([first, second]))
        return total

    @property
    def truncation_level(self) -> int:
        """How many trees carry dependence: the last tree with a non-independence pair-copula.

        A vine is *truncated* at level ``t`` when every pair-copula in trees
        ``t + 1, ..., d - 1`` is the independence copula, as :func:`fit_vine`
        produces with ``truncate=t``. Those trees contribute nothing to the
        density and their conditional transforms are the identity, so the
        density, :meth:`rvs` and :meth:`rosenblatt` stop after tree ``t``.

        Returns
        -------
        int
            Between 0 (every pair-copula is independence) and ``d - 1`` (the
            last tree carries dependence, i.e. the vine is not truncated).
        """
        for k in range(len(self.pair_copulas) - 1, -1, -1):
            if not all(_is_independence(cop) for cop in self.pair_copulas[k]):
                return k + 1
        return 0

    def _edges(self, u: NDArray[np.float64]):
        """Yield ``(copula, first argument, second argument)`` for every edge.

        Walking the trees once and handing back the arguments keeps the density,
        the log-likelihood and the Rosenblatt transform on a single traversal,
        so there is one place for the recursion to be right or wrong.
        """
        # Trees past the truncation level hold only independence copulas: they
        # add nothing to the density, so they are not walked at all.
        depth = self.truncation_level
        if self.structure == "C":
            level = [u[:, j] for j in range(self.dim)]
            for k, copulas in enumerate(self.pair_copulas[:depth]):
                root = level[0]
                for i, copula in enumerate(copulas):
                    yield copula, root, level[i + 1]
                if k < depth - 1:
                    level = [
                        _h(copula, level[i + 1], root, given=1) for i, copula in enumerate(copulas)
                    ]
            return

        # D-vine: tree k edge i joins variables i and i+k+1 given i+1..i+k, and
        # its two arguments are the corresponding conditionals.
        left = [u[:, j] for j in range(self.dim - 1)]
        right = [u[:, j + 1] for j in range(self.dim - 1)]
        for k, copulas in enumerate(self.pair_copulas[:depth]):
            for i, copula in enumerate(copulas):
                yield copula, left[i], right[i]
            if k < depth - 1:
                new_left = [
                    _h(copulas[i], left[i], right[i], given=1) for i in range(len(copulas) - 1)
                ]
                new_right = [
                    _h(copulas[i + 1], left[i + 1], right[i + 1], given=0)
                    for i in range(len(copulas) - 1)
                ]
                left, right = new_left, new_right

    def _cdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError(
            "a vine copula has no closed-form distribution function -- the "
            "construction factorises the *density*. Integrate it, or estimate "
            "the CDF from rvs(); the density, sampler and Rosenblatt transform "
            "are all exact."
        )

    def loglik(self, data: ArrayLike) -> float:
        """Sum the log-density over all rows: how well the vine explains the data.

        Higher is better. Useful for comparing fitted vines on the same data,
        or for building an AIC/BIC by hand.

        Parameters
        ----------
        data : array_like of float, shape (n, d)
            Observations. If every value already lies strictly inside
            ``(0, 1)`` they are used as-is (treated as copula-scale data);
            otherwise they are converted to pseudo-observations (ranks scaled
            into ``(0, 1)``) first.

        Returns
        -------
        float
            The total log-likelihood, ``sum(log c(u_i))``.
        """
        u = np.atleast_2d(np.asarray(data, dtype=np.float64))
        if not np.all((u > 0.0) & (u < 1.0)):
            u = pseudo_obs(u)
        return float(np.sum(self.logpdf(u)))

    # -- sampling ------------------------------------------------------

    def _rvs(
        self, size: int, params: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Inverse Rosenblatt: draw independent uniforms and unwind the trees.

        Only the first :attr:`truncation_level` trees are unwound: past it every
        pair-copula is the independence copula, whose inverse h-function is the
        identity, so a vine truncated after ``t`` trees costs ``O(t d)``
        h-function evaluations rather than ``O(d^2)``. Every pair-copula with
        an exact inverse h-function is inverted in closed form (see
        :func:`_h_inverse`); uniform column ``i`` drives variable ``order[i]``.
        """
        w = rng.uniform(size=(size, self.dim))
        if self.structure == "R":
            return np.clip(self._simulate_r_vine(w), np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        depth = self.truncation_level
        arranged = (
            self._simulate_c_vine(w, depth, closed_form=True)
            if self.structure == "C"
            else self._simulate_d_vine(w, depth, closed_form=True)
        )
        out = np.empty_like(arranged)
        out[:, list(self.order)] = arranged
        return np.clip(out, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))

    def _simulate_c_vine(
        self, w: NDArray[np.float64], depth: int | None = None, closed_form: bool = True
    ) -> NDArray[np.float64]:
        """Aas et al. (2009), Algorithm 3, stopping at tree ``depth``.

        ``v[i][j]`` is variable ``i`` conditioned on the first ``j`` roots. Only
        ``j < depth`` is ever needed: deeper trees are independence and their
        transforms the identity. With ``depth = d - 1`` this is the textbook
        algorithm, operation for operation.
        """
        d = self.dim
        depth = d - 1 if depth is None else depth
        v: list[list[NDArray[np.float64]]] = [[np.empty(0)] * (depth + 1) for _ in range(d)]
        cache: dict[Any, NDArray[np.float64]] = {}
        x = np.empty_like(w)
        x[:, 0] = v[0][0] = w[:, 0]

        for i in range(1, d):
            value = w[:, i]
            for k in range(min(i - 1, depth - 1), -1, -1):
                value = _h_inverse(
                    self.pair_copulas[k][i - k - 1],
                    value,
                    v[k][k],
                    side=1,
                    closed_form=closed_form,
                    cache=cache,
                )
            x[:, i] = v[i][0] = value
            if i == d - 1:
                break
            for j in range(min(i, depth - 1)):
                v[i][j + 1] = _h(self.pair_copulas[j][i - j - 1], v[i][j], v[j][j], given=1)
        return x

    def _simulate_d_vine(
        self, w: NDArray[np.float64], depth: int | None = None, closed_form: bool = True
    ) -> NDArray[np.float64]:
        r"""Inverse Rosenblatt for a D-vine, stopping at tree ``depth``.

        Variable :math:`i` is drawn from its conditional given the ones before
        it, which unwinds through the trees: apply the tree-:math:`k` inverse
        h-function, then rebuild the conditionals the next variable will need.
        Trees from ``depth`` on are independence, so they are skipped; with
        ``depth = d - 1`` nothing is.
        """
        d = w.shape[1]
        depth = d - 1 if depth is None else depth
        x = np.empty_like(w)
        x[:, 0] = w[:, 0]
        # left[k] holds the tree-k conditional the next variable needs. The
        # matching "right" conditionals are consumed within a step and never
        # carried across one, so only `left` persists.
        left: list[NDArray[np.float64]] = []

        for i in range(1, d):
            value = w[:, i]
            if depth == 0:
                x[:, i] = value
                continue
            # Unwind from the deepest tree that reaches this variable.
            for k in range(min(i - 1, depth - 1), 0, -1):
                value = _h_inverse(
                    self.pair_copulas[k][i - k - 1],
                    value,
                    left[k - 1],
                    side=0,
                    closed_form=closed_form,
                )
            value = _h_inverse(
                self.pair_copulas[0][i - 1], value, x[:, i - 1], side=0, closed_form=closed_form
            )
            x[:, i] = value
            if depth == 1 or i == d - 1:
                continue  # no deeper tree will ask for a conditional

            # Rebuild the conditionals for the next variable.
            new_left = [_h(self.pair_copulas[0][i - 1], x[:, i - 1], x[:, i], given=1)]
            new_right = [_h(self.pair_copulas[0][i - 1], x[:, i - 1], x[:, i], given=0)]
            for k in range(1, min(i, depth - 1)):
                new_left.append(
                    _h(self.pair_copulas[k][i - k - 1], left[k - 1], new_right[k - 1], given=1)
                )
                new_right.append(
                    _h(self.pair_copulas[k][i - k - 1], left[k - 1], new_right[k - 1], given=0)
                )
            left = new_left
        return x

    def rosenblatt(self, u: ArrayLike) -> NDArray[np.float64]:
        r"""Turn dependent copula data into independent uniform columns.

        This is the Rosenblatt transform: each column is replaced by its
        conditional probability given the variables before it in the vine's
        :attr:`order` (the D-vine path, the C-vine roots, or the R-vine
        matrix diagonal read bottom-up). If the vine is the right model for the data, the
        output columns are independent and uniform on ``(0, 1)``, so it is the
        standard way to check a fitted vine (test the output for independence)
        and the inverse of how :meth:`rvs` samples.

        Parameters
        ----------
        u : array_like of float, shape (n, d)
            Copula-scale data (values in ``(0, 1)``), with columns in the
            original variable order; the vine's ``order`` is applied
            internally.

        Returns
        -------
        numpy.ndarray of float, shape (n, d)
            The transformed values, in the *original* column order -- the same
            order as the input, :meth:`rvs` and :meth:`logpdf`. Column
            ``order[i]`` holds
            :math:`P(U_{order[i]} \le u_{order[i]} \mid U_{order[0]}, \dots,
            U_{order[i-1]})`, so column ``order[0]`` is passed through unchanged
            and column ``order[-1]`` is conditioned on every other variable.
            (rcopula 0.2.0 and earlier returned the columns in path order; with the
            default ``order`` the two agree.)

        Raises
        ------
        ValueError
            If ``u`` does not have ``d`` columns.

        Notes
        -----
        The forward direction of the sampler, and the sharpest available check
        on both: under the true vine the output is independent
        :math:`\mathrm{Unif}(0,1)`, so any error in either recursion shows up as
        dependence that should not be there.

        A C-vine or R-vine is transformed through its R-vine form (see
        :meth:`to_rvine`); a D-vine uses its own recursion. (rcopula 0.3.0 and
        earlier raised ``NotImplementedError`` for a C-vine.)

        Examples
        --------
        >>> import numpy as np
        >>> import rcopula as rc
        >>> from rcopula.vine import VineCopula
        >>> vine = VineCopula(
        ...     [[rc.ClaytonCopula(2.0), rc.FrankCopula(4.0)], [rc.GumbelCopula(1.5)]],
        ...     structure="D",
        ... )
        >>> w = vine.rosenblatt(vine.rvs(4000, random_state=1))
        >>> w.shape
        (4000, 3)
        >>> bool(np.all(np.abs(np.corrcoef(w, rowvar=False) - np.eye(3)) < 0.06))
        True
        """
        arranged = self._reorder(np.atleast_2d(np.asarray(u, dtype=np.float64)))
        if arranged.shape[1] != self.dim:
            raise ValueError(f"u has {arranged.shape[1]} columns, expected {self.dim}")
        if self.structure != "D":
            return self.to_rvine()._rvine_rosenblatt(np.atleast_2d(np.asarray(u, dtype=np.float64)))

        # Trees past the truncation level are independence: their h-functions
        # are the identity, so they are skipped.
        depth = self.truncation_level
        out = np.empty_like(arranged)
        out[:, 0] = arranged[:, 0]
        left: list[NDArray[np.float64]] = []
        for i in range(1, self.dim):
            if depth == 0:
                out[:, i] = arranged[:, i]
                continue
            value = _h(self.pair_copulas[0][i - 1], arranged[:, i - 1], arranged[:, i], given=0)
            for k in range(1, min(i, depth)):
                value = _h(self.pair_copulas[k][i - k - 1], left[k - 1], value, given=0)
            out[:, i] = value
            if depth == 1 or i == self.dim - 1:
                continue  # no deeper tree will ask for a conditional

            new_left = [
                _h(self.pair_copulas[0][i - 1], arranged[:, i - 1], arranged[:, i], given=1)
            ]
            new_right = [
                _h(self.pair_copulas[0][i - 1], arranged[:, i - 1], arranged[:, i], given=0)
            ]
            for k in range(1, min(i, depth - 1)):
                new_left.append(
                    _h(self.pair_copulas[k][i - k - 1], left[k - 1], new_right[k - 1], given=1)
                )
                new_right.append(
                    _h(self.pair_copulas[k][i - k - 1], left[k - 1], new_right[k - 1], given=0)
                )
            left = new_left
        # Back to the caller's column order, as rvs and logpdf use.
        result = np.empty_like(out)
        result[:, list(self.order)] = out
        return result

    # -- the Gaussian identity -----------------------------------------

    @property
    def is_gaussian(self) -> bool:
        """Whether every pair-copula is Gaussian, which makes the whole vine Gaussian.

        When ``True``, :meth:`to_gaussian` can return the equivalent
        :class:`~rcopula.core.elliptical.GaussianCopula`.

        Returns
        -------
        bool
        """
        return all(isinstance(cop, GaussianCopula) for level in self.pair_copulas for cop in level)

    def to_gaussian(self) -> GaussianCopula:
        r"""Convert an all-Gaussian vine into the single Gaussian copula it equals.

        Only works when every pair-copula is Gaussian (see :attr:`is_gaussian`).
        Useful for reading off the implied full correlation matrix, or as an
        exact correctness check on a vine.

        Returns
        -------
        GaussianCopula
            A ``d``-dimensional Gaussian copula with an unstructured
            (``dispstr="un"``) correlation matrix, indexed by the original
            variable order.

        Raises
        ------
        ValueError
            If any pair-copula is not Gaussian.

        Notes
        -----
        A vine's tree-:math:`k` parameters are **partial correlations** given the
        conditioning set, and the partial-correlation recursion

        .. math::
            \rho_{ab\mid D} = \frac{\rho_{ab\mid D'} - \rho_{ac\mid D'}\rho_{bc\mid D'}}
                                   {\sqrt{(1-\rho_{ac\mid D'}^2)(1-\rho_{bc\mid D'}^2)}}

        runs backwards to recover the ordinary correlations. So an all-Gaussian
        vine is a Gaussian copula, and its density can be checked against one
        **exactly** rather than statistically -- which is the strongest test
        available on the tree recursions.

        Examples
        --------
        >>> import numpy as np
        >>> import rcopula as rc
        >>> from rcopula.vine import VineCopula
        >>> vine = VineCopula(
        ...     [[rc.GaussianCopula(0.7), rc.GaussianCopula(0.4)], [rc.GaussianCopula(0.2)]],
        ...     structure="C",
        ... )
        >>> sigma = vine.to_gaussian().sigma()
        >>> float(round(sigma[0, 1], 6)), float(round(sigma[0, 2], 6))
        (0.7, 0.4)

        The 1-2 correlation is *implied* rather than given, since tree 2 supplies
        only the partial one:

        >>> float(round(sigma[1, 2], 6))
        0.410905
        """
        if not self.is_gaussian:
            raise ValueError(
                "to_gaussian needs every pair-copula to be Gaussian; this vine "
                "mixes families, which is the reason to build one"
            )
        d = self.dim
        if self.structure == "R":
            return self._rvine_to_gaussian()
        rho = np.eye(d)
        # partial[(a, b)] indexed on the structure's own positions
        for k, level in enumerate(self.pair_copulas):
            for i, cop in enumerate(level):
                a, b, conditioning = self._edge_indices(k, i)
                value = float(cop.params[0])
                # Peel the conditioning set off one at a time, from the end. At
                # step j the value is conditioned on `conditioning[:j+1]`, so the
                # correlations that invert it are conditioned on
                # `conditioning[:j]` -- the elements NOT yet removed. Using
                # "everything except c" instead is identical for a single
                # conditioning variable and wrong for two or more, which is
                # exactly where it first shows.
                for j in range(len(conditioning) - 1, -1, -1):
                    c, rest = conditioning[j], conditioning[:j]
                    ac = _partial(rho, a, c, rest)
                    bc = _partial(rho, b, c, rest)
                    value = value * np.sqrt((1 - ac**2) * (1 - bc**2)) + ac * bc
                rho[a, b] = rho[b, a] = value
        # rho is indexed by structure position; sigma must be indexed by the
        # original variable, so map back through the order.
        positions = np.argsort(self.order)
        sigma = rho[np.ix_(positions, positions)]
        return GaussianCopula(P2p(sigma), dim=d, dispstr="un")

    def _rvine_to_gaussian(self) -> GaussianCopula:
        r"""``to_gaussian`` for an R-vine, by inverting each partial correlation.

        :math:`\rho_{ab\mid D}` is the correlation of :math:`a` and :math:`b`
        after regressing both on :math:`D`, so
        :math:`\rho_{ab} = \rho_{ab\mid D}\,s_a s_b + r_{aD}R_{DD}^{-1}r_{Db}`
        with :math:`s_x^2 = 1 - r_{xD}R_{DD}^{-1}r_{Dx}`. Tree by tree, every
        correlation on the right is already known: the variables an edge
        touches span a sub-vine of the trees below it.
        """
        d = self.dim
        rho = np.eye(d)
        for level in self._rvine_edges(d - 1):
            for edge, copula in level:
                a, b, given = edge.first, edge.second, list(edge.conditioning)
                value = float(copula.params[0])
                if given:
                    r_dd = rho[np.ix_(given, given)]
                    r_a, r_b = rho[a, given], rho[b, given]
                    sol_a, sol_b = np.linalg.solve(r_dd, r_a), np.linalg.solve(r_dd, r_b)
                    scale = np.sqrt(max(1.0 - r_a @ sol_a, 0.0) * max(1.0 - r_b @ sol_b, 0.0))
                    value = value * scale + float(r_a @ sol_b)
                rho[a, b] = rho[b, a] = value
        return GaussianCopula(P2p(rho), dim=d, dispstr="un")

    def _positions(self) -> dict[int, int]:
        """Variable to position in :attr:`order`, cached."""
        cached = self.__dict__.get("_position_cache")
        if cached is None:
            cached = {v: s for s, v in enumerate(self.order)}
            self.__dict__["_position_cache"] = cached
        return cached

    def _edge_indices(self, tree: int, edge: int) -> tuple[int, int, list[int]]:
        """Which positions an edge joins, and what it conditions on.

        Positions index :attr:`order`, so ``order[a]`` is the variable. For an
        R-vine the edge is the one in matrix column ``edge``.
        """
        if self.structure == "R":
            assert self._matrix is not None
            m, d = self._matrix, self.dim
            position = self._positions()
            row = d - 1 - tree
            return (
                position[int(m[row, edge])],
                position[int(m[edge, edge])],
                [position[int(v)] for v in m[row + 1 :, edge]],
            )
        if self.structure == "C":
            return tree, tree + edge + 1, list(range(tree))
        return edge, edge + tree + 1, list(range(edge + 1, edge + tree + 1))

    # -- dependence ----------------------------------------------------

    def tau(self) -> float:
        """Not available for a vine: every pair has its own Kendall's tau.

        Raises
        ------
        NotImplementedError
            Always. Estimate pairwise taus from :meth:`rvs` output, or read
            them from tree 1's pair-copulas.
        """
        raise NotImplementedError(
            "a vine has a different Kendall's tau for every pair -- that is what "
            "it is for. Estimate it pairwise from rvs(), or read tree 1's "
            "pair-copulas directly."
        )

    def rho(self) -> float:
        """Not available for a vine: every pair has its own Spearman's rho.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError("a vine has a different Spearman's rho for every pair")

    def lambda_(self) -> TailDependence:
        """Not available for a vine: tail dependence differs by pair.

        Raises
        ------
        NotImplementedError
            Always. Tree 1's pair-copulas give it for the pairs they join.
        """
        raise NotImplementedError(
            "tail dependence differs by pair in a vine; tree 1's pair-copulas "
            "give it for the pairs they join"
        )

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> Copula:
        """Not available for a vine: one tau cannot pin down ``d(d-1)/2`` parameters.

        Parameters
        ----------
        tau : float
            Ignored.
        dim : int, default 2
            Ignored.
        **kwargs : Any
            Ignored.

        Raises
        ------
        NotImplementedError
            Always. Use :func:`fit_vine` to estimate a vine from data.
        """
        raise NotImplementedError(
            "a vine has d(d-1)/2 parameters; a single tau cannot identify them. Use fit_vine."
        )

    # -- presentation --------------------------------------------------

    def describe(self) -> str:
        """Return a readable, multi-line summary of the vine, one line per pair-copula.

        The first line gives the structure, dimension and order. Each further
        line names the tree, the edge as ``a,b|conditioning`` in original
        variable indices, and that pair-copula's own one-line description.

        Returns
        -------
        str
            ``1 + d(d-1)/2`` lines joined by newlines.

        Examples
        --------
        >>> import rcopula as rc
        >>> from rcopula.vine import VineCopula
        >>> vine = VineCopula(
        ...     [[rc.ClaytonCopula(2.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(3.0)]],
        ...     structure="D",
        ... )
        >>> print(vine.describe().splitlines()[0])
        D-vine copula, dim 3, order [0, 1, 2]
        >>> len(vine.describe().splitlines())
        4
        """
        rows = [f"{self.structure}-vine copula, dim {self.dim}, order {list(self.order)}"]
        for k, level in enumerate(self.pair_copulas):
            for i, cop in enumerate(level):
                a, b, conditioning = self._edge_indices(k, i)
                label = f"{self.order[a]},{self.order[b]}"
                if conditioning:
                    label += "|" + ",".join(str(self.order[c]) for c in conditioning)
                rows.append(f"  tree {k + 1}  {label:<14} {cop.describe()}")
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"<{self.structure}-vine copula, dim {self.dim}, {self.n_pairs} pair-copulas>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VineCopula):
            return NotImplemented
        return (
            self.structure == other.structure
            and self.order == other.order
            and np.array_equal(self.matrix, other.matrix)
            and self.pair_copulas == other.pair_copulas
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.structure,
                self.order,
                self.matrix.tobytes(),
                tuple(tuple(level) for level in self.pair_copulas),
            )
        )


def _partial(rho: NDArray[np.float64], a: int, b: int, conditioning: Sequence[int]) -> float:
    """Partial correlation of ``a`` and ``b`` given ``conditioning``."""
    if not conditioning:
        return float(rho[a, b])
    c, rest = conditioning[-1], conditioning[:-1]
    ab = _partial(rho, a, b, rest)
    ac = _partial(rho, a, c, rest)
    bc = _partial(rho, b, c, rest)
    denominator = np.sqrt((1.0 - ac**2) * (1.0 - bc**2))
    return float((ab - ac * bc) / denominator) if denominator > 0 else 0.0


def fit_vine(
    data: ArrayLike,
    structure: str = "D",
    families: str | Sequence[str] = DEFAULT_FAMILIES,
    order: Sequence[int] | None = None,
    criterion: str = "aic",
    truncate: int | None = None,
) -> VineCopula:
    """Fit a vine copula to data, picking the best family for each pair automatically.

    Give it a table of observations (one column per variable) and it returns a
    :class:`VineCopula` whose every pair-copula family and parameter has been
    chosen from the data. Use it when you want flexible, pair-by-pair
    dependence across three or more variables without choosing families by
    hand. With ``structure="R"`` it also chooses the trees themselves.

    Parameters
    ----------
    data : array_like or pandas.DataFrame of float, shape (n, d)
        Observations, one row per observation and one column per variable,
        with ``d >= 2``. Raw data are fine: they are converted to
        pseudo-observations (ranks scaled into ``(0, 1)``) internally.
    structure : {"C", "D", "R"}, default "D"
        ``"C"`` builds star-shaped trees around a central variable; ``"D"``
        builds path-shaped trees; ``"R"`` lets the data choose any regular
        vine, tree by tree, with Dissmann et al.'s algorithm (see Notes). See
        :class:`VineCopula`.
    families : str or sequence of str, default DEFAULT_FAMILIES
        Candidate families tried on every edge; names as in
        :data:`~rcopula.select.FAMILIES`, or one group name from there (such
        as ``"vine"``). The default is ``("independence", "gaussian",
        "student", "clayton", "gumbel", "frank")``, which spans no tail
        dependence, lower-only, upper-only and both.
        :data:`EXTENDED_FAMILIES` adds Joe, the 90/180/270-degree rotations
        (``"clayton270"``, ``"gumbel90"``, ...) and BB1/BB7 (``"bb1"``,
        ``"bb7_180"``, ...), which can capture negative dependence and
        asymmetric tails.
    order : sequence of int or None, default None
        A permutation of ``0..d-1`` giving the variable ordering of a C- or
        D-vine. ``None`` orders variables by total absolute Kendall's tau with
        the others, strongest first. For a C-vine that puts the most connected
        variable at the root -- which is what makes a C-vine worth choosing;
        for a D-vine the same rule is used for reproducibility. Must be
        ``None`` for an R-vine, whose structure is selected.
    criterion : {"aic", "bic", "loglik", "xv"}, default "aic"
        How each edge's winning family is chosen, passed to
        :func:`~rcopula.select.select_copula`: lowest AIC or BIC, highest
        log-likelihood, or cross-validated likelihood.
    truncate : int or None, default None
        Fit only the first ``truncate`` trees and set every pair-copula in the
        higher trees to independence. ``None`` fits all ``d - 1`` trees.
        Higher trees usually carry little, and truncating is the standard way
        to stop a vine from spending parameters on noise. It also keeps the
        cost linear in ``d``: neither the fit nor the resulting vine's
        density, sampler or Rosenblatt transform touches the independence
        trees (see :attr:`VineCopula.truncation_level`).

    Returns
    -------
    VineCopula
        The fitted vine. Read ``fitted.order`` for the ordering actually used,
        ``fitted.matrix`` for the R-vine matrix and ``fitted.describe()`` for
        the chosen families.

    Raises
    ------
    ValueError
        If ``data`` has fewer than two columns, ``structure`` is not ``"C"``,
        ``"D"`` or ``"R"``, ``order`` is given for an R-vine, or a family name
        or group is unknown.

    Notes
    -----
    Tree by tree, each edge's family is selected by
    :func:`~rcopula.select.select_copula` on the h-transformed data that edge
    actually sees, and the h-functions for the next tree are built from the
    winner. That is Dissmann et al.'s sequential procedure, and it is what makes
    a :math:`d(d-1)/2`-parameter model estimable at all.

    For an R-vine the trees are chosen the same way. Tree 1 is the maximum
    spanning tree of the complete graph on the variables, weighted by absolute
    empirical Kendall's tau; each later tree is the maximum spanning tree,
    weighted by the absolute tau of the h-transformed data, over the pairs of
    previous edges that share a node (the proximity condition). Trees past
    ``truncate`` are completed with any valid structure, since they hold only
    independence copulas.

    Families that can describe only one sign of dependence are tried only on
    edges whose empirical tau has that sign: the 90/270-degree rotations only
    where tau is negative, and the 0/180-degree versions of BB1 and BB7 and
    the 180-degree rotations only where it is positive, as R's ``VineCopula``
    does. (Plain Clayton, Gumbel and Joe are always tried, as before.)

    Examples
    --------
    >>> import rcopula as rc
    >>> from rcopula.vine import VineCopula, fit_vine
    >>> truth = VineCopula(
    ...     [[rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [rc.FrankCopula(4.0)]],
    ...     structure="D",
    ... )
    >>> u = truth.rvs(3000, random_state=0)
    >>> fitted = fit_vine(
    ...     u, structure="D", order=[0, 1, 2], families=["clayton", "gumbel", "frank"]
    ... )
    >>> [cop.name for cop in fitted.pair_copulas[0]]
    ['Clayton', 'Gumbel']
    >>> bool(fitted.loglik(u) > 0)
    True

    Left to itself it reorders the variables by dependence strength, which is
    what makes a C-vine's root the right one -- so read the structure from
    ``fitted.order`` rather than assuming it:

    >>> fitted = fit_vine(u, structure="D", families=["clayton", "gumbel", "frank"])
    >>> len(fitted.order) == 3
    True

    An R-vine finds the trees as well; negative dependence is picked up by a
    rotated family:

    >>> import numpy as np
    >>> v = np.column_stack([u, 1.0 - u[:, 0] ** 0.5 * 0.5 - 0.25 * u[:, 2]])
    >>> rfit = fit_vine(v, structure="R", families=["gaussian", "clayton", "clayton90"])
    >>> rfit.structure, rfit.dim
    ('R', 4)
    """
    from rcopula.core.other import IndependenceCopula

    # pseudo_obs preserves a DataFrame, and everything below indexes positionally
    # with `[:, cols]`, which a DataFrame refuses. Every other entry point in the
    # package accepts a frame, so this one drops to an array rather than making
    # the caller remember which is which.
    u = np.asarray(pseudo_obs(data), dtype=float)
    d = u.shape[1]
    if d < 2:
        raise ValueError(f"a vine needs at least two variables, got {d}")
    if structure not in STRUCTURES:
        raise ValueError(f"structure must be one of {STRUCTURES}, got {structure!r}")
    names = _family_names(families)
    depth = d - 1 if truncate is None else min(int(truncate), d - 1)

    if structure == "R":
        if order is not None:
            raise ValueError("an R-vine's structure is selected from the data; leave order=None")
        return _fit_rvine(u, names, criterion, depth)

    if order is None:
        order = _default_order(u, structure)
    arranged = u[:, list(order)]

    def choose(first: NDArray, second: NDArray, level: int) -> Copula:
        if level >= depth:
            return IndependenceCopula(2)
        return _select_pair(first, second, names, criterion)

    trees: list[list[Copula]] = []
    if structure == "C":
        level_data = [arranged[:, j] for j in range(d)]
        for k in range(d - 1):
            root = level_data[0]
            chosen = [choose(root, level_data[i + 1], k) for i in range(d - 1 - k)]
            trees.append(chosen)
            # Past the truncation depth every later edge is independence
            # whatever its data, so the next level's h-transforms are not needed.
            if k < d - 2 and k + 1 < depth:
                level_data = [
                    _h(chosen[i], level_data[i + 1], root, given=1) for i in range(d - 1 - k)
                ]
    else:
        left = [arranged[:, j] for j in range(d - 1)]
        right = [arranged[:, j + 1] for j in range(d - 1)]
        for k in range(d - 1):
            chosen = [choose(left[i], right[i], k) for i in range(d - 1 - k)]
            trees.append(chosen)
            if k < d - 2 and k + 1 < depth:
                left, right = (
                    [_h(chosen[i], left[i], right[i], given=1) for i in range(len(chosen) - 1)],
                    [
                        _h(chosen[i + 1], left[i + 1], right[i + 1], given=0)
                        for i in range(len(chosen) - 1)
                    ],
                )

    return VineCopula(trees, structure=structure, order=order)


def _family_names(families: str | Sequence[str]) -> list[str]:
    """Resolve ``families`` (a group name or a list of names) to a list of names."""
    from rcopula.select import FAMILIES

    if isinstance(families, str):
        chosen = [name for name, spec in FAMILIES.items() if families in spec.groups]
        if not chosen:
            if families in FAMILIES:
                return [families]
            raise ValueError(f"unknown family or group {families!r}")
        return [name for name in chosen if FAMILIES[name].admissible(2)]
    return list(families)


#: Families that express only positive (``+1``) or only negative (``-1``)
#: dependence, and are therefore tried only where the edge's tau has that sign.
#: Plain Clayton, Gumbel and Joe are deliberately absent: they were always
#: tried before rotations existed, and still are.
_SIGNED = {
    **{f"{base}{deg}": -1 for base in ("clayton", "gumbel", "joe") for deg in (90, 270)},
    **{f"{base}180": 1 for base in ("clayton", "gumbel", "joe")},
    **{f"{base}_{deg}": -1 for base in ("bb1", "bb7") for deg in (90, 270)},
    **{name: 1 for name in ("bb1", "bb7", "bb1_180", "bb7_180")},
}


def _select_pair(first: NDArray, second: NDArray, names: list[str], criterion: str) -> Copula:
    """The best family for one edge, among those whose sign suits its tau."""
    from scipy import stats

    from rcopula.select import select_copula

    candidates = names
    if any(isinstance(name, str) and name in _SIGNED for name in names):
        tau = float(stats.kendalltau(first, second).statistic)
        sign = 0 if not np.isfinite(tau) or tau == 0.0 else (1 if tau > 0 else -1)
        if sign:
            candidates = [
                name
                for name in names
                if not isinstance(name, str) or _SIGNED.get(name, sign) == sign
            ] or names
    return select_copula(
        np.column_stack([first, second]), families=candidates, criterion=criterion
    ).best


class _Record:
    """One selected edge during R-vine selection.

    ``first`` came from the ``parents[0]`` side and ``second`` from the
    ``parents[1]`` side; parents are node ids of the tree below (variables for
    tree 1, edge indices of the previous tree after that). ``values`` holds
    ``u_{first | rest}`` and ``u_{second | rest}`` for building the next tree.
    """

    __slots__ = ("copula", "first", "parents", "second", "values")

    def __init__(self, first: int, second: int, parents: tuple[int, int]) -> None:
        self.first, self.second, self.parents = first, second, parents
        self.copula: Copula | None = None
        self.values: dict[int, NDArray[np.float64]] = {}


def _max_spanning_tree(n: int, edges: list[tuple[float, int, int]]) -> list[tuple[int, int]]:
    """Kruskal on ``(weight, a, b)``: heaviest first, ties broken by node ids."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen = []
    for _, a, b in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            chosen.append((a, b))
            if len(chosen) == n - 1:
                break
    return chosen


def _fit_rvine(u: NDArray[np.float64], names: list[str], criterion: str, depth: int) -> VineCopula:
    """Dissmann et al. (2013): maximum spanning trees on |tau|, one tree at a time."""
    from rcopula.core.other import IndependenceCopula
    from rcopula.dependence import cor_kendall

    d = u.shape[1]
    independence = IndependenceCopula(2)
    records: list[list[_Record]] = []

    # Tree 1: maximum spanning tree on the empirical |tau| of the variables.
    if depth >= 1:
        weight = np.nan_to_num(np.abs(cor_kendall(u)))
        pairs = [(float(weight[a, b]), a, b) for a in range(d) for b in range(a + 1, d)]
        chosen = _max_spanning_tree(d, pairs)
    else:
        chosen = [(0, b) for b in range(1, d)]  # any spanning tree will do
    level: list[_Record] = []
    for a, b in chosen:
        rec = _Record(a, b, (a, b))
        if depth >= 1:
            rec.copula = _select_pair(u[:, a], u[:, b], names, criterion)
            if depth >= 2:
                rec.values = {
                    a: _h(rec.copula, u[:, a], u[:, b], given=1),
                    b: _h(rec.copula, u[:, a], u[:, b], given=0),
                }
        else:
            rec.copula = independence
        level.append(rec)
    records.append(level)

    for t in range(1, d - 1):
        previous = records[-1]
        # Edges of the previous tree that share a node may be joined (proximity).
        incident: dict[int, list[int]] = {}
        for k, rec in enumerate(previous):
            for node in rec.parents:
                incident.setdefault(node, []).append(k)

        def outer(k: int, shared: int, previous: list[_Record] = previous) -> int:
            """The conditioned variable of previous edge ``k`` not below ``shared``."""
            rec = previous[k]
            return rec.second if rec.parents[0] == shared else rec.first

        if t < depth:
            candidates = [
                (ks[i], ks[j], node)
                for node, ks in incident.items()
                for i in range(len(ks))
                for j in range(i + 1, len(ks))
            ]
            columns: dict[tuple[int, int], int] = {}
            stacked: list[NDArray[np.float64]] = []
            for k1, k2, node in candidates:
                for k in (k1, k2):
                    key = (k, outer(k, node))
                    if key not in columns:
                        columns[key] = len(stacked)
                        stacked.append(previous[k].values[key[1]])
            taus = np.nan_to_num(np.abs(cor_kendall(np.column_stack(stacked))))
            weighted = [
                (
                    float(taus[columns[(k1, outer(k1, node))], columns[(k2, outer(k2, node))]]),
                    k1,
                    k2,
                )
                for k1, k2, node in candidates
            ]
            shared_of = {(k1, k2): node for k1, k2, node in candidates}
            joined = _max_spanning_tree(len(previous), weighted)
            joined = [(a, b, shared_of[(a, b)]) for a, b in joined]
        else:
            # Past the truncation level any valid tree will do: join every edge
            # at a node to the first edge there. Summed over nodes that is
            # exactly one fewer link than edges, and connected -- a spanning tree.
            joined = [(ks[0], k, node) for node, ks in incident.items() for k in ks[1:]]

        level = []
        for k1, k2, node in joined:
            x, y = outer(k1, node), outer(k2, node)
            rec = _Record(x, y, (k1, k2))
            if t < depth:
                a, b = previous[k1].values[x], previous[k2].values[y]
                rec.copula = _select_pair(a, b, names, criterion)
                if t + 1 < depth:
                    rec.values = {
                        x: _h(rec.copula, a, b, given=1),
                        y: _h(rec.copula, a, b, given=0),
                    }
            else:
                rec.copula = independence
            level.append(rec)
        # The data of the tree below are no longer needed.
        for rec in previous:
            rec.values = {}
        records.append(level)

    matrix, trees = _records_to_matrix(d, records)
    return VineCopula(trees, structure="R", matrix=matrix)


def _records_to_matrix(
    d: int, records: list[list[_Record]]
) -> tuple[NDArray[np.int64], list[list[Copula]]]:
    """Write selected edges as an R-vine matrix, column by column.

    Each column takes the one edge left in the highest unfinished tree, puts
    one of its conditioned variables ``x`` on the diagonal and walks down
    through the parent containing ``x`` to tree 1, writing each partner
    below. Every step follows a parent pointer, so no conditioning set is
    ever built and the whole matrix costs ``O(d^2)``. Each copula is turned,
    if necessary, so that its first argument is the off-diagonal variable.
    """
    m = np.zeros((d, d), dtype=np.int64)
    trees: list[list[Copula | None]] = [[None] * (d - 1 - t) for t in range(d - 1)]
    remaining = [set(range(len(level))) for level in records]
    used: set[int] = set()

    def walk(top: int, index: int, x: int) -> list[tuple[int, int, int]] | None:
        steps = []
        t, k = top, index
        while True:
            rec = records[t][k]
            if x not in (rec.first, rec.second) or k not in remaining[t]:
                return None
            steps.append((t, k, rec.second if rec.first == x else rec.first))
            if t == 0:
                return steps
            k = rec.parents[0] if rec.first == x else rec.parents[1]
            t -= 1

    for i in range(d - 1):
        top = d - 2 - i
        (index,) = remaining[top]
        rec = records[top][index]
        for x in (rec.first, rec.second):
            steps = walk(top, index, x)
            if steps is not None:
                break
        else:  # pragma: no cover - the selection always yields a regular vine
            raise RuntimeError("selected trees do not form a regular vine")
        m[i, i] = x
        used.add(x)
        for t, k, partner in steps:
            m[d - 1 - t, i] = partner
            remaining[t].discard(k)
            edge = records[t][k]
            assert edge.copula is not None
            # The matrix convention evaluates the copula at (partner, x).
            trees[t][i] = edge.copula if edge.first == partner else _swap_arguments(edge.copula)
    (last,) = set(range(d)) - used
    m[d - 1, d - 1] = last
    return m, [[cop for cop in level if cop is not None] for level in trees]


def _default_order(u: NDArray[np.float64], structure: str) -> list[int]:
    """A sensible ordering: strongest-dependent variable first.

    For a C-vine that variable becomes the root of tree 1, which is the whole
    reason to prefer a C-vine -- it is the structure for one factor and its
    satellites. For a D-vine the ordering matters less, so the same rule is used
    for reproducibility rather than optimality.
    """
    from rcopula.dependence import cor_kendall

    strength = np.abs(cor_kendall(u)).sum(axis=1)
    return [int(j) for j in np.argsort(-strength)]
