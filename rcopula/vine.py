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

Two structures cover almost all use and are implemented here:

* a **C-vine** (canonical) puts one variable at the centre of each tree, which
  suits a market factor plus its satellites;
* a **D-vine** (drawable) lays each tree out as a path, which suits an ordering
  -- a term structure, a spatial transect, a time series of maturities.

============================  ================================================
:class:`VineCopula`           The construction: density, sampler, Rosenblatt.
:func:`fit_vine`              Sequential estimation, selecting each pair.
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
    The sequential selection used by :func:`fit_vine`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.base import Copula, TailDependence
from rcopula.core.elliptical import GaussianCopula, P2p
from rcopula.dependence import pseudo_obs
from rcopula.transforms import conditional_cdf, conditional_ppf

__all__ = ["VineCopula", "fit_vine"]

#: Structures this module builds.
STRUCTURES = ("C", "D")

#: Families tried by :func:`fit_vine` when none are named. Deliberately spans
#: the tail-dependence possibilities -- none, lower only, upper only, both --
#: since that is what a pair-copula choice is really deciding.
DEFAULT_FAMILIES = ("independence", "gaussian", "student", "clayton", "gumbel", "frank")


def _h(copula: Copula, first: NDArray, second: NDArray, given: int) -> NDArray[np.float64]:
    r"""An h-function: :math:`\partial C/\partial u_{\text{given}}`.

    ``given=1`` returns the conditional distribution of the *first* argument
    given the second, and ``given=0`` the other way round.
    """
    points = np.column_stack([first, second])
    return np.clip(conditional_cdf(copula, points, given=given), 1e-12, 1.0 - 1e-12)


def _h_inverse(copula: Copula, target: NDArray, given: NDArray, side: int) -> NDArray[np.float64]:
    """Invert :func:`_h` in its first (``side=1``) or second (``side=0``) slot."""
    return np.clip(conditional_ppf(copula, target, given, given=side), 1e-12, 1.0 - 1e-12)


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
        ``len(pair_copulas) + 1``.
    structure : {"C", "D"}, default "D"
        ``"C"`` (canonical): each tree is a star around one central variable,
        which suits one market factor plus its satellites. ``"D"`` (drawable):
        each tree is a path, which suits naturally ordered variables such as
        maturities.
    order : sequence of int or None, default None
        A permutation of ``0, 1, ..., d-1`` saying which data column takes
        which position in the structure. For a C-vine the first entry is the
        root of tree 1; for a D-vine the sequence is the path. ``None`` means
        ``0, 1, ..., d-1``.
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
        ``"C"`` or ``"D"``.
    order : tuple of int
        The variable ordering, length ``d``.
    dim : int
        Number of variables ``d``.
    n_pairs : int
        Number of bivariate copulas, ``d(d-1)/2``.
    is_gaussian : bool
        Whether every pair-copula is Gaussian.

    Raises
    ------
    ValueError
        If ``structure`` is not ``"C"`` or ``"D"``, if a tree has the wrong
        number of copulas for the dimension, if any pair-copula is not
        bivariate, or if ``order`` is not a permutation of ``0..d-1``.

    Notes
    -----
    The density, the sampler (:meth:`rvs`) and the Rosenblatt transform are
    exact. There is no closed-form distribution function, so :meth:`cdf`
    raises :class:`NotImplementedError`; neither are there single-number
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
    """

    name = "Vine"
    param_names: tuple[str, ...] = ()

    def __init__(
        self,
        pair_copulas: Sequence[Sequence[Copula]],
        structure: str = "D",
        order: Sequence[int] | None = None,
        *,
        free: ArrayLike | None = None,
    ) -> None:
        if structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {STRUCTURES}, got {structure!r}")
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
        return VineCopula(self.pair_copulas, self.structure, self.order)

    def _reorder(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return u[:, list(self.order)]

    # -- density -------------------------------------------------------

    def _logpdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        arranged = self._reorder(u)
        total = np.zeros(arranged.shape[0])
        for copula, first, second in self._edges(arranged):
            total = total + copula.logpdf(np.column_stack([first, second]))
        return total

    def _edges(self, u: NDArray[np.float64]):
        """Yield ``(copula, first argument, second argument)`` for every edge.

        Walking the trees once and handing back the arguments keeps the density,
        the log-likelihood and the Rosenblatt transform on a single traversal,
        so there is one place for the recursion to be right or wrong.
        """
        if self.structure == "C":
            level = [u[:, j] for j in range(self.dim)]
            for k, copulas in enumerate(self.pair_copulas):
                root = level[0]
                for i, copula in enumerate(copulas):
                    yield copula, root, level[i + 1]
                if k < self.dim - 2:
                    level = [
                        _h(copula, level[i + 1], root, given=1) for i, copula in enumerate(copulas)
                    ]
            return

        # D-vine: tree k edge i joins variables i and i+k+1 given i+1..i+k, and
        # its two arguments are the corresponding conditionals.
        left = [u[:, j] for j in range(self.dim - 1)]
        right = [u[:, j + 1] for j in range(self.dim - 1)]
        for k, copulas in enumerate(self.pair_copulas):
            for i, copula in enumerate(copulas):
                yield copula, left[i], right[i]
            if k < self.dim - 2:
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
        """Inverse Rosenblatt: draw independent uniforms and unwind the trees."""
        w = rng.uniform(size=(size, self.dim))
        arranged = self._simulate_c_vine(w) if self.structure == "C" else self._simulate_d_vine(w)
        out = np.empty_like(arranged)
        out[:, list(self.order)] = arranged
        return np.clip(out, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))

    def _simulate_c_vine(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        """Aas et al. (2009), Algorithm 3."""
        d = self.dim
        v: list[list[NDArray[np.float64]]] = [[np.empty(0)] * d for _ in range(d)]
        x = np.empty_like(w)
        x[:, 0] = v[0][0] = w[:, 0]

        for i in range(1, d):
            value = w[:, i]
            for k in range(i - 1, -1, -1):
                value = _h_inverse(self.pair_copulas[k][i - k - 1], value, v[k][k], side=1)
            x[:, i] = v[i][0] = value
            if i == d - 1:
                break
            for j in range(i):
                v[i][j + 1] = _h(self.pair_copulas[j][i - j - 1], v[i][j], v[j][j], given=1)
        return x

    def _simulate_d_vine(self, w: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Inverse Rosenblatt for a D-vine.

        Variable :math:`i` is drawn from its conditional given the ones before
        it, which unwinds through the trees: apply the tree-:math:`k` inverse
        h-function, then rebuild the conditionals the next variable will need.
        """
        d = w.shape[1]
        x = np.empty_like(w)
        x[:, 0] = w[:, 0]
        # left[k] holds the tree-k conditional the next variable needs. The
        # matching "right" conditionals are consumed within a step and never
        # carried across one, so only `left` persists.
        left: list[NDArray[np.float64]] = []

        for i in range(1, d):
            value = w[:, i]
            # Unwind from the deepest tree that reaches this variable.
            for k in range(i - 1, 0, -1):
                value = _h_inverse(self.pair_copulas[k][i - k - 1], value, left[k - 1], side=0)
            value = _h_inverse(self.pair_copulas[0][i - 1], value, x[:, i - 1], side=0)
            x[:, i] = value

            # Rebuild the conditionals for the next variable.
            new_left = [_h(self.pair_copulas[0][i - 1], x[:, i - 1], x[:, i], given=1)]
            new_right = [_h(self.pair_copulas[0][i - 1], x[:, i - 1], x[:, i], given=0)]
            for k in range(1, i):
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
        conditional probability given the columns before it in the D-vine
        path. If the vine is the right model for the data, the output columns
        are independent and uniform on ``(0, 1)``, so it is the standard way to
        check a fitted vine (test the output for independence) and the inverse
        of how :meth:`rvs` samples.

        Parameters
        ----------
        u : array_like of float, shape (n, d)
            Copula-scale data (values in ``(0, 1)``), with columns in the
            original variable order; the vine's ``order`` is applied
            internally.

        Returns
        -------
        numpy.ndarray of float, shape (n, d)
            The transformed values, in the *structure* order (position ``i``
            is the ``i``-th variable along the D-vine path, i.e. column
            ``order[i]`` of the input).

        Raises
        ------
        ValueError
            If ``u`` does not have ``d`` columns.
        NotImplementedError
            If the vine is a C-vine; only D-vines are supported here.

        Notes
        -----
        The forward direction of the sampler, and the sharpest available check
        on both: under the true vine the output is independent
        :math:`\mathrm{Unif}(0,1)`, so any error in either recursion shows up as
        dependence that should not be there.

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
            raise NotImplementedError(
                "the Rosenblatt transform is implemented for D-vines; for a "
                "C-vine, use rvs and compare distributions instead"
            )

        out = np.empty_like(arranged)
        out[:, 0] = arranged[:, 0]
        left: list[NDArray[np.float64]] = []
        for i in range(1, self.dim):
            value = _h(self.pair_copulas[0][i - 1], arranged[:, i - 1], arranged[:, i], given=0)
            for k in range(1, i):
                value = _h(self.pair_copulas[k][i - k - 1], left[k - 1], value, given=0)
            out[:, i] = value

            new_left = [
                _h(self.pair_copulas[0][i - 1], arranged[:, i - 1], arranged[:, i], given=1)
            ]
            new_right = [
                _h(self.pair_copulas[0][i - 1], arranged[:, i - 1], arranged[:, i], given=0)
            ]
            for k in range(1, i):
                new_left.append(
                    _h(self.pair_copulas[k][i - k - 1], left[k - 1], new_right[k - 1], given=1)
                )
                new_right.append(
                    _h(self.pair_copulas[k][i - k - 1], left[k - 1], new_right[k - 1], given=0)
                )
            left = new_left
        return out

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

    def _edge_indices(self, tree: int, edge: int) -> tuple[int, int, list[int]]:
        """Which variables an edge joins, and what it conditions on."""
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
            and self.pair_copulas == other.pair_copulas
        )

    def __hash__(self) -> int:
        return hash(
            (self.structure, self.order, tuple(tuple(level) for level in self.pair_copulas))
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
    families: Sequence[str] = DEFAULT_FAMILIES,
    order: Sequence[int] | None = None,
    criterion: str = "aic",
    truncate: int | None = None,
) -> VineCopula:
    """Fit a vine copula to data, picking the best family for each pair automatically.

    Give it a table of observations (one column per variable) and it returns a
    :class:`VineCopula` whose every pair-copula family and parameter has been
    chosen from the data. Use it when you want flexible, pair-by-pair
    dependence across three or more variables without choosing families by
    hand.

    Parameters
    ----------
    data : array_like or pandas.DataFrame of float, shape (n, d)
        Observations, one row per observation and one column per variable,
        with ``d >= 2``. Raw data are fine: they are converted to
        pseudo-observations (ranks scaled into ``(0, 1)``) internally.
    structure : {"C", "D"}, default "D"
        ``"C"`` builds star-shaped trees around a central variable; ``"D"``
        builds path-shaped trees. See :class:`VineCopula`.
    families : sequence of str, default DEFAULT_FAMILIES
        Candidate families tried on every edge; names as in
        :data:`~rcopula.select.FAMILIES`. The default is ``("independence",
        "gaussian", "student", "clayton", "gumbel", "frank")``, which spans
        no tail dependence, lower-only, upper-only and both.
    order : sequence of int or None, default None
        A permutation of ``0..d-1`` giving the variable ordering. ``None``
        orders variables by total absolute Kendall's tau with the others,
        strongest first. For a C-vine that puts the most connected variable at
        the root -- which is what makes a C-vine worth choosing; for a D-vine
        the same rule is used for reproducibility.
    criterion : {"aic", "bic", "loglik", "xv"}, default "aic"
        How each edge's winning family is chosen, passed to
        :func:`~rcopula.select.select_copula`: lowest AIC or BIC, highest
        log-likelihood, or cross-validated likelihood.
    truncate : int or None, default None
        Fit only the first ``truncate`` trees and set every pair-copula in the
        higher trees to independence. ``None`` fits all ``d - 1`` trees.
        Higher trees usually carry little, and truncating is the standard way
        to stop a vine from spending parameters on noise.

    Returns
    -------
    VineCopula
        The fitted vine. Read ``fitted.order`` for the ordering actually used
        and ``fitted.describe()`` for the chosen families.

    Raises
    ------
    ValueError
        If ``data`` has fewer than two columns or ``structure`` is not ``"C"``
        or ``"D"``.

    Notes
    -----
    Tree by tree, each edge's family is selected by
    :func:`~rcopula.select.select_copula` on the h-transformed data that edge
    actually sees, and the h-functions for the next tree are built from the
    winner. That is Dissmann et al.'s sequential procedure, and it is what makes
    a :math:`d(d-1)/2`-parameter model estimable at all.

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
    """
    from rcopula.core.other import IndependenceCopula
    from rcopula.select import select_copula

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

    if order is None:
        order = _default_order(u, structure)
    arranged = u[:, list(order)]
    depth = d - 1 if truncate is None else min(int(truncate), d - 1)

    def choose(first: NDArray, second: NDArray, level: int) -> Copula:
        if level >= depth:
            return IndependenceCopula(2)
        return select_copula(
            np.column_stack([first, second]), families=list(families), criterion=criterion
        ).best

    trees: list[list[Copula]] = []
    if structure == "C":
        level_data = [arranged[:, j] for j in range(d)]
        for k in range(d - 1):
            root = level_data[0]
            chosen = [choose(root, level_data[i + 1], k) for i in range(d - 1 - k)]
            trees.append(chosen)
            if k < d - 2:
                level_data = [
                    _h(chosen[i], level_data[i + 1], root, given=1) for i in range(d - 1 - k)
                ]
    else:
        left = [arranged[:, j] for j in range(d - 1)]
        right = [arranged[:, j + 1] for j in range(d - 1)]
        for k in range(d - 1):
            chosen = [choose(left[i], right[i], k) for i in range(len(left))]
            trees.append(chosen)
            if k < d - 2:
                left, right = (
                    [_h(chosen[i], left[i], right[i], given=1) for i in range(len(chosen) - 1)],
                    [
                        _h(chosen[i + 1], left[i + 1], right[i + 1], given=0)
                        for i in range(len(chosen) - 1)
                    ],
                )

    return VineCopula(trees, structure=structure, order=order)


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
