r"""Nested (hierarchical) Archimedean copulas.

An Archimedean copula gives every pair of variables the same dependence. That is
often wrong in an obvious way: three equity indices and two bond yields do not
form one exchangeable block, and forcing them to does not merely lose detail, it
puts the wrong number on every cross-asset pair.

Nesting fixes it while keeping the Archimedean machinery. Replace an argument of
one generator by a whole copula built from another:

.. math::

    C(\mathbf u) = \psi_0\Bigl(
        \sum_{j \in \text{direct}} \psi_0^{-1}(u_j)
        + \sum_{\text{children}} \psi_0^{-1}\bigl(C_{\text{child}}\bigr)\Bigr).

The result is a **tree**. Leaves sharing a parent are more dependent than leaves
that meet only at the root, and the dependence between any two variables is
governed by their **lowest common ancestor** -- which makes Kendall's tau
between them exact and immediate, with no integration at all.

For the result to be a copula the generators must nest compatibly. For two
copulas of the same family the condition is simply that the inner parameter is
at least the outer one: **dependence may increase as you descend, never
decrease**. That is checked on construction rather than left to the user.

============================  ================================================
:class:`NestedArchimedean`    The tree, its CDF, sampler and tau structure.
:func:`fit_nested`            Estimate every node by inverting Kendall's tau.
============================  ================================================

.. note::

   Sampling is exact for **Gumbel** and **Clayton**. Gumbel's inner frailty is
   an ordinary positive stable law; Clayton's is an *exponentially tilted*
   stable, which is why nested Archimedean copulas have not been available in
   Python before -- see :func:`~rcopula.special.stable.retstable`. Frank and
   Joe need their own conditional frailties and are refused explicitly rather
   than approximated.

   The **density** is not implemented. It requires high-order derivatives of a
   composition of generators, and every quantity this module does provide --
   the CDF, exact sampling, exact pairwise tau and tail dependence, and
   estimation by tau inversion -- is available without it.

References
----------
Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman & Hall.
    The nesting condition.
McNeil, A. J. (2008). Sampling nested Archimedean copulas.
    *Journal of Statistical Computation and Simulation* 78(6), 567-581.
    The frailty algorithm implemented here.
Hofert, M. (2011). Efficiently sampling nested Archimedean copulas.
    *Computational Statistics & Data Analysis* 55(1), 57-70.
    The tilted-stable inner frailty that nested Clayton needs.
Okhrin, O., Okhrin, Y. and Schmid, W. (2013). On the structure and estimation
    of hierarchical Archimedean copulas.
    *Journal of Econometrics* 173(2), 189-204.
    Estimation by inverting pairwise Kendall's tau.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rcopula.core.archimedean import (
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
    _ConcreteArchimedean,
)
from rcopula.core.base import Copula, TailDependence
from rcopula.dependence import cor_kendall
from rcopula.special.stable import retstable, rstable_positive

__all__ = ["NestedArchimedean", "fit_nested"]

#: Families whose inner frailty is known in closed form, and how to draw it.
#: Frank and Joe need conditional frailties that are not implemented; they are
#: refused rather than silently approximated.
_SAMPLEABLE = ("Gumbel", "Clayton")


class NestedArchimedean(Copula):
    """A hierarchical Archimedean copula.

    Parameters
    ----------
    generator : Copula
        A one-parameter Archimedean copula supplying this node's generator. Its
        ``dim`` is ignored -- the tree determines dimension.
    components : sequence of int, optional
        Zero-based indices of the variables attached **directly** to this node.
    children : sequence of NestedArchimedean, optional
        Sub-trees nested inside it.

    Notes
    -----
    Every variable must appear exactly once across the whole tree, and the
    indices must be ``0 .. d-1``. Both are checked, because a mis-specified
    tree otherwise produces a plausible-looking object that is not a copula.

    Examples
    --------
    Five variables in two blocks -- the shape a single Archimedean copula cannot
    express:

    >>> import rcopula as rc
    >>> from rcopula.structural import NestedArchimedean
    >>> equities = NestedArchimedean(rc.GumbelCopula(4.0), [0, 1, 2])
    >>> bonds = NestedArchimedean(rc.GumbelCopula(3.0), [3, 4])
    >>> cop = NestedArchimedean(rc.GumbelCopula(1.5), children=[equities, bonds])
    >>> cop.dim
    5
    >>> print(cop.describe())
    Nested Archimedean copula, dim 5
    Gumbel(theta=1.5)
      Gumbel(theta=4) on [0, 1, 2]
      Gumbel(theta=3) on [3, 4]

    Kendall's tau depends on where two variables meet, and is exact:

    >>> tau = cop.tau_matrix()
    >>> float(round(tau[0, 1], 6))     # both inside the equity block
    0.75
    >>> float(round(tau[3, 4], 6))     # both inside the bond block
    0.666667
    >>> float(round(tau[0, 3], 6))     # they meet only at the root
    0.333333

    No single Archimedean copula can produce three different pairwise taus.
    """

    name = "NestedArchimedean"
    param_names: tuple[str, ...] = ()

    def __init__(
        self,
        generator: Copula,
        components: Sequence[int] | None = None,
        children: Sequence[NestedArchimedean] | None = None,
        *,
        free: ArrayLike | None = None,
        _validate: bool = True,
    ) -> None:
        if not isinstance(generator, _ConcreteArchimedean):
            raise TypeError(
                f"the generator must be a one-parameter Archimedean copula, got "
                f"{type(generator).__name__}"
            )
        self.generator_copula = generator
        self.components = tuple(int(j) for j in (components or ()))
        self.children = tuple(children or ())

        leaves = self.leaves()
        if _validate:
            if not leaves:
                raise ValueError("a node must contain at least one variable")
            if len(set(leaves)) != len(leaves):
                raise ValueError(f"variables repeat in the tree: {sorted(leaves)}")
            self._check_nesting()

        super().__init__(np.empty(0), max(len(leaves), 2), free=free)

    def _require_root(self) -> None:
        """Only a *root* is a copula, and only then must its indices be 0..d-1.

        A sub-tree legitimately holds indices like ``[3, 4]`` -- that is what
        makes it a sub-tree. So contiguity is checked when the object is used as
        a copula rather than when it is built, since at construction time a node
        does not know whether it will be a root or a branch.
        """
        leaves = sorted(self.leaves())
        if leaves != list(range(len(leaves))):
            raise ValueError(
                f"this node covers variables {leaves}, so it is a sub-tree rather "
                f"than a copula. A root must cover 0..{len(leaves) - 1} exactly."
            )

    # -- structure -----------------------------------------------------

    def leaves(self) -> tuple[int, ...]:
        """Every variable index below this node, in tree order."""
        found = list(self.components)
        for child in self.children:
            found.extend(child.leaves())
        return tuple(found)

    @property
    def theta(self) -> float:
        """This node's dependence parameter."""
        return float(self.generator_copula.params[0])

    @property
    def depth(self) -> int:
        """Height of the tree below this node; a flat copula has depth 1."""
        return 1 + max((child.depth for child in self.children), default=0)

    def nodes(self) -> list[NestedArchimedean]:
        """This node and every node beneath it, root first."""
        out = [self]
        for child in self.children:
            out.extend(child.nodes())
        return out

    def _check_nesting(self) -> None:
        r"""Dependence may increase as you descend, never decrease.

        For two copulas of the same Archimedean family the sufficient and
        necessary condition for :math:`\psi_0^{-1}\circ\psi_1` to have a
        completely monotone derivative -- which is what makes the composition a
        copula -- reduces to :math:`\theta_1 \ge \theta_0`. Across families the
        condition is more delicate, so mixing them is refused.
        """
        for child in self.children:
            if type(child.generator_copula) is not type(self.generator_copula):
                raise ValueError(
                    f"mixing families in one tree is not supported: "
                    f"{self.generator_copula.name} nests {child.generator_copula.name}. "
                    "The nesting condition across families is family-specific and "
                    "not checked here, so allowing it would risk building something "
                    "that is not a copula."
                )
            if child.theta < self.theta - 1e-12:
                raise ValueError(
                    f"nesting requires the inner parameter to be at least the outer "
                    f"one: {self.generator_copula.name} theta={self.theta:g} nests "
                    f"theta={child.theta:g}. Dependence may increase as you descend, "
                    "never decrease."
                )

    # -- plumbing ------------------------------------------------------

    @property
    def param_bounds(self) -> list[tuple[float, float]]:
        return []

    def _reconstruct(self, params: ArrayLike, free: ArrayLike) -> NestedArchimedean:
        return NestedArchimedean(self.generator_copula, self.components, self.children)

    def with_thetas(self, thetas: Sequence[float]) -> NestedArchimedean:
        """Rebuild the tree with new parameters, in :meth:`nodes` order."""
        values = list(thetas)
        if len(values) != len(self.nodes()):
            raise ValueError(f"got {len(values)} parameters for {len(self.nodes())} nodes")

        def rebuild(node: NestedArchimedean, position: int) -> tuple[NestedArchimedean, int]:
            theta = values[position]
            position += 1
            rebuilt = []
            for child in node.children:
                new_child, position = rebuild(child, position)
                rebuilt.append(new_child)
            return (
                NestedArchimedean(
                    node.generator_copula.with_params([theta]), node.components, rebuilt
                ),
                position,
            )

        return rebuild(self, 0)[0]

    # -- evaluation ----------------------------------------------------

    def _cdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        self._require_root()
        return self._value(u)

    def _inverse_sum(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        r""":math:`\sum_j \psi_0^{-1}(u_j)` over direct components and children."""
        total = np.zeros(u.shape[0])
        if self.components:
            total = total + self.generator_copula.ipsi(u[:, list(self.components)]).sum(axis=1)
        for child in self.children:
            total = total + self.generator_copula.ipsi(child._value(u))
        return total

    def _value(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """This node's copula value, used recursively by its parent.

        Separate from ``_cdf`` because ``_cdf`` asserts it is being used as a
        root, and a child is by definition not one.
        """
        return np.asarray(self.generator_copula.psi(self._inverse_sum(u)))

    def _logpdf(self, u: NDArray[np.float64], params: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError(
            "the density of a nested Archimedean copula needs high-order "
            "derivatives of a composition of generators and is not implemented. "
            "Its CDF, exact sampling, exact pairwise tau and tail dependence, and "
            "estimation by tau inversion (rcopula.structural.fit_nested) all are."
        )

    def _rvs(
        self, size: int, params: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """McNeil's (2008) nested frailty algorithm."""
        self._require_root()
        family = self.generator_copula.name
        for node in self.nodes():
            if node.generator_copula.name not in _SAMPLEABLE:
                raise NotImplementedError(
                    f"sampling a nested {node.generator_copula.name} copula needs its "
                    f"conditional inner frailty, which is not implemented. "
                    f"Nested {' and '.join(_SAMPLEABLE)} copulas are exact."
                )
        out = np.empty((size, self.dim))
        root = self.generator_copula.generator.rvs_frailty(size, self.theta, rng)
        self._fill(out, root, rng, family)
        return np.clip(out, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))

    def _fill(
        self,
        out: NDArray[np.float64],
        frailty: NDArray[np.float64],
        rng: np.random.Generator,
        family: str,
    ) -> None:
        r"""Attach this node's variables, then recurse with the inner frailty.

        Given the frailty :math:`V` at this node, each attached variable is
        :math:`U_j = \psi(E_j / V)` with independent unit exponentials. Each
        child needs a new frailty whose conditional Laplace transform is
        :math:`\exp(-V\,\psi_0^{-1}(\psi_1(t)))`, which for these two families
        is a stable law -- untilted for Gumbel, exponentially tilted for
        Clayton.
        """
        size = out.shape[0]
        if self.components:
            exponential = rng.exponential(1.0, size=(size, len(self.components)))
            out[:, list(self.components)] = self.generator_copula.psi(
                exponential / frailty[:, None]
            )
        for child in self.children:
            child._fill(out, self._inner_frailty(child, frailty, rng, family), rng, family)

    def _inner_frailty(
        self,
        child: NestedArchimedean,
        frailty: NDArray[np.float64],
        rng: np.random.Generator,
        family: str,
    ) -> NDArray[np.float64]:
        r"""Draw :math:`V_{01}` with :math:`E[e^{-tV_{01}}|V_0] = e^{-V_0\psi_0^{-1}(\psi_1(t))}`.

        * **Gumbel**: :math:`\psi_0^{-1}(\psi_1(t)) = t^{\alpha}` with
          :math:`\alpha = \theta_0/\theta_1`, so :math:`V_{01} = V_0^{1/\alpha} S`
          for an ordinary positive stable :math:`S`.
        * **Clayton**: :math:`\psi_0^{-1}(\psi_1(t)) = (1+t)^{\alpha} - 1`, which
          is the tilted stable with tilt 1.
        """
        alpha = self.theta / child.theta
        if alpha >= 1.0:
            # Equal parameters: the child adds no extra dependence, so the
            # frailty passes straight through.
            return frailty
        if family == "Gumbel":
            return frailty ** (1.0 / alpha) * rstable_positive(frailty.size, alpha, rng)
        return retstable(frailty.size, alpha, frailty, 1.0, rng)

    # -- dependence ----------------------------------------------------

    def lowest_common_ancestor(self, i: int, j: int) -> NestedArchimedean:
        """The node whose generator governs the pair ``(i, j)``."""
        if i == j:
            raise ValueError("a variable has no common ancestor with itself")
        for child in self.children:
            below = set(child.leaves())
            if i in below and j in below:
                return child.lowest_common_ancestor(i, j)
        return self

    def tau_matrix(self) -> NDArray[np.float64]:
        """Pairwise Kendall's tau, exactly.

        Two variables meet at exactly one node, and that node's generator is the
        bivariate copula of the pair -- so the tau is the generator's own, with
        no integration. A flat Archimedean copula would give one number for
        every pair; this gives one per branch of the tree.
        """
        self._require_thetas()
        d = self.dim
        out = np.eye(d)
        for i in range(d):
            for j in range(i + 1, d):
                out[i, j] = out[j, i] = self.lowest_common_ancestor(i, j).generator_copula.tau()
        return out

    def lambda_matrix(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Pairwise lower and upper tail dependence, by the same argument."""
        self._require_thetas()
        d = self.dim
        lower, upper = np.ones((d, d)), np.ones((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                pair = self.lowest_common_ancestor(i, j).generator_copula.lambda_()
                lower[i, j] = lower[j, i] = pair.lower
                upper[i, j] = upper[j, i] = pair.upper
        return lower, upper

    def tau(self) -> float:
        """Refused: a nested copula has no single tau, which is the point of it."""
        raise NotImplementedError(
            "a nested Archimedean copula has a different Kendall's tau for "
            "different pairs -- that is what it is for. Use tau_matrix()."
        )

    def rho(self) -> float:
        raise NotImplementedError(
            "a nested Archimedean copula has a different Spearman's rho for "
            "different pairs. Estimate it pairwise from a sample."
        )

    def lambda_(self) -> TailDependence:
        raise NotImplementedError(
            "tail dependence differs by pair in a nested copula; use lambda_matrix()."
        )

    def _require_thetas(self) -> None:
        self._require_root()
        for node in self.nodes():
            if np.isnan(node.theta):
                raise ValueError(
                    "the tree has unspecified parameters; fit it with "
                    "rcopula.structural.fit_nested, or supply values"
                )

    # -- presentation --------------------------------------------------

    def describe(self) -> str:
        return f"Nested Archimedean copula, dim {self.dim}\n{self._render(0)}"

    def _render(self, level: int) -> str:
        pad = "  " * level
        label = f"{self.generator_copula.name}(theta={self.theta:g})"
        if self.components:
            label += f" on {list(self.components)}"
        rows = [pad + label]
        rows.extend(child._render(level + 1) for child in self.children)
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"<Nested {self.generator_copula.name} copula, dim {self.dim}, depth {self.depth}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NestedArchimedean):
            return NotImplemented
        return (
            self.generator_copula == other.generator_copula
            and self.components == other.components
            and self.children == other.children
        )

    def __hash__(self) -> int:
        return hash(("Nested", self.generator_copula, self.components, self.children))

    @classmethod
    def from_tau(cls, tau: float, dim: int = 2, **kwargs: Any) -> Copula:
        raise NotImplementedError(
            "a nested copula has one parameter per node; a single tau cannot "
            "identify them. Use fit_nested."
        )


def fit_nested(structure: NestedArchimedean, data: ArrayLike) -> NestedArchimedean:
    r"""Estimate every node by inverting Kendall's tau.

    Each node governs the pairs whose lowest common ancestor it is, so the
    natural estimator averages the sample tau over exactly those pairs and
    inverts the generator's own tau function. That is Okhrin, Okhrin & Schmid's
    estimator and R's ``etau``, and it needs no density -- which matters here,
    because the nested density is not available.

    The estimates are then made to respect the nesting condition by taking a
    running maximum down the tree. Sampling noise can otherwise leave a child
    below its parent, which would not be a copula.

    Parameters
    ----------
    structure : NestedArchimedean
        The tree shape. Its parameters are ignored and replaced.
    data : array_like
        ``(n, d)`` observations, on any scale -- only ranks are used.

    Examples
    --------
    >>> import rcopula as rc
    >>> from rcopula.structural import NestedArchimedean, fit_nested
    >>> inner = NestedArchimedean(rc.GumbelCopula(4.0), [0, 1, 2])
    >>> truth = NestedArchimedean(rc.GumbelCopula(1.6), [3], [inner])
    >>> u = truth.rvs(4000, random_state=0)
    >>> fitted = fit_nested(truth, u)
    >>> bool(abs(fitted.theta - 1.6) < 0.15)
    True
    >>> bool(abs(fitted.children[0].theta - 4.0) < 0.4)
    True
    """
    u = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if u.shape[1] != structure.dim:
        raise ValueError(f"data has {u.shape[1]} columns but the tree has dim={structure.dim}")
    sample_tau = cor_kendall(u)

    def estimate(node: NestedArchimedean, floor: float) -> NestedArchimedean:
        governed = [
            sample_tau[i, j]
            for i in range(structure.dim)
            for j in range(i + 1, structure.dim)
            if structure.lowest_common_ancestor(i, j) is node
        ]
        family = type(node.generator_copula)
        if governed:
            target = float(np.mean(governed))
            fitted = family.from_tau(float(np.clip(target, -0.999, 0.999)))
            theta = max(float(fitted.params[0]), floor)
        else:  # pragma: no cover - a node governing no pair is degenerate
            theta = floor
        return NestedArchimedean(
            node.generator_copula.with_params([theta]),
            node.components,
            [estimate(child, theta) for child in node.children],
        )

    # The root's own floor is the family's independence point.
    independence = 1.0 if isinstance(structure.generator_copula, GumbelCopula | JoeCopula) else 0.0
    if isinstance(structure.generator_copula, FrankCopula | ClaytonCopula):
        independence = -np.inf
    return estimate(structure, independence)
