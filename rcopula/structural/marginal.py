r"""Lower-dimensional margins of a copula (R's ``margCopula``).

A :math:`d`-dimensional copula contains every lower-dimensional copula inside
it: fix a subset of coordinates and let the rest go to one, and what is left is
itself a copula. Extracting it is the natural way to ask "what does my
five-asset model say about these two?", and to check a fitted model against a
bivariate diagnostic that only makes sense in two dimensions -- a Pickands plot,
a K-plot, a tail concentration function.

For most families the answer is immediate: an Archimedean copula's margin is the
same generator in fewer dimensions, and an elliptical copula's is the
corresponding sub-matrix of its correlation. R supports exactly those two
classes and refuses everything else. Here the structural constructions come too,
because a rotation, a mixture and a Khoudraji device all commute with taking a
margin.

============================  ================================================
:func:`marginal_copula`       The margin on a chosen subset of coordinates.
============================  ================================================

Examples
--------
>>> import rcopula as rc
>>> from rcopula.structural.marginal import marginal_copula
>>> five = rc.GaussianCopula(
...     [0.72, 0.63, 0.54, 0.45, 0.56, 0.48, 0.40, 0.42, 0.35, 0.30],
...     dim=5, dispstr="un",
... )
>>> pair = marginal_copula(five, [0, 3])
>>> pair.dim
2
>>> float(pair.params[0]) == float(five.sigma()[0, 3])
True

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer,
    Theorem 2.10.13 -- margins of a copula are copulas.
Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall/CRC.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from rcopula.core.base import Copula

__all__ = ["marginal_copula"]


def marginal_copula(copula: Copula, indices: Sequence[int]) -> Copula:
    r"""The copula of a subset of the coordinates.

    Parameters
    ----------
    copula : Copula
    indices : sequence of int
        Which coordinates to keep, in the order they should appear. At least
        two, all distinct, all within range. The order matters for an
        asymmetric copula: ``[1, 0]`` is not ``[0, 1]`` unless the copula is
        exchangeable.

    Returns
    -------
    Copula
        Of dimension ``len(indices)``.

    Raises
    ------
    NotImplementedError
        For families whose margins are not of the same family -- the empirical
        copula, and extreme-value families defined only in two dimensions.

    Notes
    -----
    Exchangeable families (Archimedean, and elliptical with ``dispstr="ex"``)
    have the same margin whichever coordinates are chosen, so ``indices`` only
    fixes the dimension. For an unstructured correlation matrix it selects the
    sub-matrix, and for ``ar1`` or ``toep`` the result is generally *not* of the
    same structure -- dropping a coordinate from an AR(1) chain leaves a gap --
    so it comes back unstructured.

    Examples
    --------
    An Archimedean margin is the same generator in fewer dimensions, so its
    Kendall's tau is unchanged:

    >>> import rcopula as rc
    >>> from rcopula.structural.marginal import marginal_copula
    >>> five = rc.ClaytonCopula(2.0, dim=5)
    >>> pair = marginal_copula(five, [1, 4])
    >>> pair.dim, float(pair.params[0])
    (2, 2.0)
    >>> bool(abs(pair.tau() - five.tau()) < 1e-12)
    True

    And the margin agrees with the parent, which is the property that defines
    it -- setting the dropped coordinates to one:

    >>> import numpy as np
    >>> point = np.array([[0.3, 0.7]])
    >>> full = np.array([[1.0, 0.3, 1.0, 1.0, 0.7]])
    >>> bool(abs(pair.cdf(point)[0] - five.cdf(full)[0]) < 1e-12)
    True

    Dropping a coordinate from an AR(1) chain leaves a gap, so the result is
    unstructured rather than AR(1):

    >>> chain = rc.GaussianCopula(0.8, dim=4, dispstr="ar1")
    >>> gapped = marginal_copula(chain, [0, 2, 3])
    >>> gapped.dispstr
    'un'
    >>> bool(np.allclose(gapped.sigma(), chain.sigma()[np.ix_([0, 2, 3], [0, 2, 3])]))
    True
    """
    from rcopula.core.archimedean import ArchimedeanCopula
    from rcopula.core.elliptical import EllipticalCopula, GaussianCopula, P2p, StudentCopula
    from rcopula.core.empirical import EmpiricalCopula
    from rcopula.core.other import FrechetUpperCopula, IndependenceCopula
    from rcopula.structural.khoudraji import KhoudrajiCopula
    from rcopula.structural.mixture import MixtureCopula
    from rcopula.structural.nested import NestedArchimedean
    from rcopula.structural.rotated import RotatedCopula

    chosen = [int(i) for i in indices]
    if len(chosen) < 2:
        raise ValueError(f"a margin needs at least 2 coordinates, got {len(chosen)}")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"indices must be distinct, got {chosen}")
    if any(i < 0 or i >= copula.dim for i in chosen):
        raise ValueError(f"indices must lie in [0, {copula.dim}), got {chosen}")
    if len(chosen) == copula.dim and chosen == sorted(chosen):
        return copula

    size = len(chosen)

    if isinstance(copula, EmpiricalCopula):
        raise NotImplementedError(
            "an EmpiricalCopula's margin is the empirical copula of the "
            "corresponding columns; build it directly from data[:, indices]."
        )

    if isinstance(copula, IndependenceCopula | FrechetUpperCopula):
        return type(copula)(dim=size)

    if isinstance(copula, ArchimedeanCopula):
        # Exchangeable, so only the dimension changes. Rebuilding through the
        # *concrete* class matters: ArchimedeanCopula(generator, ...) would work
        # but returns a copula whose type is no longer ClaytonCopula, which
        # breaks isinstance checks and the serialization registry.
        builder = cast("Any", type(copula))
        return cast("Copula", builder(float(copula.params[0]), dim=size))

    if isinstance(copula, EllipticalCopula):
        block = np.asarray(copula.sigma())[np.ix_(chosen, chosen)]
        flat = P2p(block)
        if isinstance(copula, StudentCopula):
            return StudentCopula(
                flat,
                dim=size,
                dispstr="un",
                df=float(copula.df),
                df_fixed=bool(getattr(copula, "df_fixed", False)),
            )
        return GaussianCopula(flat, dim=size, dispstr="un")

    if isinstance(copula, RotatedCopula):
        flip = np.asarray(copula.flip, dtype=bool)[chosen]
        return RotatedCopula(marginal_copula(copula.base, chosen), flip)

    if isinstance(copula, MixtureCopula):
        # A mixture of margins is the margin of the mixture: integrating out a
        # coordinate is linear, so it passes through the weights untouched.
        return MixtureCopula(
            [marginal_copula(component, chosen) for component in copula.copulas],
            weights=np.asarray(copula.weights),
        )

    if isinstance(copula, NestedArchimedean):
        if size != 2:
            raise NotImplementedError(
                "a nested Archimedean's margins are implemented for pairs only. "
                "A larger subset is a nested copula on the induced sub-tree, "
                f"which is not built here; asked for {size} coordinates."
            )
        # Two variables meet at exactly one node of the tree, and that node's
        # generator *is* their bivariate copula -- the same fact tau_matrix
        # rests on. So the margin needs no integration at all.
        node = copula.lowest_common_ancestor(chosen[0], chosen[1])
        return marginal_copula(node.generator_copula, [0, 1])

    if isinstance(copula, KhoudrajiCopula):
        if size != 2:
            raise NotImplementedError(
                "a Khoudraji copula's margins are implemented for pairs only; "
                f"asked for {size} coordinates"
            )
        shapes = np.asarray(copula.shapes, dtype=float)[chosen]
        return KhoudrajiCopula(
            marginal_copula(copula.copula1, chosen),
            marginal_copula(copula.copula2, chosen),
            shapes=shapes,
        )

    raise NotImplementedError(
        f"margins of a {type(copula).__name__} are not available. Bivariate-only "
        "families have no lower-dimensional margin to take, and for anything "
        "else the margin is generally not in the same family -- fit the "
        "sub-copula to data[:, indices] instead."
    )
