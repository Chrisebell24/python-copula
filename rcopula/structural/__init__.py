"""Copulas built from other copulas.

Parametric families are a fixed menu. These constructions extend it without
adding new mathematics: reflect a family to move its tail dependence, mix two
families to get both of their shapes, or apply the Khoudraji device to break the
exchangeability that every Archimedean copula imposes.

Each result is a genuine copula and carries the base families' parameters, so
``fit``, ``select_copula`` and the goodness-of-fit tests all work on them
unchanged.

==========================  ==================================================
:class:`RotatedCopula`      Reflect any subset of coordinates.
:func:`survival`            The survival copula -- every coordinate reflected.
:class:`KhoudrajiCopula`    Break exchangeability with a shape per coordinate.
:class:`MixtureCopula`      Convex combination -- both tails at once.
:class:`NestedArchimedean`  A tree: dependence that varies by branch.
==========================  ==================================================
"""

from __future__ import annotations

from rcopula.structural.khoudraji import KhoudrajiCopula
from rcopula.structural.mixture import MixtureCopula
from rcopula.structural.nested import NestedArchimedean, fit_nested
from rcopula.structural.rotated import RotatedCopula, survival

__all__ = [
    "KhoudrajiCopula",
    "MixtureCopula",
    "NestedArchimedean",
    "RotatedCopula",
    "fit_nested",
    "survival",
]
