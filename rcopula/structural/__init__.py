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
==========================  ==================================================
"""

from __future__ import annotations

from rcopula.structural.rotated import RotatedCopula, survival

__all__ = ["RotatedCopula", "survival"]
