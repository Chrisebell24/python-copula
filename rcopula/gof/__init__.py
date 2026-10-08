"""Goodness-of-fit testing for copulas: does a fitted copula family match the data?

:func:`gof_test` tests one family against one sample, :func:`gof_two_sample`
compares the copulas of two samples, and :func:`gof_statistic` computes the
underlying distance without a p-value.
"""

from __future__ import annotations

from rcopula.gof.api import GofResult, gof_test, gof_two_sample
from rcopula.gof.statistics import STATISTICS, empirical_copula_at, gof_statistic

__all__ = [
    "STATISTICS",
    "GofResult",
    "empirical_copula_at",
    "gof_statistic",
    "gof_test",
    "gof_two_sample",
]
