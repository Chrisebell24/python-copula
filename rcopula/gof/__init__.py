"""Goodness-of-fit testing for copulas."""

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
