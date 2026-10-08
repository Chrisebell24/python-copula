"""Tests of basic properties of the dependence, to run before choosing a copula family.

Specification tests: independence (:func:`indep_test`, :func:`dependogram`,
:func:`serial_indep_test`), exchangeability (:func:`exch_test`), radial
symmetry (:func:`rad_sym_test`) and extreme-value dependence (:func:`ev_test`).
"""

from __future__ import annotations

from rcopula.htest.api import (
    DependogramResult,
    TestResult,
    dependogram,
    ev_test,
    exch_test,
    indep_test,
    rad_sym_test,
    serial_indep_test,
)

__all__ = [
    "DependogramResult",
    "TestResult",
    "dependogram",
    "ev_test",
    "exch_test",
    "indep_test",
    "rad_sym_test",
    "serial_indep_test",
]
