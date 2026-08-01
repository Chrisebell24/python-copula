"""rcopula - copula modelling in Python.

A full-featured replication of the R ``copula`` package (Hofert, Kojadinovic,
Maechler, Yan), implemented clean-room from the published literature and
verified numerically against R's outputs.

See ``NOTICE`` for the reference list and ``CONTRIBUTING.md`` for the
clean-room rule.
"""

from __future__ import annotations

from rcopula import credit, derivatives, garch, insurance, portfolio, risk
from rcopula.core.archimedean import (
    AMHCopula,
    ArchimedeanCopula,
    ClaytonCopula,
    FrankCopula,
    GumbelCopula,
    JoeCopula,
)
from rcopula.core.base import Copula, TailDependence
from rcopula.core.elliptical import (
    EllipticalCopula,
    GaussianCopula,
    P2p,
    StudentCopula,
    p2P,
)
from rcopula.core.empirical import EmpiricalCopula
from rcopula.core.extreme_value import (
    ExtremeValueCopula,
    GalambosCopula,
    HuslerReissCopula,
    TawnCopula,
    TEVCopula,
)
from rcopula.core.other import (
    FGMCopula,
    FrechetLowerCopula,
    FrechetUpperCopula,
    IndependenceCopula,
    MarshallOlkinCopula,
    PlackettCopula,
)
from rcopula.dependence import beta_n, cor_kendall, cor_spearman, pseudo_obs
from rcopula.distribution import CopulaDistribution, Margin
from rcopula.fit import CopulaFitResult, fit, loglik_copula
from rcopula.gof import GofResult, gof_statistic, gof_test
from rcopula.htest import TestResult, ev_test, exch_test, indep_test, rad_sym_test
from rcopula.select import SelectionResult, cross_validate, select_copula
from rcopula.transforms import conditional_cdf, conditional_ppf, rosenblatt

__version__ = "0.1.0.dev0"

__all__ = [
    "AMHCopula",
    "ArchimedeanCopula",
    "ClaytonCopula",
    "Copula",
    "CopulaDistribution",
    "CopulaFitResult",
    "EllipticalCopula",
    "EmpiricalCopula",
    "ExtremeValueCopula",
    "FGMCopula",
    "FrankCopula",
    "FrechetLowerCopula",
    "FrechetUpperCopula",
    "GalambosCopula",
    "GaussianCopula",
    "GofResult",
    "GumbelCopula",
    "HuslerReissCopula",
    "IndependenceCopula",
    "JoeCopula",
    "Margin",
    "MarshallOlkinCopula",
    "P2p",
    "PlackettCopula",
    "SelectionResult",
    "StudentCopula",
    "TEVCopula",
    "TailDependence",
    "TawnCopula",
    "TestResult",
    "__version__",
    "beta_n",
    "conditional_cdf",
    "conditional_ppf",
    "cor_kendall",
    "cor_spearman",
    "credit",
    "cross_validate",
    "derivatives",
    "ev_test",
    "exch_test",
    "fit",
    "garch",
    "gof_statistic",
    "gof_test",
    "indep_test",
    "insurance",
    "loglik_copula",
    "p2P",
    "portfolio",
    "pseudo_obs",
    "rad_sym_test",
    "risk",
    "rosenblatt",
    "select_copula",
]
