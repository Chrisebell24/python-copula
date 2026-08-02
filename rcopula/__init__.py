"""rcopula - copula modelling in Python.

A full-featured replication of the R ``copula`` package (Hofert, Kojadinovic,
Maechler, Yan), implemented clean-room from the published literature and
verified numerically against R's outputs.

See ``NOTICE`` for the reference list and ``CONTRIBUTING.md`` for the
clean-room rule.
"""

from __future__ import annotations

from rcopula import (
    bootstrap,
    credit,
    datasets,
    derivatives,
    discrete,
    dynamic,
    garch,
    insurance,
    plots,
    portfolio,
    risk,
    sampling,
    serialize,
)
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
from rcopula.kendall import (
    kendall_cdf,
    kendall_empirical,
    kendall_pdf,
    kendall_ppf,
    kendall_return_period,
    kendall_rvs,
    return_period_level,
)
from rcopula.select import SelectionResult, cross_validate, select_copula
from rcopula.structural import (
    KhoudrajiCopula,
    MixtureCopula,
    NestedArchimedean,
    RotatedCopula,
    fit_nested,
    survival,
)
from rcopula.transforms import (
    conditional_cdf,
    conditional_ppf,
    htrafo,
    inverse_rosenblatt,
    radial_simplex,
    rosenblatt,
)
from rcopula.vine import VineCopula, fit_vine

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
    "KhoudrajiCopula",
    "Margin",
    "MarshallOlkinCopula",
    "MixtureCopula",
    "NestedArchimedean",
    "P2p",
    "PlackettCopula",
    "RotatedCopula",
    "SelectionResult",
    "StudentCopula",
    "TEVCopula",
    "TailDependence",
    "TawnCopula",
    "TestResult",
    "VineCopula",
    "__version__",
    "beta_n",
    "bootstrap",
    "conditional_cdf",
    "conditional_ppf",
    "cor_kendall",
    "cor_spearman",
    "credit",
    "cross_validate",
    "datasets",
    "derivatives",
    "discrete",
    "dynamic",
    "ev_test",
    "exch_test",
    "fit",
    "fit_nested",
    "fit_vine",
    "garch",
    "gof_statistic",
    "gof_test",
    "htrafo",
    "indep_test",
    "insurance",
    "inverse_rosenblatt",
    "kendall_cdf",
    "kendall_empirical",
    "kendall_pdf",
    "kendall_ppf",
    "kendall_return_period",
    "kendall_rvs",
    "loglik_copula",
    "p2P",
    "plots",
    "portfolio",
    "pseudo_obs",
    "rad_sym_test",
    "radial_simplex",
    "return_period_level",
    "risk",
    "rosenblatt",
    "sampling",
    "select_copula",
    "serialize",
    "survival",
]
