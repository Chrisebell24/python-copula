"""Regression tests for the multivariate normal / Student-t CDFs at extreme limits.

A Student-t copula with tiny ``df`` (StudentCopula allows ``df >= 0.01``) maps
ordinary probabilities to astronomically large t quantiles. The normal CDFs
underneath must treat those as the saturated limits they are: no overflow, and
no quadrature window stretched out to 1e40.
"""

from __future__ import annotations

import numpy as np
import pytest

import rcopula as rc
from rcopula.special.mvtnorm import bvn_cdf, mvt_cdf, tvn_cdf

EXCHANGEABLE = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])


class TestTinyDegreesOfFreedom:
    @pytest.mark.parametrize("df", [0.01, 0.02, 0.05])
    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_the_cdf_is_finite_and_a_probability(self, df: float, dim: int) -> None:
        # Used to raise "overflow encountered in multiply" (an error under the
        # suite's RuntimeWarning filter) for df <= 0.05.
        points = np.array([[0.3, 0.6, 0.5, 0.4][:dim], [0.9] * dim, [0.01] * dim])
        values = np.asarray(rc.StudentCopula(0.5, df=df, dim=dim).cdf(points))
        assert np.all(np.isfinite(values))
        assert np.all((values >= 0.0) & (values <= 1.0))

    def test_the_trivariate_value_is_right_not_just_finite(self) -> None:
        # The d = 3 route spread its nodes up to the huge limit and returned
        # 0.09 here; the true value sits between the d = 2 and d = 4 ones.
        cdf = {
            d: float(rc.StudentCopula(0.5, df=0.02, dim=d).cdf([[0.9] * d])[0]) for d in (2, 3, 4)
        }
        assert cdf[4] < cdf[3] < cdf[2]
        assert cdf[3] == pytest.approx(0.849, abs=5e-3)

    @pytest.mark.parametrize(("df", "tol"), [(0.02, 1e-3), (0.01, 1e-2)])
    def test_the_margin_identity_degrades_gracefully(self, df: float, tol: float) -> None:
        cop = rc.StudentCopula(0.5, df=df, dim=3)
        assert float(cop.cdf([[0.4, 1.0, 1.0]])[0]) == pytest.approx(0.4, abs=tol)

    def test_mvt_cdf_accepts_astronomical_limits(self) -> None:
        value = mvt_cdf([[1e300, -1e300, 0.0]], EXCHANGEABLE, df=0.5)[0]
        assert value == pytest.approx(0.0, abs=1e-12)


class TestSaturatedNormalLimits:
    def test_bvn_does_not_overflow_on_huge_limits(self) -> None:
        assert float(bvn_cdf(1e200, 0.3, 0.5)) == pytest.approx(float(bvn_cdf(50.0, 0.3, 0.5)))
        assert float(bvn_cdf(-1e200, 1e200, 0.5)) == 0.0

    @pytest.mark.parametrize("c", [30.0, 1e40])
    def test_a_huge_third_limit_marginalises_out(self, c: float) -> None:
        tri = tvn_cdf([[0.3, -0.2, c]], EXCHANGEABLE)[0]
        assert tri == pytest.approx(float(bvn_cdf(0.3, -0.2, 0.5)), abs=1e-13)
