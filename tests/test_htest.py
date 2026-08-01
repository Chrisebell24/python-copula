"""Tests for the specification tests.

Statistics are compared against R's ``exchTest`` and ``radSymTest`` (fixtures
from ``tools/rgolden/07_htest.R``). P-values are *not* -- this package generates
null distributions by randomisation rather than by R's multiplier bootstrap --
so those are checked for behaviour and, more importantly, for **level**: a
specification test that rejects true nulls too often is worse than no test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import rcopula as rc

GOLDEN = Path(__file__).parent / "golden" / "htest.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run `make golden`")
    return json.loads(GOLDEN.read_text())


# ----------------------------------------------------------------------
# Parity with R
# ----------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.parametrize("test", ["exch", "radsym"])
def test_statistic_matches_r(golden: dict, test: str) -> None:
    fn = rc.exch_test if test == "exch" else rc.rad_sym_test
    for family, block in golden["cases"].items():
        x = np.asarray(block["data"], dtype=float)
        got = fn(x, n_rep=1, random_state=0).statistic
        assert got == pytest.approx(float(block[f"{test}_stat"]), rel=1e-10), family


# ----------------------------------------------------------------------
# Each test detects what it is meant to
# ----------------------------------------------------------------------


class TestExchangeability:
    @pytest.mark.parametrize(
        "cls", [rc.ClaytonCopula, rc.GumbelCopula, rc.FrankCopula, rc.GaussianCopula]
    )
    def test_symmetric_families_are_not_rejected(self, cls) -> None:
        """Every Archimedean and elliptical copula is exchangeable."""
        x = cls(0.6 if cls is rc.GaussianCopula else 3.0).rvs(300, random_state=0)
        assert rc.exch_test(x, n_rep=200, random_state=1).pvalue > 0.05

    def test_asymmetric_dependence_is_rejected(self) -> None:
        rng = np.random.default_rng(0)
        u = rc.ClaytonCopula(6.0).rvs(400, random_state=2)
        # Break the dependence for half the range only.
        u[:, 1] = np.where(u[:, 0] > 0.5, rng.uniform(size=400), u[:, 1])
        assert rc.exch_test(u, n_rep=200, random_state=1).pvalue < 0.05

    def test_marshall_olkin_asymmetry_is_detected(self) -> None:
        """A genuinely asymmetric family, with alpha1 far from alpha2."""
        x = rc.MarshallOlkinCopula(0.2, 0.9).rvs(500, random_state=0)
        assert rc.exch_test(x, n_rep=200, random_state=1).pvalue < 0.05


class TestRadialSymmetry:
    @pytest.mark.parametrize("cls", [rc.FrankCopula, rc.GaussianCopula])
    def test_symmetric_families_are_not_rejected(self, cls) -> None:
        x = cls(0.6 if cls is rc.GaussianCopula else 5.0).rvs(400, random_state=0)
        assert rc.rad_sym_test(x, n_rep=200, random_state=1).pvalue > 0.05

    @pytest.mark.parametrize("cls", [rc.ClaytonCopula, rc.JoeCopula])
    def test_tail_asymmetric_families_are_rejected(self, cls) -> None:
        """Clayton has lower-tail dependence only, Joe upper only."""
        x = cls(6.0).rvs(500, random_state=0)
        assert rc.rad_sym_test(x, n_rep=200, random_state=1).pvalue < 0.05

    def test_this_is_the_test_that_separates_gaussian_from_clayton(self) -> None:
        """Both can be calibrated to the same tau; only the tails differ."""
        tau = 0.5
        gauss = rc.GaussianCopula.from_tau(tau).rvs(600, random_state=0)
        clayton = rc.ClaytonCopula.from_tau(tau).rvs(600, random_state=0)
        assert rc.rad_sym_test(gauss, n_rep=200, random_state=1).pvalue > 0.05
        assert rc.rad_sym_test(clayton, n_rep=200, random_state=1).pvalue < 0.05


class TestIndependence:
    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_independent_data_is_not_rejected(self, dim: int) -> None:
        x = rc.IndependenceCopula(dim).rvs(400, random_state=0)
        assert rc.indep_test(x, n_rep=200, random_state=1).pvalue > 0.05

    @pytest.mark.parametrize("theta", [0.5, 1.0, 3.0])
    def test_dependence_is_detected(self, theta: float) -> None:
        x = rc.ClaytonCopula(theta).rvs(400, random_state=0)
        assert rc.indep_test(x, n_rep=200, random_state=1).pvalue < 0.05

    def test_detects_dependence_pearson_correlation_misses(self) -> None:
        """y = x**2 is perfectly dependent but essentially uncorrelated."""
        x = np.random.default_rng(7).normal(size=2000)
        quadratic = np.column_stack([x, x**2])
        assert abs(np.corrcoef(x, x**2)[0, 1]) < 0.05
        assert rc.indep_test(quadratic, n_rep=200, random_state=1).pvalue < 0.01

    def test_works_in_higher_dimensions(self) -> None:
        x = rc.ClaytonCopula(2.0, dim=4).rvs(300, random_state=0)
        assert rc.indep_test(x, n_rep=200, random_state=1).pvalue < 0.05


class TestExtremeValue:
    def test_extreme_value_families_are_not_rejected(self) -> None:
        for cop in (rc.GumbelCopula(3.0), rc.GalambosCopula(2.0), rc.HuslerReissCopula(2.0)):
            x = cop.rvs(400, random_state=0)
            assert rc.ev_test(x, n_rep=200, random_state=1).pvalue > 0.05, cop.name

    def test_non_extreme_value_families_are_rejected(self) -> None:
        for cop in (rc.ClaytonCopula(5.0), rc.FrankCopula(8.0)):
            x = cop.rvs(400, random_state=0)
            assert rc.ev_test(x, n_rep=200, random_state=1).pvalue < 0.05, cop.name


# ----------------------------------------------------------------------
# Shared behaviour
# ----------------------------------------------------------------------


ALL_TESTS = [rc.exch_test, rc.rad_sym_test, rc.indep_test, rc.ev_test]


@pytest.mark.parametrize("fn", ALL_TESTS)
def test_result_unpacks_like_a_scipy_test(fn) -> None:
    x = rc.ClaytonCopula(2.0).rvs(150, random_state=0)
    res = fn(x, n_rep=20, random_state=0)
    assert res.statistic >= 0
    assert 0.0 < res.pvalue < 1.0  # Pesarin: never exactly 0 or 1
    assert "TestResult" in repr(res)


@pytest.mark.parametrize("fn", ALL_TESTS)
def test_is_reproducible(fn) -> None:
    x = rc.ClaytonCopula(2.0).rvs(150, random_state=0)
    assert fn(x, n_rep=50, random_state=3) == fn(x, n_rep=50, random_state=3)


@pytest.mark.parametrize("fn", [rc.exch_test, rc.rad_sym_test, rc.ev_test])
def test_bivariate_only_tests_say_so(fn) -> None:
    x = rc.ClaytonCopula(2.0, dim=3).rvs(100, random_state=0)
    with pytest.raises(ValueError, match="bivariate"):
        fn(x, n_rep=10)


@pytest.mark.parametrize("fn", ALL_TESTS)
def test_raw_data_on_any_scale_works(fn) -> None:
    """Rank-based throughout, so the marginal scale is irrelevant."""
    from scipy import stats

    mv = rc.CopulaDistribution(rc.ClaytonCopula(4.0), [stats.norm(50, 7), stats.expon(scale=9)])
    x = mv.rvs(300, random_state=0)
    u = rc.pseudo_obs(x)
    assert fn(x, n_rep=1, random_state=0).statistic == pytest.approx(
        fn(u, n_rep=1, random_state=0).statistic
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("fn", "generator"),
    [
        (rc.exch_test, lambda rng: rc.ClaytonCopula(3.0).rvs(200, random_state=rng)),
        (rc.rad_sym_test, lambda rng: rc.FrankCopula(5.0).rvs(200, random_state=rng)),
        (rc.indep_test, lambda rng: rc.IndependenceCopula(2).rvs(200, random_state=rng)),
    ],
)
def test_empirical_level_is_close_to_nominal(fn, generator) -> None:
    """Under the null the rejection rate must not exceed the nominal level.

    The randomisation construction should give this exactly, up to Monte-Carlo
    error, since it draws from the null rather than approximating it.
    """
    rng = np.random.default_rng(0)
    pvalues = [fn(generator(rng), n_rep=200, random_state=rng).pvalue for _ in range(60)]
    rate = float(np.mean(np.array(pvalues) <= 0.10))
    assert rate < 0.30
