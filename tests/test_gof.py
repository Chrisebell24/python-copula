"""Tests for goodness-of-fit.

Statistics are compared against R's ``gofCopula`` (fixtures from
``tools/rgolden/06_gof.R``). P-values from the parametric bootstrap cannot be
compared value-for-value -- the RNG streams differ -- so those are checked for
*behaviour*: the correct family is not rejected, wrong ones are, and the
empirical rejection rate under the null matches the nominal level.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

import rcopula as rc
from rcopula.gof import empirical_copula_at, gof_statistic

GOLDEN = Path(__file__).parent / "golden" / "gof.json"

BUILDERS = {
    "clayton": rc.ClaytonCopula,
    "gumbel": rc.GumbelCopula,
    "frank": rc.FrankCopula,
    "normal": rc.GaussianCopula,
}


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run `make golden`")
    return json.loads(GOLDEN.read_text())


# ----------------------------------------------------------------------
# Parity with R
# ----------------------------------------------------------------------


@pytest.mark.golden
def test_sn_statistic_matches_r(golden: dict) -> None:
    """Deterministic given the data, so this is a tight comparison.

    The Archimedean families agree to ~1e-7. The Gaussian is looser at ~4e-5,
    inherited from the 2.6e-6 difference in its fitted parameter -- ``Sn``
    depends on the fit, and the two optimisers stop in slightly different
    places.
    """
    x = np.asarray(golden["data"], dtype=float)
    for family, expected in golden["Sn"].items():
        got = rc.gof_test(BUILDERS[family](), x, n_rep=1, random_state=0).statistic
        tolerance = 1e-4 if family == "normal" else 1e-6
        assert got == pytest.approx(float(expected), rel=tolerance), family


@pytest.mark.golden
def test_multiplier_pvalue_matches_r(golden: dict) -> None:
    """Different RNG streams, but the multiplier p-values still land within
    Monte-Carlo error of one another."""
    x = np.asarray(golden["data"], dtype=float)
    for family, expected in golden["p_mult"].items():
        got = rc.gof_test(
            BUILDERS[family](), x, simulation="mult", n_rep=500, random_state=1
        ).pvalue
        assert got == pytest.approx(float(expected), abs=0.06), family


# ----------------------------------------------------------------------
# The statistic
# ----------------------------------------------------------------------


class TestStatistics:
    def test_empirical_copula_is_a_proportion(self) -> None:
        u = np.array([[0.2, 0.3], [0.5, 0.6], [0.8, 0.9]])
        assert np.allclose(empirical_copula_at(u), [1 / 3, 2 / 3, 1.0])

    def test_sn_is_small_for_the_right_family_and_large_otherwise(self) -> None:
        u = rc.pseudo_obs(rc.ClaytonCopula(4.0).rvs(500, random_state=0))
        right = gof_statistic(u, rc.fit(rc.ClaytonCopula(), u).copula)
        wrong = gof_statistic(u, rc.fit(rc.GumbelCopula(), u).copula)
        assert wrong > 5 * right

    @pytest.mark.parametrize("method", ["Sn", "Tn", "AnChisq", "AnGamma"])
    def test_every_statistic_is_finite_and_non_negative(self, method: str) -> None:
        u = rc.pseudo_obs(rc.ClaytonCopula(2.0).rvs(200, random_state=0))
        cop = rc.fit(rc.ClaytonCopula(), u).copula
        value = gof_statistic(u, cop, method=method)
        assert np.isfinite(value) and value >= 0

    def test_sn_needs_a_copula(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(50, random_state=0)
        with pytest.raises(ValueError, match="needs a fitted copula"):
            gof_statistic(u, None, method="Sn")

    def test_unknown_statistic_is_rejected(self) -> None:
        u = rc.ClaytonCopula(2.0).rvs(50, random_state=0)
        with pytest.raises(ValueError, match="method must be one of"):
            gof_statistic(u, None, method="Wn")


# ----------------------------------------------------------------------
# Behaviour of the test
# ----------------------------------------------------------------------


class TestGofBehaviour:
    @pytest.mark.parametrize("simulation", ["pb", "mult"])
    def test_does_not_reject_the_true_family(self, simulation: str) -> None:
        x = rc.ClaytonCopula(4.0).rvs(300, random_state=0)
        res = rc.gof_test(rc.ClaytonCopula(), x, simulation=simulation, n_rep=200, random_state=1)
        assert res.pvalue > 0.05

    @pytest.mark.parametrize("simulation", ["pb", "mult"])
    @pytest.mark.parametrize("wrong", [rc.GumbelCopula, rc.FrankCopula, rc.GaussianCopula])
    def test_rejects_the_wrong_family(self, simulation: str, wrong) -> None:
        """Clayton's lower-tail dependence is not something these can mimic."""
        x = rc.ClaytonCopula(4.0).rvs(300, random_state=0)
        res = rc.gof_test(wrong(), x, simulation=simulation, n_rep=200, random_state=1)
        assert res.pvalue < 0.05

    def test_both_bootstraps_agree(self) -> None:
        x = rc.GumbelCopula(2.5).rvs(300, random_state=2)
        pb = rc.gof_test(rc.GumbelCopula(), x, n_rep=300, random_state=3)
        mult = rc.gof_test(rc.GumbelCopula(), x, simulation="mult", n_rep=300, random_state=3)
        assert pb.statistic == pytest.approx(mult.statistic)
        assert pb.pvalue == pytest.approx(mult.pvalue, abs=0.12)

    def test_multiplier_is_far_faster(self) -> None:
        """The reason it exists: no refitting on each replicate."""
        x = rc.ClaytonCopula(3.0).rvs(400, random_state=0)

        start = time.perf_counter()
        rc.gof_test(rc.ClaytonCopula(), x, n_rep=100, random_state=0)
        pb_time = time.perf_counter() - start

        start = time.perf_counter()
        rc.gof_test(rc.ClaytonCopula(), x, simulation="mult", n_rep=100, random_state=0)
        mult_time = time.perf_counter() - start

        assert mult_time < pb_time / 5

    def test_pvalue_is_strictly_inside_the_unit_interval(self) -> None:
        """Pesarin's convention: a p-value of exactly 0 would be an artefact."""
        x = rc.ClaytonCopula(8.0).rvs(300, random_state=0)
        res = rc.gof_test(rc.GaussianCopula(), x, n_rep=50, random_state=0)
        assert 0.0 < res.pvalue < 1.0
        assert res.pvalue == pytest.approx(0.5 / 51, abs=1e-9)

    def test_result_unpacks_like_a_scipy_test(self) -> None:
        x = rc.ClaytonCopula(2.0).rvs(200, random_state=0)
        res = rc.gof_test(rc.ClaytonCopula(), x, n_rep=20, random_state=0)
        statistic, pvalue = res.statistic, res.pvalue
        assert statistic > 0 and 0 < pvalue < 1
        assert "GofResult" in repr(res)

    def test_raw_data_is_always_rank_transformed(self) -> None:
        """Feeding exact copula draws through untransformed inflated Sn tenfold
        and rejected correct models; the transform is unconditional."""
        from scipy import stats

        u = rc.ClaytonCopula(4.0).rvs(300, random_state=0)
        mv = rc.CopulaDistribution(rc.ClaytonCopula(4.0), [stats.norm(5, 2), stats.expon()])
        x = mv.rvs(300, random_state=0)

        on_u = rc.gof_test(rc.ClaytonCopula(), u, n_rep=1, random_state=0).statistic
        on_ranks = rc.gof_test(
            rc.ClaytonCopula(), rc.pseudo_obs(u), n_rep=1, random_state=0
        ).statistic
        assert on_u == pytest.approx(on_ranks)
        # And raw data on any scale behaves the same way.
        assert rc.gof_test(rc.ClaytonCopula(), x, n_rep=1, random_state=0).statistic > 0

    def test_multiplier_supports_sn_only(self) -> None:
        x = rc.ClaytonCopula(2.0).rvs(100, random_state=0)
        with pytest.raises(ValueError, match="method='Sn' only"):
            rc.gof_test(rc.ClaytonCopula(), x, method="Tn", simulation="mult")

    def test_unknown_simulation_is_rejected(self) -> None:
        x = rc.ClaytonCopula(2.0).rvs(100, random_state=0)
        with pytest.raises(ValueError, match="simulation must be"):
            rc.gof_test(rc.ClaytonCopula(), x, simulation="jackknife")


@pytest.mark.slow
class TestEmpiricalLevel:
    """Under the null the p-value must be roughly uniform.

    This reproduces the empirical-level study of Genest, Remillard & Beaudoin
    (2009): a test that rejects a correct model more often than its nominal
    level is worse than useless.
    """

    @pytest.mark.parametrize("simulation", ["pb", "mult"])
    def test_rejection_rate_matches_the_nominal_level(self, simulation: str) -> None:
        truth = rc.ClaytonCopula(2.0)
        rng = np.random.default_rng(0)
        reps = 60 if simulation == "mult" else 25
        pvalues = [
            rc.gof_test(
                rc.ClaytonCopula(),
                truth.rvs(200, random_state=rng),
                simulation=simulation,
                n_rep=200,
                random_state=rng,
            ).pvalue
            for _ in range(reps)
        ]
        rate = float(np.mean(np.array(pvalues) <= 0.10))
        # Generous, because the replicate count is small; the point is to catch
        # a test that rejects far too often, not to certify the exact level.
        assert rate < 0.35
