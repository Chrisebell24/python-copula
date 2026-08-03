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


class TestDependogram:
    """Independence, decomposed subset by subset.

    The property that justifies the whole apparatus is that it *locates* the
    dependence rather than merely detecting it. So the central test uses a
    construction where a global test must reject and every pair must not:
    ``Z = (X + Y) mod 1`` is exactly independent of ``X`` and of ``Y``
    separately, and determined by them together.
    """

    @staticmethod
    def _pairwise_independent(n: int = 400, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x, y = rng.uniform(size=(2, n))
        return np.column_stack([x, y, (x + y) % 1.0])

    def test_it_finds_only_the_triple(self) -> None:
        from rcopula.htest import dependogram

        result = dependogram(self._pairwise_independent(), n_rep=200, random_state=1)
        assert result.significant() == [(0, 1, 2)]
        assert result.global_pvalue < 0.05

    def test_the_pairs_really_are_independent(self) -> None:
        # Guards the test above: if the pairs were dependent, finding only the
        # triple would be a failure rather than the point.
        data = self._pairwise_independent(2000)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            assert abs(float(rc.cor_kendall(data)[i, j])) < 0.05

    def test_independent_data_does_not_reject_globally(self) -> None:
        from rcopula.htest import dependogram

        # Deliberately *not* asserting that no individual subset rejects: with
        # four subsets tested at 5% the chance of at least one is 18%, so that
        # assertion would be a flake generator rather than a test. The global
        # p-value is the one that accounts for the multiplicity.
        for seed in (0, 3, 7):
            data = np.random.default_rng(seed).uniform(size=(300, 3))
            assert dependogram(data, n_rep=200, random_state=seed).global_pvalue > 0.05

    @pytest.mark.slow
    def test_both_levels_are_calibrated(self) -> None:
        """The claim the multiplicity correction rests on.

        A per-subset p-value should reject 5% of the time under independence,
        and so should the global one -- the correction has to remove the
        multiplicity without becoming conservative, which a Bonferroni bound
        would not.
        """
        from rcopula.htest import dependogram

        rejected_subsets = subsets_tested = rejected_global = 0
        trials = 40
        for seed in range(trials):
            data = np.random.default_rng(seed).uniform(size=(200, 3))
            result = dependogram(data, n_rep=200, random_state=seed + 100)
            rejected_subsets += int(np.sum(result.pvalues < 0.05))
            subsets_tested += len(result.subsets)
            rejected_global += int(result.global_pvalue < 0.05)
        # Binomial standard error at p = 0.05 is 1.7% for 160 subsets and 3.4%
        # for 40 datasets, so these bands are about three of them.
        assert 0.01 <= rejected_subsets / subsets_tested <= 0.11
        assert rejected_global / trials <= 0.20

    def test_it_finds_a_dependent_pair_and_leaves_the_third_alone(self) -> None:
        from rcopula.htest import dependogram

        rng = np.random.default_rng(0)
        pair = rc.ClaytonCopula(4.0).rvs(400, random_state=0)
        data = np.column_stack([pair, rng.uniform(size=400)])
        result = dependogram(data, n_rep=200, random_state=1)
        assert (0, 1) in result.significant()
        assert (0, 2) not in result.significant()
        assert (1, 2) not in result.significant()

    def test_every_subset_of_size_two_or_more_is_present(self) -> None:
        from rcopula.htest import dependogram

        data = np.random.default_rng(0).uniform(size=(150, 4))
        result = dependogram(data, n_rep=50, random_state=1)
        # 2^d - d - 1 = 11 for d = 4.
        assert len(result.subsets) == 11
        assert all(len(s) >= 2 for s in result.subsets)
        assert result.statistics.shape == result.pvalues.shape == (11,)

    def test_pvalues_are_strictly_inside_the_unit_interval(self) -> None:
        from rcopula.htest import dependogram

        data = rc.ClaytonCopula(3.0, dim=3).rvs(200, random_state=0)
        result = dependogram(data, n_rep=100, random_state=1)
        # Pesarin's estimator can never return exactly 0 or 1, which matters
        # when the p-values are combined or transformed downstream.
        assert np.all(result.pvalues > 0.0)
        assert np.all(result.pvalues < 1.0)
        assert 0.0 < result.global_pvalue < 1.0

    def test_summary_marks_the_significant_subsets(self) -> None:
        from rcopula.htest import dependogram

        result = dependogram(self._pairwise_independent(), n_rep=200, random_state=1)
        text = result.summary()
        assert "{0,1,2}" in text
        assert "*" in text
        assert "global p-value" in text

    def test_rejects_a_single_column(self) -> None:
        from rcopula.htest import dependogram

        with pytest.raises(ValueError, match="at least 2 columns"):
            dependogram(np.random.default_rng(0).uniform(size=(50, 1)), n_rep=10)

    def test_the_mobius_factorisation_matches_the_literal_sum(self) -> None:
        """The identity the implementation rests on.

        The Mobius transform is defined as an alternating sum over the 2^|A|
        subsets of A; the code evaluates a single product instead, because the
        sum is exactly that product expanded. If that ever stopped being true
        the statistic would be silently wrong, so it is checked directly.
        """
        import itertools as it

        from rcopula.htest.api import _mobius_matrices

        rng = np.random.default_rng(0)
        u = rng.uniform(size=(40, 3))
        subset = (0, 1, 2)

        literal = np.empty(u.shape[0])
        for i in range(u.shape[0]):
            total = 0.0
            for size in range(len(subset) + 1):
                for part in it.combinations(subset, size):
                    empirical = (
                        float(np.mean(np.all(u[:, list(part)] <= u[i, list(part)], axis=1)))
                        if part
                        else 1.0
                    )
                    rest = [j for j in subset if j not in part]
                    total += (
                        (-1) ** (len(subset) - len(part))
                        * empirical
                        * float(np.prod([u[i, j] for j in rest]))
                    )
            literal[i] = total

        matrices = _mobius_matrices(u)
        product = matrices[0] * matrices[1] * matrices[2]
        np.testing.assert_allclose(np.mean(product, axis=0), literal, atol=1e-14)


class TestSerialIndependence:
    """Serial dependence, located by lag structure rather than just detected.

    The case that justifies it over an autocorrelation function is a series
    that is uncorrelated in level and dependent in magnitude -- which is what
    financial returns are, and what a correlogram reports as clean.
    """

    @staticmethod
    def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        series = np.zeros(n)
        for t in range(1, n):
            series[t] = phi * series[t - 1] + rng.standard_normal()
        return series

    def test_white_noise_passes(self) -> None:
        from rcopula.htest import serial_indep_test

        noise = np.random.default_rng(0).standard_normal(600)
        assert serial_indep_test(noise, n_rep=200, random_state=1).global_pvalue > 0.05

    def test_an_ar1_is_detected(self) -> None:
        from rcopula.htest import serial_indep_test

        result = serial_indep_test(self._ar1(600, 0.6, 0), n_rep=200, random_state=1)
        assert result.global_pvalue < 0.05
        # Lag 1 is where an AR(1) lives, so that subset must be among those
        # rejected -- detecting *something* is not the claim.
        assert (0, 1) in result.significant()

    def test_it_is_invariant_to_an_increasing_transform(self) -> None:
        # The reason to use ranks: an autocorrelation would change completely.
        from rcopula.htest import serial_indep_test

        series = self._ar1(500, 0.6, 0)
        plain = serial_indep_test(series, n_rep=200, random_state=1)
        warped = serial_indep_test(np.exp(series / 2), n_rep=200, random_state=1)
        np.testing.assert_allclose(plain.statistics, warped.statistics, rtol=1e-12)

    def test_it_finds_dependence_a_correlogram_misses(self) -> None:
        from rcopula.htest import serial_indep_test

        rng = np.random.default_rng(3)
        volatility = np.exp(0.5 * rng.standard_normal(2000).cumsum() / 30)
        returns = volatility * rng.standard_normal(2000)
        # Uncorrelated in level...
        assert abs(float(np.corrcoef(returns[:-1], returns[1:])[0, 1])) < 0.06
        # ...and plainly dependent in magnitude.
        assert serial_indep_test(np.abs(returns), n_rep=200, random_state=1).global_pvalue < 0.05

    def test_the_embedding_dimension_sets_the_subsets(self) -> None:
        from rcopula.htest import serial_indep_test

        noise = np.random.default_rng(0).standard_normal(400)
        for lags, expected in ((2, 1), (3, 4), (4, 11)):
            result = serial_indep_test(noise, lags=lags, n_rep=50, random_state=1)
            assert len(result.subsets) == expected  # 2^lags - lags - 1

    def test_rejects_too_few_lags(self) -> None:
        from rcopula.htest import serial_indep_test

        with pytest.raises(ValueError, match="at least 2"):
            serial_indep_test(np.zeros(100), lags=1)

    def test_rejects_a_series_shorter_than_the_embedding(self) -> None:
        from rcopula.htest import serial_indep_test

        with pytest.raises(ValueError, match="more than 3 observations"):
            serial_indep_test(np.zeros(3), lags=3)
