"""Tests for the empirical copula and the copula-plus-margins distribution.

The empirical copula is compared against R's ``C.n`` for all three smoothings
(fixtures in ``tests/golden/empirical.json``); the joint distribution is
compared against R's ``pMvdc``/``dMvdc``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import rcopula as rc

GOLDEN = Path(__file__).parent / "golden" / "empirical.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run `make golden`")
    return json.loads(GOLDEN.read_text())


# ----------------------------------------------------------------------
# Parity with R
# ----------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.parametrize("smoothing", ["none", "beta", "checkerboard"])
def test_empirical_cdf_matches_r(golden: dict, smoothing: str) -> None:
    x = np.asarray(golden["data"], dtype=float)
    grid = np.asarray(golden["grid"], dtype=float)
    expected = np.asarray(golden[f"Cn_{smoothing}"], dtype=float)
    got = rc.EmpiricalCopula(x, smoothing=smoothing).cdf(grid)
    assert np.allclose(got, expected, rtol=0, atol=1e-13)


@pytest.mark.golden
def test_mvdc_matches_r(golden: dict) -> None:
    mv = rc.CopulaDistribution(rc.ClaytonCopula(2.0), [stats.norm(1, 2), stats.expon(scale=1 / 3)])
    pts = np.asarray(golden["mvdc_x"], dtype=float)
    assert np.allclose(mv.cdf(pts), np.asarray(golden["pMvdc"], dtype=float), rtol=1e-12)
    assert np.allclose(mv.pdf(pts), np.asarray(golden["dMvdc"], dtype=float), rtol=1e-12)


# ----------------------------------------------------------------------
# Empirical copula behaviour
# ----------------------------------------------------------------------


class TestEmpiricalCopula:
    @pytest.mark.parametrize("smoothing", ["none", "beta", "checkerboard"])
    def test_converges_to_the_truth(self, smoothing: str) -> None:
        truth = rc.ClaytonCopula(2.0)
        emp = rc.EmpiricalCopula(truth.rvs(4000, random_state=0), smoothing=smoothing)
        grid = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [0.3, 0.8]])
        assert np.max(np.abs(emp.cdf(grid) - truth.cdf(grid))) < 0.02

    @pytest.mark.parametrize("smoothing", ["beta", "checkerboard"])
    def test_smoothed_versions_are_genuine_copulas(self, smoothing: str) -> None:
        """The smoothed estimators satisfy the Frechet-Hoeffding bounds exactly.

        The *raw* estimator does not, and cannot: ``C_n(u, 1)`` is a step
        function of ``u``, not ``u`` itself, so it can exceed ``min(u, v)`` by up
        to about ``1/n``. That is exactly the defect the smoothings remove.
        """
        x = rc.GumbelCopula(2.0).rvs(500, random_state=1)
        emp = rc.EmpiricalCopula(x, smoothing=smoothing)
        rng = np.random.default_rng(0)
        u = rng.uniform(0.02, 0.98, size=(200, 2))
        c = emp.cdf(u)
        assert np.all(c >= np.maximum(u.sum(axis=1) - 1.0, 0.0) - 1e-12)
        assert np.all(c <= u.min(axis=1) + 1e-12)

    def test_raw_estimator_respects_the_bounds_only_up_to_1_over_n(self) -> None:
        n = 500
        x = rc.GumbelCopula(2.0).rvs(n, random_state=1)
        emp = rc.EmpiricalCopula(x)
        rng = np.random.default_rng(0)
        u = rng.uniform(0.02, 0.98, size=(200, 2))
        c = emp.cdf(u)
        assert np.all(c >= np.maximum(u.sum(axis=1) - 1.0, 0.0) - 1e-12)
        assert np.all(c <= u.min(axis=1) + 3.0 / n)

    def test_smoothing_reduces_error_at_small_samples(self) -> None:
        """The point of smoothing: the raw step function is the worst estimator."""
        truth = rc.ClaytonCopula(3.0)
        rng = np.random.default_rng(11)
        grid = rng.uniform(0.05, 0.95, size=(300, 2))
        exact = truth.cdf(grid)

        errors = {}
        for smoothing in ("none", "beta", "checkerboard"):
            total = 0.0
            for seed in range(12):
                x = truth.rvs(30, random_state=seed)
                emp = rc.EmpiricalCopula(x, smoothing=smoothing)
                total += float(np.mean((emp.cdf(grid) - exact) ** 2))
            errors[smoothing] = total / 12

        assert errors["beta"] < errors["none"]

    def test_only_beta_smoothing_has_a_density(self) -> None:
        x = rc.ClaytonCopula(2.0).rvs(300, random_state=2)
        grid = np.array([[0.3, 0.4], [0.6, 0.7]])
        assert np.all(rc.EmpiricalCopula(x, smoothing="beta").pdf(grid) > 0)
        for smoothing in ("none", "checkerboard"):
            with pytest.raises(NotImplementedError, match="density"):
                rc.EmpiricalCopula(x, smoothing=smoothing).pdf(grid)

    @pytest.mark.parametrize("smoothing", ["none", "beta", "checkerboard"])
    def test_sampling_has_uniform_margins(self, smoothing: str) -> None:
        x = rc.ClaytonCopula(2.0).rvs(1000, random_state=3)
        u = rc.EmpiricalCopula(x, smoothing=smoothing).rvs(20_000, random_state=4)
        for j in range(2):
            assert stats.kstest(u[:, j], "uniform").pvalue > 1e-4

    def test_dependence_measures_track_the_truth(self) -> None:
        truth = rc.GumbelCopula(3.0)
        emp = rc.EmpiricalCopula(truth.rvs(5000, random_state=5))
        assert emp.tau() == pytest.approx(truth.tau(), abs=0.02)
        assert emp.rho() == pytest.approx(truth.rho(), abs=0.03)
        assert emp.beta() == pytest.approx(truth.beta(), abs=0.04)

    def test_tail_dependence_is_indicative(self) -> None:
        """Clayton has lower but no upper tail dependence; the estimator should
        at least get the ordering right."""
        emp = rc.EmpiricalCopula(rc.ClaytonCopula(4.0).rvs(20_000, random_state=6))
        lam = emp.lambda_()
        assert lam.lower > lam.upper

    def test_partial_derivatives(self) -> None:
        """For the independence copula dC/du = v.

        The tolerance is loose on purpose: the Remillard-Scaillet estimator
        differences a step function over a window of width ``2/sqrt(n)``, so it
        is genuinely noisy. It is used inside the multiplier bootstrap, where
        that noise averages out.
        """
        x = rc.IndependenceCopula(2).rvs(20_000, random_state=7)
        d = rc.EmpiricalCopula(x).dCdu([[0.3, 0.5], [0.7, 0.5]])
        assert np.allclose(d[:, 0], 0.5, atol=0.06)

    def test_works_in_higher_dimensions(self) -> None:
        x = rc.ClaytonCopula(2.0, dim=4).rvs(1000, random_state=8)
        emp = rc.EmpiricalCopula(x)
        assert emp.dim == 4
        assert 0.0 < float(emp.cdf([[0.5] * 4])[0]) < 1.0

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="smoothing must be one of"):
            rc.EmpiricalCopula(np.zeros((10, 2)), smoothing="spline")
        with pytest.raises(ValueError, match="at least two observations"):
            rc.EmpiricalCopula(np.zeros((1, 2)))

    def test_describe_reports_sample_size_and_smoothing(self) -> None:
        text = rc.EmpiricalCopula(np.random.default_rng(0).uniform(size=(50, 2))).describe()
        assert "n=50" in text and "'none'" in text


# ----------------------------------------------------------------------
# CopulaDistribution
# ----------------------------------------------------------------------


class TestCopulaDistribution:
    def test_margins_and_dependence_are_independent_choices(self) -> None:
        """Same copula, different margins: dependence unchanged, margins not."""
        cop = rc.ClaytonCopula(2.0)
        a = rc.CopulaDistribution(cop, [stats.norm(0, 1), stats.norm(0, 1)])
        b = rc.CopulaDistribution(cop, [stats.gamma(2), stats.expon()])

        xa = a.rvs(20_000, random_state=0)
        xb = b.rvs(20_000, random_state=0)

        ta = stats.kendalltau(xa[:, 0], xa[:, 1]).statistic
        tb = stats.kendalltau(xb[:, 0], xb[:, 1]).statistic
        # Kendall's tau is invariant under increasing marginal transforms, so
        # these must agree essentially exactly -- not merely approximately.
        assert ta == pytest.approx(tb, abs=1e-12)
        assert ta == pytest.approx(cop.tau(), abs=0.02)

    def test_marginal_distributions_are_recovered(self) -> None:
        mv = rc.CopulaDistribution(
            rc.GaussianCopula(0.6, dim=3),
            [stats.norm(5, 2), stats.expon(scale=3), stats.gamma(2, scale=1.5)],
        )
        x = mv.rvs(40_000, random_state=1)
        for j, m in enumerate(mv.margins):
            assert stats.kstest(x[:, j], m.cdf).pvalue > 1e-4

    def test_cdf_matches_the_copula_of_the_transformed_point(self) -> None:
        cop = rc.FrankCopula(4.0)
        mv = rc.CopulaDistribution(cop, [stats.norm(), stats.expon()])
        pt = np.array([[0.3, 1.2]])
        u = np.column_stack([stats.norm.cdf(0.3), stats.expon.cdf(1.2)])
        assert mv.cdf(pt) == pytest.approx(cop.cdf(u))

    def test_pdf_factorises_as_copula_density_times_margins(self) -> None:
        cop = rc.GumbelCopula(2.0)
        mv = rc.CopulaDistribution(cop, [stats.norm(), stats.expon()])
        pt = np.array([[0.4, 0.9]])
        u = np.column_stack([stats.norm.cdf(0.4), stats.expon.cdf(0.9)])
        expected = cop.pdf(u) * stats.norm.pdf(0.4) * stats.expon.pdf(0.9)
        assert mv.pdf(pt) == pytest.approx(expected)

    def test_a_single_margin_is_broadcast(self) -> None:
        mv = rc.CopulaDistribution(rc.ClaytonCopula(2.0, dim=4), stats.norm())
        assert mv.dim == 4 and len(mv.margins) == 4

    def test_pandas_names_flow_through(self) -> None:
        mv = rc.CopulaDistribution(
            rc.ClaytonCopula(2.0), [stats.norm(), stats.expon()], names=["loss", "alae"]
        )
        out = mv.rvs(10, random_state=0)
        assert list(out.columns) == ["loss", "alae"]

    def test_rejects_mismatched_or_invalid_margins(self) -> None:
        with pytest.raises(ValueError, match="margin"):
            rc.CopulaDistribution(rc.ClaytonCopula(2.0, dim=3), [stats.norm()])
        with pytest.raises(TypeError, match="missing"):
            rc.CopulaDistribution(rc.ClaytonCopula(2.0), [stats.norm(), object()])
        with pytest.raises(TypeError, match="must be a Copula"):
            rc.CopulaDistribution("not a copula", [stats.norm(), stats.norm()])

    def test_logpdf_survives_where_pdf_underflows(self) -> None:
        """In higher dimensions the product of marginal densities underflows
        long before the joint density is genuinely zero.

        Heavy-tailed margins are needed to show this: with normal margins the
        marginal *CDF* saturates at 1.0 (already at ``x = 10``) before the
        density underflows, so the copula argument leaves the unit cube first.
        A t(2) margin has a polynomial tail, so its CDF stays below 1 far out
        while its density gets arbitrarily small.
        """
        mv = rc.CopulaDistribution(rc.IndependenceCopula(dim=25), stats.t(2))
        far = np.full((1, 25), 1e6)
        assert np.isfinite(mv.logpdf(far)[0])
        assert mv.logpdf(far)[0] < -745  # below log of the smallest normal float
        assert mv.pdf(far)[0] == 0.0

    def test_logpdf_is_the_copula_plus_the_margins(self) -> None:
        """The decomposition the log-space implementation relies on."""
        cop = rc.GaussianCopula(0.5, dim=3)
        mv = rc.CopulaDistribution(cop, stats.t(4))
        x = np.array([[0.3, -1.2, 2.0]])
        u = stats.t(4).cdf(x)
        expected = cop.logpdf(u)[0] + float(np.sum(stats.t(4).logpdf(x)))
        assert mv.logpdf(x)[0] == pytest.approx(expected, rel=1e-12)
