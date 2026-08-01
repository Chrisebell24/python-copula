"""Tests for :mod:`rcopula.discrete`.

The identities that do the work here are exact, so they are asserted at machine
precision rather than with tolerances:

* the mass function sums to one over the lattice, and summing it over one
  coordinate returns the other margin -- so the copula and the margins really do
  compose into the distribution claimed;
* ``mixed_pdf`` reduces to the ordinary copula density when nothing is discrete
  and to ``discrete_pmf`` when everything is, so the general case is pinned at
  both ends;
* the distributional transform is *exactly* uniform, which is the property that
  separates it from jittering;
* ``tau_upper_bound`` is checked against the sample tau of an explicitly
  comonotone coupling, which is what it claims to compute.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.discrete import (
    checkerboard,
    discrete_loglik,
    discrete_pmf,
    distributional_transform,
    fit_discrete,
    mixed_pdf,
    tau_upper_bound,
)

LATTICE = np.array([[i, j] for i in range(45) for j in range(45)], dtype=float)


class TestDiscretePmf:
    @pytest.mark.parametrize(
        "copula",
        [
            rc.IndependenceCopula(2),
            rc.GaussianCopula(0.6),
            rc.GaussianCopula(-0.5),
            rc.ClaytonCopula(2.0),
            rc.GumbelCopula(1.8),
            rc.FrankCopula(4.0),
        ],
        ids=lambda c: type(c).__name__ + str(c.params),
    )
    def test_sums_to_one_over_the_lattice(self, copula: rc.Copula) -> None:
        margins = [stats.poisson(2.0), stats.poisson(3.0)]
        assert discrete_pmf(copula, LATTICE, margins).sum() == pytest.approx(1.0, abs=1e-9)

    def test_reproduces_both_margins(self) -> None:
        margins = [stats.poisson(2.0), stats.poisson(3.0)]
        mass = discrete_pmf(rc.ClaytonCopula(2.0), LATTICE, margins).reshape(45, 45)
        np.testing.assert_allclose(mass.sum(axis=1), margins[0].pmf(np.arange(45)), atol=1e-9)
        np.testing.assert_allclose(mass.sum(axis=0), margins[1].pmf(np.arange(45)), atol=1e-9)

    def test_independence_factorises_exactly(self) -> None:
        margins = [stats.poisson(2.0), stats.binom(8, 0.4)]
        x = np.array([[i, j] for i in range(12) for j in range(9)], dtype=float)
        mass = discrete_pmf(rc.IndependenceCopula(2), x, margins)
        expected = margins[0].pmf(x[:, 0]) * margins[1].pmf(x[:, 1])
        np.testing.assert_allclose(mass, expected, atol=1e-12)

    def test_never_negative(self) -> None:
        # A C-volume cannot be negative. Near-comonotone copulas are where the
        # inclusion-exclusion sum cancels hardest, so that is where to look.
        margins = [stats.poisson(1.0), stats.poisson(1.0)]
        for copula in (rc.ClaytonCopula(30.0), rc.GumbelCopula(20.0), rc.FrankCopula(35.0)):
            assert np.all(discrete_pmf(copula, LATTICE, margins) >= 0.0)

    def test_three_dimensions(self) -> None:
        margins = [stats.poisson(1.5)] * 3
        grid = np.array(
            [[i, j, k] for i in range(18) for j in range(18) for k in range(18)], dtype=float
        )
        mass = discrete_pmf(rc.GaussianCopula(0.4, dim=3, dispstr="ex"), grid, margins)
        assert mass.sum() == pytest.approx(1.0, abs=1e-8)

    def test_dependence_moves_mass_to_the_diagonal(self) -> None:
        margins = [stats.poisson(3.0), stats.poisson(3.0)]
        diagonal = np.column_stack([np.arange(15.0)] * 2)
        weak = discrete_pmf(rc.GaussianCopula(0.0), diagonal, margins).sum()
        strong = discrete_pmf(rc.GaussianCopula(0.8), diagonal, margins).sum()
        assert strong > 2 * weak

    def test_rejects_a_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="columns"):
            discrete_pmf(rc.GaussianCopula(0.5), np.zeros((4, 3)), [stats.poisson(1.0)] * 2)

    def test_rejects_the_wrong_number_of_margins(self) -> None:
        with pytest.raises(ValueError, match="margins"):
            discrete_pmf(rc.GaussianCopula(0.5), np.zeros((4, 2)), [stats.poisson(1.0)])


class TestMixedPdf:
    def test_reduces_to_the_copula_density_with_no_atoms(self) -> None:
        margins = [stats.norm(), stats.expon()]
        x = np.column_stack([np.linspace(-2, 2, 40), np.linspace(0.1, 4, 40)])
        mixed = mixed_pdf(rc.ClaytonCopula(2.0), x, margins, [False, False])
        reference = rc.CopulaDistribution(rc.ClaytonCopula(2.0), margins).pdf(x)
        np.testing.assert_allclose(mixed, reference, rtol=1e-10)

    def test_reduces_to_the_mass_function_with_no_continuous_part(self) -> None:
        margins = [stats.poisson(2.0), stats.poisson(3.0)]
        mixed = mixed_pdf(rc.GumbelCopula(1.7), LATTICE, margins, [True, True])
        np.testing.assert_allclose(
            mixed, discrete_pmf(rc.GumbelCopula(1.7), LATTICE, margins), rtol=1e-12
        )

    @pytest.mark.parametrize(
        "copula",
        [rc.GaussianCopula(0.5), rc.ClaytonCopula(2.0), rc.GumbelCopula(1.9), rc.FrankCopula(3.0)],
        ids=lambda c: type(c).__name__,
    )
    def test_integrates_and_sums_to_one(self, copula: rc.Copula) -> None:
        margins = [stats.norm(), stats.poisson(2.0)]
        grid = np.linspace(-9, 9, 3001)
        total = 0.0
        for count in range(25):
            rows = np.column_stack([grid, np.full_like(grid, float(count))])
            total += float(np.trapezoid(mixed_pdf(copula, rows, margins, [False, True]), grid))
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_recovers_the_continuous_margin(self) -> None:
        # Summing over the discrete coordinate must give the continuous density
        # back, whatever the copula does in between.
        margins = [stats.norm(), stats.poisson(2.0)]
        grid = np.linspace(-3, 3, 61)
        total = np.zeros_like(grid)
        for count in range(30):
            rows = np.column_stack([grid, np.full_like(grid, float(count))])
            total += mixed_pdf(rc.ClaytonCopula(2.0), rows, margins, [False, True])
        np.testing.assert_allclose(total, margins[0].pdf(grid), atol=1e-6)

    def test_recovers_the_discrete_margin(self) -> None:
        margins = [stats.norm(), stats.poisson(2.0)]
        grid = np.linspace(-10, 10, 4001)
        for count in (0, 1, 3, 6):
            rows = np.column_stack([grid, np.full_like(grid, float(count))])
            values = mixed_pdf(rc.GumbelCopula(2.0), rows, margins, [False, True])
            mass = float(np.trapezoid(values, grid))
            assert mass == pytest.approx(float(margins[1].pmf(count)), abs=1e-5)

    def test_the_conditioning_coordinate_is_not_transposed(self) -> None:
        # Invisible for an exchangeable copula and silently wrong for an
        # asymmetric one, so the check uses an asymmetric one.
        copula = rc.KhoudrajiCopula(
            rc.IndependenceCopula(2), rc.GumbelCopula(4.0), shapes=(0.4, 0.95)
        )
        margins = [stats.norm(), stats.poisson(2.0)]
        grid = np.linspace(-9, 9, 3001)
        total = 0.0
        for count in range(25):
            rows = np.column_stack([grid, np.full_like(grid, float(count))])
            total += float(np.trapezoid(mixed_pdf(copula, rows, margins, [False, True]), grid))
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_discrete_first_works_too(self) -> None:
        margins = [stats.poisson(2.0), stats.norm()]
        grid = np.linspace(-9, 9, 3001)
        total = 0.0
        for count in range(25):
            rows = np.column_stack([np.full_like(grid, float(count)), grid])
            values = mixed_pdf(rc.ClaytonCopula(2.0), rows, margins, [True, False])
            total += float(np.trapezoid(values, grid))
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_two_continuous_coordinates_are_refused_clearly(self) -> None:
        margins = [stats.norm(), stats.norm(), stats.poisson(1.0)]
        with pytest.raises(NotImplementedError, match="at most one continuous"):
            mixed_pdf(
                rc.GaussianCopula(0.3, dim=3, dispstr="ex"),
                np.zeros((3, 3)),
                margins,
                [False, False, True],
            )

    def test_rejects_a_bad_discrete_flag(self) -> None:
        with pytest.raises(ValueError, match="discrete must have"):
            mixed_pdf(rc.GaussianCopula(0.5), np.zeros((4, 2)), [stats.norm()] * 2, [True])


class TestCopulaDistributionWithAtoms:
    def test_discrete_margins_are_detected(self) -> None:
        joint = rc.CopulaDistribution(rc.GaussianCopula(0.5), [stats.norm(), stats.poisson(2.0)])
        np.testing.assert_array_equal(joint.discrete, [False, True])

    def test_pdf_delegates_to_mixed_pdf(self) -> None:
        margins = [stats.norm(), stats.poisson(2.0)]
        joint = rc.CopulaDistribution(rc.ClaytonCopula(2.0), margins)
        x = np.column_stack([np.linspace(-2, 2, 20), np.full(20, 3.0)])
        np.testing.assert_allclose(
            joint.pdf(x), mixed_pdf(rc.ClaytonCopula(2.0), x, margins, [False, True]), rtol=1e-12
        )

    def test_all_continuous_is_unchanged(self) -> None:
        margins = [stats.norm(), stats.expon()]
        joint = rc.CopulaDistribution(rc.GumbelCopula(2.0), margins)
        x = np.column_stack([np.linspace(-2, 2, 20), np.linspace(0.2, 4, 20)])
        u = np.column_stack([margins[0].cdf(x[:, 0]), margins[1].cdf(x[:, 1])])
        expected = rc.GumbelCopula(2.0).pdf(u) * margins[0].pdf(x[:, 0]) * margins[1].pdf(x[:, 1])
        np.testing.assert_allclose(joint.pdf(x), expected, rtol=1e-10)

    def test_sampling_gives_integers_on_the_discrete_coordinate(self) -> None:
        joint = rc.CopulaDistribution(rc.GaussianCopula(0.6), [stats.norm(), stats.poisson(2.0)])
        x = joint.rvs(500, random_state=0)
        assert np.all(x[:, 1] == np.floor(x[:, 1]))

    def test_a_margin_with_neither_pdf_nor_pmf_is_refused(self) -> None:
        class Broken:
            def cdf(self, x: object) -> object: ...
            def ppf(self, q: object) -> object: ...

        with pytest.raises(TypeError, match="pdf or pmf"):
            rc.CopulaDistribution(rc.GaussianCopula(0.5), [stats.norm(), Broken()])


class TestFitDiscrete:
    @pytest.mark.parametrize(
        ("copula", "truth"),
        [
            (rc.GaussianCopula(0.0), rc.GaussianCopula(0.6)),
            (rc.ClaytonCopula(1.0), rc.ClaytonCopula(2.5)),
            (rc.FrankCopula(1.0), rc.FrankCopula(5.0)),
        ],
        ids=lambda c: type(c).__name__,
    )
    def test_recovers_the_parameter(self, copula: rc.Copula, truth: rc.Copula) -> None:
        margins = [stats.poisson(4.0), stats.poisson(4.0)]
        x = rc.CopulaDistribution(truth, margins).rvs(3000, random_state=0)
        result = fit_discrete(x, copula, margins)
        assert result.converged
        assert result.params[0] == pytest.approx(truth.params[0], rel=0.12)

    def test_beats_independence_on_dependent_data(self) -> None:
        margins = [stats.poisson(4.0), stats.poisson(4.0)]
        x = rc.CopulaDistribution(rc.GaussianCopula(0.6), margins).rvs(2000, random_state=0)
        result = fit_discrete(x, rc.GaussianCopula(0.0), margins)
        statistic, pvalue = result.independence_test()
        assert result.loglik > result.independent_loglik
        assert statistic > 20 and pvalue < 1e-4

    def test_does_not_reject_independence_on_independent_data(self) -> None:
        margins = [stats.poisson(4.0), stats.poisson(4.0)]
        x = rc.CopulaDistribution(rc.IndependenceCopula(2), margins).rvs(2000, random_state=1)
        result = fit_discrete(x, rc.GaussianCopula(0.0), margins)
        assert result.independence_test()[1] > 0.05

    def test_information_criteria(self) -> None:
        margins = [stats.poisson(3.0), stats.poisson(3.0)]
        x = rc.CopulaDistribution(rc.GaussianCopula(0.5), margins).rvs(500, random_state=0)
        result = fit_discrete(x, rc.GaussianCopula(0.0), margins)
        assert result.n_params == 1
        assert result.aic == pytest.approx(2 - 2 * result.loglik)
        assert result.bic == pytest.approx(np.log(500) - 2 * result.loglik)

    def test_summary_carries_the_identifiability_caveat(self) -> None:
        margins = [stats.poisson(3.0), stats.poisson(3.0)]
        x = rc.CopulaDistribution(rc.GaussianCopula(0.5), margins).rvs(300, random_state=0)
        text = fit_discrete(x, rc.GaussianCopula(0.0), margins).summary()
        assert "identified only on the margins" in text
        assert "LR vs independence" in text

    def test_a_parameter_free_copula_short_circuits(self) -> None:
        margins = [stats.poisson(2.0), stats.poisson(2.0)]
        x = rc.CopulaDistribution(rc.IndependenceCopula(2), margins).rvs(200, random_state=0)
        result = fit_discrete(x, rc.IndependenceCopula(2), margins)
        assert result.converged
        assert result.loglik == result.independent_loglik
        assert result.independence_test()[0] == 0.0

    def test_binary_margins_still_fit(self) -> None:
        # The hardest identifiability case: three points of support per margin.
        margins = [stats.bernoulli(0.4), stats.bernoulli(0.5)]
        x = rc.CopulaDistribution(rc.GaussianCopula(0.7), margins).rvs(4000, random_state=0)
        result = fit_discrete(x, rc.GaussianCopula(0.0), margins)
        assert result.params[0] > 0.4  # direction and rough magnitude, not more


class TestDistributionalTransform:
    def test_is_exactly_uniform(self) -> None:
        margin = stats.poisson(3.0)
        x = margin.rvs(200_000, random_state=0).reshape(-1, 1).astype(float)
        u = distributional_transform(x, [margin], random_state=0)
        assert float(u.mean()) == pytest.approx(0.5, abs=0.005)
        assert float(u.var()) == pytest.approx(1 / 12, abs=0.002)
        assert stats.kstest(u.ravel(), "uniform").pvalue > 0.01

    def test_leaves_continuous_margins_alone(self) -> None:
        margin = stats.norm()
        x = margin.rvs(500, random_state=0).reshape(-1, 1)
        u = distributional_transform(x, [margin], random_state=0)
        np.testing.assert_allclose(u.ravel(), margin.cdf(x.ravel()), atol=1e-9)

    def test_preserves_the_sign_of_the_dependence(self) -> None:
        margins = [stats.poisson(3.0), stats.poisson(3.0)]
        x = rc.CopulaDistribution(rc.GaussianCopula(0.7), margins).rvs(4000, random_state=0)
        u = distributional_transform(x, margins, random_state=0)
        assert rc.cor_kendall(u)[0, 1] > 0.35

    def test_replicates_average(self) -> None:
        margins = [stats.poisson(2.0)]
        x = margins[0].rvs(2000, random_state=0).reshape(-1, 1).astype(float)
        single = distributional_transform(x, margins, random_state=0)
        many = distributional_transform(x, margins, random_state=0, replicates=50)
        # Averaging pulls each value towards the middle of its atom, so the
        # spread within an atom shrinks. That is the bias the docstring warns of.
        assert many.std() < single.std()

    def test_rejects_zero_replicates(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            distributional_transform(np.zeros((5, 1)), [stats.poisson(1.0)], replicates=0)

    def test_rejects_a_margin_mismatch(self) -> None:
        with pytest.raises(ValueError, match="margins"):
            distributional_transform(np.zeros((5, 2)), [stats.poisson(1.0)])


class TestTauUpperBound:
    @pytest.mark.parametrize(
        "margins",
        [
            [stats.bernoulli(0.1), stats.bernoulli(0.9)],
            [stats.bernoulli(0.3), stats.bernoulli(0.5)],
            [stats.poisson(1.0), stats.poisson(1.0)],
            [stats.poisson(3.0), stats.nbinom(4, 0.5)],
            [stats.binom(5, 0.3), stats.poisson(2.0)],
        ],
        ids=lambda m: f"{m[0].dist.name}-{m[1].dist.name}",
    )
    def test_matches_an_explicitly_comonotone_sample(self, margins: list) -> None:
        # The bound claims to be tau at the comonotone coupling. Build that
        # coupling from a shared uniform and measure it.
        v = np.random.default_rng(0).uniform(size=300_000)
        x, y = margins[0].ppf(v), margins[1].ppf(v)
        empirical = float(stats.kendalltau(x, y).statistic)
        assert tau_upper_bound(margins) == pytest.approx(empirical, abs=0.005)

    def test_identical_margins_reach_one(self) -> None:
        assert tau_upper_bound([stats.poisson(2.0)] * 2) == pytest.approx(1.0, abs=1e-6)

    def test_mismatched_bernoullis_are_capped_far_below_one(self) -> None:
        assert tau_upper_bound([stats.bernoulli(0.1), stats.bernoulli(0.9)]) < 0.12

    def test_no_fitted_copula_can_exceed_it(self) -> None:
        margins = [stats.bernoulli(0.1), stats.bernoulli(0.9)]
        x = rc.CopulaDistribution(rc.GaussianCopula(0.95), margins).rvs(20_000, random_state=0)
        observed = float(stats.kendalltau(x[:, 0], x[:, 1]).statistic)
        assert observed <= tau_upper_bound(margins) + 0.01

    def test_is_bivariate_only(self) -> None:
        with pytest.raises(ValueError, match="bivariate"):
            tau_upper_bound([stats.poisson(1.0)] * 3)


class TestCheckerboard:
    def test_is_a_probability_distribution(self) -> None:
        mass = checkerboard(rc.GaussianCopula(0.5), [stats.poisson(3.0)] * 2)
        assert mass.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(mass >= 0)

    def test_matches_the_mass_function_cell_by_cell(self) -> None:
        margins = [stats.poisson(2.0), stats.poisson(3.0)]
        mass = checkerboard(rc.ClaytonCopula(2.0), margins, support=20)
        direct = discrete_pmf(
            rc.ClaytonCopula(2.0),
            np.array([[i, j] for i in range(21) for j in range(21)], dtype=float),
            margins,
        ).reshape(21, 21)
        np.testing.assert_allclose(mass, direct, rtol=1e-12)

    def test_is_bivariate_only(self) -> None:
        with pytest.raises(ValueError, match="bivariate"):
            checkerboard(rc.GaussianCopula(0.3, dim=3, dispstr="ex"), [stats.poisson(1.0)] * 3)


class TestLoglik:
    def test_prefers_the_generating_copula(self) -> None:
        margins = [stats.poisson(3.0), stats.poisson(3.0)]
        x = rc.CopulaDistribution(rc.GaussianCopula(0.7), margins).rvs(2000, random_state=0)
        best = discrete_loglik(rc.GaussianCopula(0.7), x, margins)
        for wrong in (0.0, 0.3, 0.9):
            assert best > discrete_loglik(rc.GaussianCopula(wrong), x, margins)

    def test_impossible_data_gives_a_finite_floor(self) -> None:
        # A comonotone copula assigns essentially zero mass to an off-diagonal
        # cell. The result must be a very negative number, not -inf or a NaN.
        margins = [stats.poisson(2.0), stats.poisson(2.0)]
        value = discrete_loglik(rc.ClaytonCopula(40.0), np.array([[0.0, 12.0]]), margins)
        assert np.isfinite(value)
        assert value < -20
