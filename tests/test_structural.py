"""Tests for copulas built from other copulas.

A structural construction is only worth having if the result is a *genuine*
copula, so most of these check that directly rather than checking agreement with
a formula:

* **uniform margins** -- ``C(u, 1, ..., 1) = u`` in every coordinate;
* **the density is the mixed derivative of the CDF**, by finite differences;
* **samples reproduce the theoretical tau**, which ties the sampler to the CDF;
* **tau and rho from the analytic shortcut match tanh-sinh quadrature of the
  copula itself**, which is an entirely independent route to the same number.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.core.measures import rho_by_quadrature, tau_by_quadrature
from rcopula.structural import KhoudrajiCopula, MixtureCopula, RotatedCopula, survival

GRID = np.array([[0.2, 0.7], [0.5, 0.5], [0.85, 0.15], [0.35, 0.9], [0.6, 0.4]])


def mixed_second_difference(cop: rc.Copula, x: float, y: float, h: float = 1e-5) -> float:
    corners = np.array([[x + h, y + h], [x + h, y - h], [x - h, y + h], [x - h, y - h]])
    c = cop.cdf(corners)
    return float((c[0] - c[1] - c[2] + c[3]) / (4.0 * h * h))


BASES = [rc.ClaytonCopula(3.0), rc.GumbelCopula(2.0), rc.FrankCopula(5.0), rc.GaussianCopula(0.6)]


class TestQuadratureMeasures:
    """The independent yardstick everything else is checked against."""

    @pytest.mark.parametrize("cop", [*BASES, rc.JoeCopula(3.0), rc.StudentCopula(0.5, df=4.0)])
    def test_reproduces_the_analytic_values(self, cop: rc.Copula) -> None:
        assert tau_by_quadrature(cop) == pytest.approx(cop.tau(), abs=1e-7)
        assert rho_by_quadrature(cop) == pytest.approx(cop.rho(), abs=1e-7)

    def test_handles_a_divergent_corner_density(self) -> None:
        """Gumbel's density blows up at (1, 1); Gauss-Legendre only manages 4e-4."""
        cop = rc.GumbelCopula(4.0)
        assert tau_by_quadrature(cop) == pytest.approx(cop.tau(), abs=1e-8)

    def test_independence_gives_zero(self) -> None:
        cop = rc.IndependenceCopula(2)
        assert tau_by_quadrature(cop) == pytest.approx(0.0, abs=1e-12)
        assert rho_by_quadrature(cop) == pytest.approx(0.0, abs=1e-12)

    def test_is_bivariate_only(self) -> None:
        with pytest.raises(ValueError, match="bivariate"):
            tau_by_quadrature(rc.ClaytonCopula(2.0, dim=3))


class TestRotatedIsACopula:
    @pytest.mark.parametrize("base", BASES)
    @pytest.mark.parametrize("degrees", [0, 90, 180, 270])
    def test_margins_are_uniform(self, base: rc.Copula, degrees: int) -> None:
        cop = RotatedCopula(base, degrees)
        u = np.linspace(0.02, 0.98, 13)
        assert np.allclose(cop.cdf(np.column_stack([u, np.ones_like(u)])), u, atol=1e-10)
        assert np.allclose(cop.cdf(np.column_stack([np.ones_like(u), u])), u, atol=1e-10)

    @pytest.mark.parametrize("base", BASES)
    @pytest.mark.parametrize("degrees", [90, 180, 270])
    def test_density_is_the_mixed_derivative_of_the_cdf(
        self, base: rc.Copula, degrees: int
    ) -> None:
        """Ties the inclusion-exclusion CDF to the reflected density."""
        cop = RotatedCopula(base, degrees)
        for x, y in GRID:
            assert cop.pdf([[x, y]])[0] == pytest.approx(
                mixed_second_difference(cop, x, y), rel=2e-4, abs=1e-5
            )

    @pytest.mark.parametrize("base", BASES)
    @pytest.mark.parametrize("degrees", [90, 180, 270])
    def test_the_cdf_respects_the_frechet_bounds(self, base: rc.Copula, degrees: int) -> None:
        cop = RotatedCopula(base, degrees)
        c = cop.cdf(GRID)
        assert np.all(c >= np.maximum(GRID.sum(axis=1) - 1.0, 0.0) - 1e-12)
        assert np.all(c <= GRID.min(axis=1) + 1e-12)

    @pytest.mark.parametrize("degrees", [0, 90, 180, 270])
    def test_samples_have_uniform_margins(self, degrees: int) -> None:
        sample = RotatedCopula(rc.ClaytonCopula(3.0), degrees).rvs(20_000, random_state=0)
        for j in range(2):
            assert stats.kstest(sample[:, j], "uniform").pvalue > 0.01

    @pytest.mark.parametrize("base", BASES)
    @pytest.mark.parametrize("degrees", [0, 90, 180, 270])
    def test_samples_reproduce_the_theoretical_tau(self, base: rc.Copula, degrees: int) -> None:
        cop = RotatedCopula(base, degrees)
        sample = cop.rvs(30_000, random_state=0)
        assert stats.kendalltau(sample[:, 0], sample[:, 1]).statistic == pytest.approx(
            cop.tau(), abs=0.015
        )


class TestRotatedDependence:
    @pytest.mark.parametrize("base", BASES)
    @pytest.mark.parametrize("degrees", [0, 90, 180, 270])
    def test_analytic_tau_and_rho_match_quadrature(self, base: rc.Copula, degrees: int) -> None:
        """The sign rule, checked against integrating the rotated copula itself."""
        cop = RotatedCopula(base, degrees)
        assert cop.tau() == pytest.approx(tau_by_quadrature(cop), abs=1e-6)
        assert cop.rho() == pytest.approx(rho_by_quadrature(cop), abs=1e-6)

    @pytest.mark.parametrize("degrees", [90, 270])
    def test_odd_reflections_reverse_concordance(self, degrees: int) -> None:
        base = rc.GumbelCopula(3.0)
        cop = RotatedCopula(base, degrees)
        assert cop.tau() == pytest.approx(-base.tau())
        assert cop.rho() == pytest.approx(-base.rho())

    def test_survival_preserves_concordance(self) -> None:
        base = rc.ClaytonCopula(3.0)
        assert survival(base).tau() == pytest.approx(base.tau())
        assert survival(base).rho() == pytest.approx(base.rho())

    def test_survival_swaps_the_tails(self) -> None:
        base = rc.ClaytonCopula(3.0)
        lower, upper = base.lambda_()
        assert lower > 0.0 and upper == 0.0
        assert survival(base).lambda_() == (upper, lower)

    def test_partial_reflection_removes_tail_dependence(self) -> None:
        cop = RotatedCopula(rc.ClaytonCopula(3.0), 90)
        assert cop.lambda_() == (0.0, 0.0)

    def test_the_counter_monotone_bound_is_the_exception(self) -> None:
        """Rotating W gives M, which is comonotone -- the one non-zero case."""
        assert RotatedCopula(rc.FrechetLowerCopula(2), 90).lambda_() == (1.0, 1.0)

    def test_partial_reflection_has_no_scalar_tau_above_two_dimensions(self) -> None:
        cop = RotatedCopula(rc.ClaytonCopula(2.0, dim=3), [True, False, False])
        with pytest.raises(NotImplementedError, match="no scalar value"):
            cop.tau()
        with pytest.raises(NotImplementedError, match="no scalar value"):
            cop.rho()

    def test_full_reflection_keeps_a_scalar_tau_above_two_dimensions(self) -> None:
        base = rc.ClaytonCopula(2.0, dim=3)
        assert survival(base).tau() == pytest.approx(base.tau())
        sample = survival(base).rvs(20_000, random_state=0)
        assert stats.kendalltau(sample[:, 0], sample[:, 1]).statistic == pytest.approx(
            base.tau(), abs=0.02
        )


class TestRotationAlgebra:
    def test_composition_follows_the_klein_group_not_the_cyclic_one(self) -> None:
        """Reflections, not rotations -- so composing 90 with 90 gives 0, not 180.

        The degree labels are the vine convention and the copulas they name are
        the right ones, but the underlying operation is a coordinate reflection,
        which is an involution. Two "90-degree rotations" therefore cancel; it
        takes a 90 and a 270 to reach 180.
        """
        base = rc.ClaytonCopula(3.0)
        assert RotatedCopula(RotatedCopula(base, 90), 90).degrees == 0
        assert RotatedCopula(RotatedCopula(base, 270), 270).degrees == 0
        assert RotatedCopula(RotatedCopula(base, 180), 180).degrees == 0
        assert RotatedCopula(RotatedCopula(base, 90), 270).degrees == 180
        assert RotatedCopula(RotatedCopula(base, 90), 180).degrees == 270

    def test_double_reflection_is_the_identity(self) -> None:
        base = rc.GumbelCopula(2.5)
        twice = RotatedCopula(RotatedCopula(base, True), True)
        assert not twice.flip.any()
        assert np.allclose(twice.cdf(GRID), base.cdf(GRID))
        assert np.allclose(twice.pdf(GRID), base.pdf(GRID))

    def test_composition_does_not_nest(self) -> None:
        """Collapsing keeps the tail-dependence bookkeeping non-recursive."""
        inner = RotatedCopula(rc.ClaytonCopula(3.0), 90)
        outer = RotatedCopula(inner, 90)
        assert not isinstance(outer.base, RotatedCopula)
        assert outer.lambda_() == inner.base.lambda_()

    def test_zero_rotation_is_the_base(self) -> None:
        base = rc.FrankCopula(4.0)
        cop = RotatedCopula(base, 0)
        assert np.allclose(cop.cdf(GRID), base.cdf(GRID))
        assert cop.lambda_() == base.lambda_()

    def test_flip_accepts_several_spellings(self) -> None:
        base = rc.ClaytonCopula(2.0)
        assert RotatedCopula(base, True).flip.tolist() == [True, True]
        assert RotatedCopula(base, 180).flip.tolist() == [True, True]
        assert RotatedCopula(base, [True, True]).flip.tolist() == [True, True]

    def test_rejects_bad_flips(self) -> None:
        base = rc.ClaytonCopula(2.0)
        with pytest.raises(ValueError, match="rotation must be one of"):
            RotatedCopula(base, 45)
        with pytest.raises(ValueError, match="flip has length"):
            RotatedCopula(base, [True, False, True])
        with pytest.raises(ValueError, match="degree convention is bivariate"):
            RotatedCopula(rc.ClaytonCopula(2.0, dim=3), 90)
        with pytest.raises(ValueError, match="degree convention is bivariate"):
            _ = survival(rc.ClaytonCopula(2.0, dim=3)).degrees

    def test_equality_and_hashing_respect_the_flip(self) -> None:
        base = rc.ClaytonCopula(3.0)
        assert RotatedCopula(base, 90) == RotatedCopula(base, 90)
        assert RotatedCopula(base, 90) != RotatedCopula(base, 270)
        assert len({RotatedCopula(base, d) for d in (0, 90, 180, 270)}) == 4

    def test_describe_names_the_rotation(self) -> None:
        assert "90-degree rotated" in RotatedCopula(rc.ClaytonCopula(3.0), 90).describe()
        assert "survival" in survival(rc.ClaytonCopula(3.0, dim=3)).describe()
        assert (
            "reflected on [0]"
            in RotatedCopula(rc.ClaytonCopula(3.0, dim=3), [True, False, False]).describe()
        )


class TestRotatedInference:
    @pytest.mark.parametrize("degrees", [0, 90, 180, 270])
    @pytest.mark.parametrize("method", ["mpl", "itau", "irho"])
    def test_estimation_recovers_the_parameter(self, degrees: int, method: str) -> None:
        truth = RotatedCopula(rc.ClaytonCopula(3.0), degrees)
        sample = truth.rvs(4000, random_state=0)
        res = rc.fit(RotatedCopula(rc.ClaytonCopula(), degrees), sample, method=method)
        assert res.params[0] == pytest.approx(3.0, rel=0.1)
        assert isinstance(res.copula, RotatedCopula)
        assert res.copula.degrees == degrees

    def test_calibration_targets_the_rotated_copula(self) -> None:
        cop = RotatedCopula.from_tau(-0.5, base=rc.ClaytonCopula, flip=90)
        assert cop.tau() == pytest.approx(-0.5)
        assert RotatedCopula.from_rho(-0.6, base=rc.ClaytonCopula, flip=270).rho() == (
            pytest.approx(-0.6)
        )

    def test_calibration_needs_a_base(self) -> None:
        with pytest.raises(TypeError, match="requires a base="):
            RotatedCopula.from_tau(0.5)
        with pytest.raises(TypeError, match="requires a base="):
            RotatedCopula.from_rho(0.5)

    def test_selection_picks_the_right_rotation(self) -> None:
        """The practical payoff: rotations slot into the family sweep."""
        truth = RotatedCopula(rc.ClaytonCopula(4.0), 180)
        sample = truth.rvs(3000, random_state=0)
        ranking = rc.select_copula(
            sample, families=[RotatedCopula(rc.ClaytonCopula(), d) for d in (0, 90, 180, 270)]
        )
        assert ranking.best.degrees == 180

    def test_goodness_of_fit_accepts_the_truth_and_rejects_a_rotation(self) -> None:
        truth = survival(rc.ClaytonCopula(4.0))
        sample = truth.rvs(500, random_state=0)
        good = rc.gof_test(
            RotatedCopula(rc.ClaytonCopula(), 180),
            sample,
            simulation="mult",
            n_rep=200,
            random_state=0,
        )
        bad = rc.gof_test(
            RotatedCopula(rc.ClaytonCopula(), 0),
            sample,
            simulation="mult",
            n_rep=200,
            random_state=0,
        )
        assert good.pvalue > 0.05
        assert bad.pvalue < 0.05


class TestSurvivalChangesTheAnswer:
    def test_survival_clayton_straddles_the_gaussian_on_upper_tail_risk(self) -> None:
        """Why the class exists.

        Three copulas at identical Kendall's tau, on identical exponential
        margins. Plain Clayton lands *below* the Gaussian on 99% expected
        shortfall and survival Clayton well above -- so "we used Clayton because
        we care about the tail" is, for loss aggregation, exactly backwards.
        """

        def es(cop: rc.Copula) -> float:
            losses = -np.log1p(-cop.rvs(200_000, random_state=0)).sum(axis=1)
            return float(rc.risk.expected_shortfall(losses, 0.99))

        plain = rc.ClaytonCopula.from_tau(0.5)
        assert es(plain) < es(rc.GaussianCopula.from_tau(0.5)) < es(survival(plain))
        assert survival(plain).tau() == pytest.approx(plain.tau())


# ======================================================================
# Khoudraji's device
# ======================================================================

KHOUDRAJI = [
    KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(3.0), [0.4, 0.95]),
    KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(2.0), [0.3, 0.9]),
    KhoudrajiCopula(rc.GumbelCopula(1.8), rc.GalambosCopula(2.0), [0.2, 0.8]),
    KhoudrajiCopula(rc.ClaytonCopula(2.0), rc.GumbelCopula(3.0), [0.35, 0.85]),
]


class TestKhoudrajiIsACopula:
    @pytest.mark.parametrize("cop", KHOUDRAJI)
    def test_margins_are_uniform(self, cop: rc.Copula) -> None:
        u = np.linspace(0.02, 0.98, 25)
        assert np.allclose(cop.cdf(np.column_stack([u, np.ones_like(u)])), u, atol=1e-12)
        assert np.allclose(cop.cdf(np.column_stack([np.ones_like(u), u])), u, atol=1e-12)

    @pytest.mark.parametrize("cop", KHOUDRAJI)
    def test_density_is_the_mixed_derivative_of_the_cdf(self, cop: rc.Copula) -> None:
        """Ties the product-rule density to the CDF it is supposed to differentiate."""
        for x, y in [(0.3, 0.4), (0.5, 0.5), (0.7, 0.75), (0.2, 0.85), (0.85, 0.2)]:
            assert cop.pdf([[x, y]])[0] == pytest.approx(
                mixed_second_difference(cop, x, y), rel=1e-5
            )

    @pytest.mark.parametrize("cop", KHOUDRAJI)
    def test_the_sampler_reproduces_the_cdf(self, cop: rc.Copula) -> None:
        """The max-of-two-independent-draws construction, checked against C itself."""
        sample = cop.rvs(200_000, random_state=0)
        pts = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.6], [0.35, 0.9], [0.9, 0.35]])
        empirical = np.array([np.mean((sample[:, 0] <= a) & (sample[:, 1] <= b)) for a, b in pts])
        assert np.allclose(cop.cdf(pts), empirical, atol=4e-3)
        for j in range(2):
            assert stats.kstest(sample[:, j], "uniform").pvalue > 0.01

    @pytest.mark.parametrize("cop", KHOUDRAJI)
    def test_tau_and_rho_match_the_sample(self, cop: rc.Copula) -> None:
        sample = cop.rvs(200_000, random_state=0)
        assert cop.tau() == pytest.approx(
            stats.kendalltau(sample[:, 0], sample[:, 1]).statistic, abs=5e-3
        )
        assert cop.rho() == pytest.approx(
            stats.spearmanr(sample[:, 0], sample[:, 1]).statistic, abs=5e-3
        )

    @pytest.mark.parametrize("cop", KHOUDRAJI)
    def test_the_cdf_respects_the_frechet_bounds(self, cop: rc.Copula) -> None:
        c = cop.cdf(GRID)
        assert np.all(c >= np.maximum(GRID.sum(axis=1) - 1.0, 0.0) - 1e-12)
        assert np.all(c <= GRID.min(axis=1) + 1e-12)


class TestKhoudrajiAsymmetry:
    def test_unequal_shapes_break_exchangeability(self) -> None:
        """The entire point: no Archimedean or exchangeable elliptical can do this."""
        cop = KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(3.0), [0.4, 0.95])
        assert not cop.is_exchangeable
        assert abs(cop.cdf([[0.3, 0.7]])[0] - cop.cdf([[0.7, 0.3]])[0]) > 0.01

    @pytest.mark.parametrize("shape", [0.1, 0.5, 0.9])
    def test_equal_shapes_leave_it_exchangeable(self, shape: float) -> None:
        cop = KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(3.0), [shape, shape])
        assert cop.is_exchangeable
        flipped = GRID[:, ::-1]
        assert np.allclose(cop.cdf(GRID), cop.cdf(flipped), atol=1e-14)

    @pytest.mark.parametrize("shape", [0.1, 0.5, 0.9])
    def test_two_identical_components_give_that_component_back(self, shape: float) -> None:
        """C1 = C2 = C means C(u^{1-a}) C(u^a) = C(u) for any equal shapes.

        A sharp self-consistency check: it must hold exactly, at every shape.
        """
        base = rc.GumbelCopula(3.0)
        cop = KhoudrajiCopula(base, base, [shape, shape])
        assert np.allclose(cop.cdf(GRID), base.cdf(GRID), atol=1e-14)
        assert cop.lambda_().upper == pytest.approx(base.lambda_().upper, abs=1e-12)

    def test_the_shapes_interpolate_between_the_components(self) -> None:
        first, second = rc.ClaytonCopula(2.0), rc.GumbelCopula(3.0)
        at_zero = KhoudrajiCopula(first, second, [1e-9, 1e-9])
        at_one = KhoudrajiCopula(first, second, [1 - 1e-9, 1 - 1e-9])
        assert np.allclose(at_zero.cdf(GRID), first.cdf(GRID), atol=1e-8)
        assert np.allclose(at_one.cdf(GRID), second.cdf(GRID), atol=1e-8)


class TestKhoudrajiExtremeValueStructure:
    """Both components extreme-value means the result is too -- so tail
    dependence is exact rather than estimated."""

    EV: ClassVar[list[rc.Copula]] = KHOUDRAJI[:3]

    @pytest.mark.parametrize("cop", EV)
    def test_the_pickands_function_reproduces_the_cdf(self, cop: rc.Copula) -> None:
        assert cop.is_extreme_value
        for u, v in [(0.4, 0.7), (0.2, 0.9), (0.6, 0.6), (0.95, 0.15)]:
            log_uv = np.log(u) + np.log(v)
            via_a = float(np.exp(log_uv * cop.pickands(np.log(v) / log_uv)))
            assert via_a == pytest.approx(float(cop.cdf([[u, v]])[0]), rel=1e-12)

    @pytest.mark.parametrize("cop", EV)
    def test_the_pickands_function_satisfies_its_defining_bounds(self, cop: rc.Copula) -> None:
        t = np.linspace(1e-6, 1 - 1e-6, 2001)
        a = cop.pickands(t)
        assert np.all(a >= np.maximum(t, 1 - t) - 1e-12)
        assert np.all(a <= 1.0 + 1e-12)
        assert np.all(np.diff(a, 2) >= -1e-12)  # convex

    @pytest.mark.parametrize("cop", EV)
    def test_tail_dependence_matches_the_diagonal_limit(self, cop: rc.Copula) -> None:
        """2(1 - A(1/2)) against lim (1 - 2u + C(u,u))/(1-u), computed from the CDF.

        Deliberately not against an empirical exceedance rate: that estimator is
        badly biased at any reachable quantile, and reads 0.42 where the truth
        is 0.377.
        """
        q = 1 - 1e-9
        limit = (1 - 2 * q + float(cop.cdf([[q, q]])[0])) / (1 - q)
        assert cop.lambda_().upper == pytest.approx(limit, abs=1e-6)
        assert cop.lambda_().lower == 0.0

    def test_a_non_extreme_value_component_refuses_rather_than_guesses(self) -> None:
        cop = KhoudrajiCopula(rc.ClaytonCopula(2.0), rc.GumbelCopula(3.0), [0.35, 0.85])
        assert not cop.is_extreme_value
        with pytest.raises(NotImplementedError, match="no closed form"):
            cop.lambda_()
        with pytest.raises(NotImplementedError, match="extreme-value"):
            cop.pickands(0.5)


class TestKhoudrajiPlumbing:
    def test_fitting_recovers_the_shapes(self) -> None:
        truth = KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(3.0), [0.4, 0.95])
        res = rc.fit(
            KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(), [0.5, 0.5]),
            truth.rvs(4000, random_state=0),
            method="mpl",
        )
        assert res.copula.copula2.theta == pytest.approx(3.0, rel=0.2)
        assert np.allclose(res.copula.shapes, [0.4, 0.95], atol=0.12)

    def test_it_works_above_two_dimensions_except_for_the_density(self) -> None:
        cop = KhoudrajiCopula(
            rc.IndependenceCopula(3), rc.GumbelCopula(3.0, dim=3), [0.3, 0.6, 0.9]
        )
        u = np.linspace(0.02, 0.98, 20)
        pts = np.ones((20, 3))
        pts[:, 1] = u
        assert np.allclose(cop.cdf(pts), u, atol=1e-12)
        sample = cop.rvs(20_000, random_state=0)
        for j in range(3):
            assert stats.kstest(sample[:, j], "uniform").pvalue > 0.01
        with pytest.raises(NotImplementedError, match="dim=2"):
            cop.pdf([[0.5, 0.5, 0.5]])

    def test_it_rejects_mismatched_input(self) -> None:
        with pytest.raises(ValueError, match="share a dimension"):
            KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(2.0, dim=3), [0.5, 0.5])
        with pytest.raises(ValueError, match="shapes has length"):
            KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(2.0), [0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="outside admissible range"):
            KhoudrajiCopula(rc.IndependenceCopula(2), rc.GumbelCopula(2.0), [0.5, 1.5])


# ======================================================================
# Mixtures
# ======================================================================


class TestMixtureIsACopula:
    MIXTURES: ClassVar[list[rc.Copula]] = [
        MixtureCopula([rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [0.4, 0.6]),
        MixtureCopula(
            [rc.ClaytonCopula(4.0), rc.GumbelCopula(3.0), rc.FrankCopula(5.0)], [0.3, 0.3, 0.4]
        ),
        MixtureCopula([rc.GaussianCopula(0.7), rc.GaussianCopula(-0.7)], [0.5, 0.5]),
    ]

    @pytest.mark.parametrize("cop", MIXTURES)
    def test_margins_are_uniform(self, cop: rc.Copula) -> None:
        u = np.linspace(0.02, 0.98, 25)
        assert np.allclose(cop.cdf(np.column_stack([u, np.ones_like(u)])), u, atol=1e-12)

    @pytest.mark.parametrize("cop", MIXTURES)
    def test_density_is_the_mixed_derivative_of_the_cdf(self, cop: rc.Copula) -> None:
        for x, y in [(0.3, 0.4), (0.5, 0.5), (0.7, 0.75)]:
            assert cop.pdf([[x, y]])[0] == pytest.approx(
                mixed_second_difference(cop, x, y), rel=1e-5
            )

    @pytest.mark.parametrize("cop", MIXTURES)
    def test_the_sampler_reproduces_the_cdf(self, cop: rc.Copula) -> None:
        sample = cop.rvs(200_000, random_state=0)
        pts = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.6]])
        empirical = np.array([np.mean((sample[:, 0] <= a) & (sample[:, 1] <= b)) for a, b in pts])
        assert np.allclose(cop.cdf(pts), empirical, atol=4e-3)
        for j in range(2):
            assert stats.kstest(sample[:, j], "uniform").pvalue > 0.01

    def test_it_works_above_two_dimensions(self) -> None:
        cop = MixtureCopula(
            [rc.GaussianCopula(0.5, dim=3), rc.ClaytonCopula(2.0, dim=3)], [0.6, 0.4]
        )
        u = np.linspace(0.02, 0.98, 20)
        pts = np.ones((20, 3))
        pts[:, 2] = u
        assert np.allclose(cop.cdf(pts), u, atol=1e-12)
        sample = cop.rvs(40_000, random_state=0)
        for j in range(3):
            assert stats.kstest(sample[:, j], "uniform").pvalue > 0.01


class TestMixtureDependenceIsLinearExceptWhereItIsNot:
    """Three measures mix exactly; Kendall's tau does not. Getting that backwards
    is the natural mistake, so both halves are pinned."""

    PARTS: ClassVar[list[rc.Copula]] = [rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)]
    WEIGHTS: ClassVar[list[float]] = [0.4, 0.6]

    def test_spearman_rho_is_the_weighted_average(self) -> None:
        cop = MixtureCopula(self.PARTS, self.WEIGHTS)
        expected = float(np.dot(self.WEIGHTS, [c.rho() for c in self.PARTS]))
        assert cop.rho() == pytest.approx(expected, abs=1e-12)

    def test_tail_dependence_is_the_weighted_average_in_both_tails(self) -> None:
        cop = MixtureCopula(self.PARTS, self.WEIGHTS)
        pairs = [c.lambda_() for c in self.PARTS]
        assert cop.lambda_().lower == pytest.approx(
            float(np.dot(self.WEIGHTS, [p.lower for p in pairs])), abs=1e-12
        )
        assert cop.lambda_().upper == pytest.approx(
            float(np.dot(self.WEIGHTS, [p.upper for p in pairs])), abs=1e-12
        )

    def test_blomqvist_beta_is_the_weighted_average(self) -> None:
        cop = MixtureCopula(self.PARTS, self.WEIGHTS)
        expected = float(np.dot(self.WEIGHTS, [c.beta() for c in self.PARTS]))
        assert cop.beta() == pytest.approx(expected, abs=1e-10)

    def test_kendall_tau_is_not(self) -> None:
        """Half comonotone, half independent: rho is exactly 0.5, tau is 0.416."""
        cop = MixtureCopula([rc.FrechetUpperCopula(2), rc.IndependenceCopula(2)], [0.5, 0.5])
        assert cop.rho() == pytest.approx(0.5, abs=1e-10)
        assert cop.tau() == pytest.approx(0.4159, abs=1e-3)
        assert abs(cop.tau() - 0.5) > 0.08

    @pytest.mark.parametrize(
        ("parts", "weights"),
        [
            ([rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [0.4, 0.6]),
            ([rc.ClaytonCopula(8.0), rc.GumbelCopula(1.05)], [0.5, 0.5]),
            ([rc.ClaytonCopula(6.0), rc.FrankCopula(-8.0)], [0.5, 0.5]),
        ],
    )
    def test_the_quadrature_tau_matches_the_sample(self, parts: list, weights: list) -> None:
        cop = MixtureCopula(parts, weights)
        sample = cop.rvs(200_000, random_state=0)
        assert cop.tau() == pytest.approx(
            stats.kendalltau(sample[:, 0], sample[:, 1]).statistic, abs=5e-3
        )

    def test_a_mixture_can_have_dependence_in_both_tails(self) -> None:
        """The reason to build one: no single family here manages it."""
        cop = MixtureCopula([rc.ClaytonCopula(4.0), rc.GumbelCopula(3.0)], [0.5, 0.5])
        lam = cop.lambda_()
        assert lam.lower > 0.3
        assert lam.upper > 0.3
        for part in cop.copulas:
            assert min(part.lambda_()) == 0.0


class TestMixturePlumbing:
    def test_the_weights_round_trip_through_the_log_odds_scale(self) -> None:
        for weights in ([0.25, 0.75], [0.1, 0.2, 0.7], [0.5, 0.5], [1e-8, 1 - 1e-8]):
            parts = [rc.ClaytonCopula(2.0)] * len(weights)
            assert np.allclose(MixtureCopula(parts, weights).weights, weights, atol=1e-9)

    def test_the_log_odds_scale_is_unconstrained(self) -> None:
        """Which is the point: an optimiser sees a box, not a simplex."""
        cop = MixtureCopula([rc.ClaytonCopula(2.0), rc.GumbelCopula(2.0)], [0.3, 0.7])
        lo, hi = cop.param_bounds[-1]
        assert lo < 0 < hi
        moved = cop.with_params([*cop.params[:-1], 5.0])
        assert np.isclose(moved.weights.sum(), 1.0)
        assert moved.weights[0] > 0.99

    def test_fitting_recovers_the_weights(self) -> None:
        truth = MixtureCopula([rc.ClaytonCopula(4.0), rc.GumbelCopula(3.0)], [0.3, 0.7])
        res = rc.fit(
            MixtureCopula([rc.ClaytonCopula(), rc.GumbelCopula()], [0.5, 0.5]),
            truth.rvs(4000, random_state=0),
            method="mpl",
        )
        assert np.allclose(res.copula.weights, [0.3, 0.7], atol=0.12)

    def test_a_degenerate_weight_reduces_to_the_other_component(self) -> None:
        cop = MixtureCopula([rc.ClaytonCopula(3.0), rc.GumbelCopula(2.5)], [1e-14, 1 - 1e-14])
        assert np.allclose(cop.cdf(GRID), rc.GumbelCopula(2.5).cdf(GRID), atol=1e-12)

    def test_it_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="at least two components"):
            MixtureCopula([rc.ClaytonCopula(2.0)])
        with pytest.raises(ValueError, match="share a dimension"):
            MixtureCopula([rc.ClaytonCopula(2.0), rc.ClaytonCopula(2.0, dim=3)])
        with pytest.raises(ValueError, match="must sum to 1"):
            MixtureCopula([rc.ClaytonCopula(2.0), rc.GumbelCopula(2.0)], [0.5, 0.7])
        with pytest.raises(ValueError, match="non-negative"):
            MixtureCopula([rc.ClaytonCopula(2.0), rc.GumbelCopula(2.0)], [-0.5, 1.5])
        with pytest.raises(ValueError, match="weights for"):
            MixtureCopula([rc.ClaytonCopula(2.0), rc.GumbelCopula(2.0)], [0.2, 0.3, 0.5])


class TestMarginalCopula:
    """Lower-dimensional margins.

    The defining property is that the margin's CDF equals the parent's with the
    dropped coordinates set to one. Everything else -- which class comes back,
    what happens to a correlation structure -- follows from getting that right,
    so it is checked for every supported family rather than argued about.
    """

    CASES: ClassVar[list] = [
        (rc.ClaytonCopula(2.0, dim=5), [1, 4]),
        (rc.GumbelCopula(2.5, dim=4), [0, 2, 3]),
        (rc.FrankCopula(6.0, dim=4), [1, 2]),
        (rc.JoeCopula(2.0, dim=3), [0, 2]),
        (
            rc.GaussianCopula(
                [0.72, 0.63, 0.54, 0.45, 0.56, 0.48, 0.40, 0.42, 0.35, 0.30],
                dim=5,
                dispstr="un",
            ),
            [0, 3],
        ),
        (rc.GaussianCopula(0.6, dim=4, dispstr="ex"), [1, 2, 3]),
        (rc.GaussianCopula(0.8, dim=4, dispstr="ar1"), [0, 2, 3]),
        (rc.StudentCopula(0.5, df=5.0, dim=4, dispstr="ex"), [0, 1]),
        (rc.RotatedCopula(rc.ClaytonCopula(2.0, dim=3), [True, False, True]), [0, 2]),
        (
            rc.MixtureCopula(
                [rc.ClaytonCopula(3.0, dim=3), rc.GumbelCopula(2.0, dim=3)], weights=[0.4, 0.6]
            ),
            [0, 1],
        ),
        (rc.IndependenceCopula(4), [0, 2]),
    ]

    @pytest.mark.parametrize(("copula", "indices"), CASES, ids=lambda v: str(v)[:28])
    def test_the_cdf_matches_the_parent_with_the_rest_at_one(
        self, copula: rc.Copula, indices: list[int]
    ) -> None:
        margin = rc.marginal_copula(copula, indices)
        points = np.random.default_rng(0).uniform(0.05, 0.95, size=(200, len(indices)))
        full = np.ones((200, copula.dim))
        full[:, indices] = points
        # 1e-6 rather than machine precision only because the *parent's* CDF
        # goes through Genz-Bretz above three dimensions.
        assert np.max(np.abs(np.asarray(margin.cdf(points)) - np.asarray(copula.cdf(full)))) < 1e-6

    @pytest.mark.parametrize(("copula", "indices"), CASES, ids=lambda v: str(v)[:28])
    def test_the_dimension_is_what_was_asked_for(
        self, copula: rc.Copula, indices: list[int]
    ) -> None:
        assert rc.marginal_copula(copula, indices).dim == len(indices)

    def test_the_concrete_class_survives(self) -> None:
        # Returning a bare ArchimedeanCopula would still be mathematically
        # right and would break isinstance checks and the serialization
        # registry, which keys on the class name.
        margin = rc.marginal_copula(rc.ClaytonCopula(2.0, dim=5), [1, 4])
        assert type(margin) is rc.ClaytonCopula
        assert isinstance(margin, rc.ClaytonCopula)

    def test_an_archimedean_margin_keeps_its_parameter(self) -> None:
        parent = rc.GumbelCopula(2.5, dim=5)
        margin = rc.marginal_copula(parent, [0, 2, 4])
        assert float(margin.params[0]) == 2.5
        assert margin.tau() == pytest.approx(parent.tau())

    def test_an_unstructured_margin_is_the_sub_matrix(self) -> None:
        parent = rc.GaussianCopula(
            [0.72, 0.63, 0.54, 0.45, 0.56, 0.48, 0.40, 0.42, 0.35, 0.30], dim=5, dispstr="un"
        )
        chosen = [0, 2, 4]
        margin = rc.marginal_copula(parent, chosen)
        np.testing.assert_allclose(
            margin.sigma(), np.asarray(parent.sigma())[np.ix_(chosen, chosen)]
        )

    def test_a_gapped_ar1_margin_is_unstructured(self) -> None:
        # Dropping a coordinate from an AR(1) chain leaves a gap, so the result
        # is not AR(1) and must not claim to be.
        parent = rc.GaussianCopula(0.8, dim=4, dispstr="ar1")
        margin = rc.marginal_copula(parent, [0, 2, 3])
        assert margin.dispstr == "un"
        np.testing.assert_allclose(
            margin.sigma(), np.asarray(parent.sigma())[np.ix_([0, 2, 3], [0, 2, 3])]
        )

    def test_a_student_margin_keeps_its_degrees_of_freedom(self) -> None:
        margin = rc.marginal_copula(rc.StudentCopula(0.5, df=3.5, dim=4, dispstr="ex"), [0, 2])
        assert isinstance(margin, rc.StudentCopula)
        assert float(margin.df) == 3.5

    def test_order_matters_for_an_asymmetric_copula(self) -> None:
        parent = rc.RotatedCopula(rc.ClaytonCopula(3.0, dim=3), [True, False, False])
        forward = rc.marginal_copula(parent, [0, 1])
        reversed_ = rc.marginal_copula(parent, [1, 0])
        point = np.array([[0.3, 0.8]])
        assert not np.allclose(forward.cdf(point), reversed_.cdf(point))

    def test_asking_for_everything_returns_the_same_object(self) -> None:
        parent = rc.ClaytonCopula(2.0, dim=3)
        assert rc.marginal_copula(parent, [0, 1, 2]) is parent

    def test_rejects_too_few_coordinates(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            rc.marginal_copula(rc.ClaytonCopula(2.0, dim=3), [1])

    def test_rejects_repeated_coordinates(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            rc.marginal_copula(rc.ClaytonCopula(2.0, dim=3), [1, 1])

    def test_rejects_out_of_range_coordinates(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 3\)"):
            rc.marginal_copula(rc.ClaytonCopula(2.0, dim=3), [0, 5])

    def test_the_empirical_copula_says_what_to_do_instead(self) -> None:
        data = rc.GaussianCopula(0.5, dim=3, dispstr="ex").rvs(50, random_state=0)
        with pytest.raises(NotImplementedError, match=r"data\[:, indices\]"):
            rc.marginal_copula(rc.EmpiricalCopula(data), [0, 1])

    @staticmethod
    def _tree() -> rc.NestedArchimedean:
        return rc.NestedArchimedean(
            rc.ClaytonCopula(1.0),
            components=[0],
            children=[rc.NestedArchimedean(rc.ClaytonCopula(4.0), components=[1, 2, 3])],
        )

    @pytest.mark.parametrize(("i", "j"), [(0, 1), (0, 3), (1, 2), (2, 3)])
    def test_a_nested_margin_is_the_lowest_common_ancestor(self, i: int, j: int) -> None:
        # Two variables meet at exactly one node, and that node's generator is
        # their bivariate copula -- the same fact tau_matrix rests on, so the
        # two must agree exactly rather than approximately.
        tree = self._tree()
        margin = rc.marginal_copula(tree, [i, j])
        assert margin.tau() == pytest.approx(float(tree.tau_matrix()[i, j]), abs=1e-12)

    @pytest.mark.parametrize(("i", "j"), [(0, 1), (1, 2)])
    def test_a_nested_margin_reproduces_the_tree_cdf(self, i: int, j: int) -> None:
        tree = self._tree()
        margin = rc.marginal_copula(tree, [i, j])
        points = np.random.default_rng(0).uniform(0.05, 0.95, size=(300, 2))
        full = np.ones((300, 4))
        full[:, [i, j]] = points
        assert np.max(np.abs(np.asarray(margin.cdf(points)) - np.asarray(tree.cdf(full)))) < 1e-12

    def test_a_nested_margin_of_three_says_what_it_would_take(self) -> None:
        with pytest.raises(NotImplementedError, match="induced sub-tree"):
            rc.marginal_copula(self._tree(), [0, 1, 2])

    def test_an_unsupported_family_is_refused_clearly(self) -> None:
        vine = rc.fit_vine(
            rc.GaussianCopula(0.5, dim=4, dispstr="ex").rvs(200, random_state=0), structure="C"
        )
        with pytest.raises(NotImplementedError, match="not available"):
            rc.marginal_copula(vine, [0, 1])

    def test_a_bivariate_only_family_has_no_margin_to_take(self) -> None:
        # The range check catches this first, which is the right error: there
        # is no third coordinate to drop.
        with pytest.raises(ValueError, match=r"\[0, 2\)"):
            rc.marginal_copula(rc.PlackettCopula(4.0), [0, 1, 2])


class TestOuterPower:
    """The outer power transformation.

    Three checks carry the weight. It reduces to the base copula at alpha = 1,
    so the transformation is the identity where it should be. Applied to the
    independence generator it *is* the Gumbel copula, which pins the whole
    construction against a family implemented independently. And Nelsen's
    closed form for Kendall's tau is confirmed against a quadrature that knows
    nothing about it.
    """

    CASES: ClassVar[list] = [
        (rc.ClaytonCopula(2.0), 1.5),
        (rc.ClaytonCopula(0.8), 2.5),
        (rc.FrankCopula(4.0), 2.0),
        (rc.GumbelCopula(1.6), 1.8),
        (rc.JoeCopula(2.0), 1.4),
    ]

    @pytest.mark.parametrize(("base", "alpha"), CASES, ids=lambda v: str(v)[:22])
    def test_alpha_one_is_the_identity(self, base: rc.Copula, alpha: float) -> None:
        del alpha
        points = np.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.2], [0.05, 0.95]])
        np.testing.assert_allclose(rc.opower(base, 1.0).cdf(points), base.cdf(points), rtol=1e-12)

    @pytest.mark.parametrize("alpha", [1.5, 2.0, 3.0, 4.5])
    def test_applied_to_independence_it_is_gumbel(self, alpha: float) -> None:
        # psi(t) = exp(-t) gives psi(t^(1/a)) = exp(-t^(1/a)), which is Gumbel's
        # generator exactly. Clayton at theta -> 0 is the independence limit.
        points = np.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.2], [0.15, 0.42]])
        lifted = rc.opower(rc.ClaytonCopula(1e-10), alpha).cdf(points)
        np.testing.assert_allclose(lifted, rc.GumbelCopula(alpha).cdf(points), atol=1e-6)

    @pytest.mark.parametrize(("base", "alpha"), CASES, ids=lambda v: str(v)[:22])
    def test_the_tau_closed_form(self, base: rc.Copula, alpha: float) -> None:
        # Nelsen: tau_alpha = 1 - (1 - tau_base)/alpha, checked against the
        # definition tau = 1 - 4 int int dC/du dC/dv, which is derived from the
        # CDF and knows nothing about the formula.
        from numpy.polynomial.legendre import leggauss

        copula = rc.opower(base, alpha)
        n = 260
        nodes, weights = leggauss(n)
        grid, weight = 0.5 * (nodes + 1.0), 0.5 * weights
        first, second = np.meshgrid(grid, grid, indexing="ij")
        points = np.column_stack([first.ravel(), second.ravel()])
        du = np.asarray(rc.conditional_cdf(copula, points, 0)).reshape(n, n)
        dv = np.asarray(rc.conditional_cdf(copula, points, 1)).reshape(n, n)
        quadrature = 1.0 - 4.0 * float(weight @ (du * dv) @ weight)
        assert copula.tau() == pytest.approx(quadrature, abs=1e-5)
        assert copula.tau() == pytest.approx(1 - (1 - base.tau()) / alpha, abs=1e-12)

    @pytest.mark.parametrize(("base", "alpha"), CASES, ids=lambda v: str(v)[:22])
    def test_the_density_integrates_to_one(self, base: rc.Copula, alpha: float) -> None:
        from numpy.polynomial.legendre import leggauss

        copula = rc.opower(base, alpha)
        n = 200
        nodes, weights = leggauss(n)
        grid, weight = 0.5 * (nodes + 1.0), 0.5 * weights
        first, second = np.meshgrid(grid, grid, indexing="ij")
        density = np.asarray(copula.pdf(np.column_stack([first.ravel(), second.ravel()]))).reshape(
            n, n
        )
        assert float(weight @ density @ weight) == pytest.approx(1.0, abs=2e-3)

    @pytest.mark.parametrize(("base", "alpha"), CASES, ids=lambda v: str(v)[:22])
    def test_the_density_is_the_mixed_derivative_of_the_cdf(
        self, base: rc.Copula, alpha: float
    ) -> None:
        copula = rc.opower(base, alpha)
        step = 1e-4
        for first, second in ((0.3, 0.4), (0.55, 0.62), (0.8, 0.25)):

            def corner(a: float, b: float) -> float:
                return float(np.asarray(copula.cdf(np.array([[a, b]])))[0])

            numeric = (
                corner(first + step, second + step)
                - corner(first + step, second - step)
                - corner(first - step, second + step)
                + corner(first - step, second - step)
            ) / (4 * step * step)
            analytic = float(np.asarray(copula.pdf(np.array([[first, second]])))[0])
            assert analytic == pytest.approx(numeric, rel=2e-4)

    def test_it_creates_upper_tail_dependence_from_none(self) -> None:
        # The point of the transformation: Clayton has no upper tail, and the
        # lifted version does.
        base = rc.ClaytonCopula(2.0)
        assert base.lambda_().upper == 0.0
        assert rc.opower(base, 2.0).lambda_().upper > 0.5

    def test_sampling_matches_the_copula(self) -> None:
        copula = rc.opower(rc.ClaytonCopula(2.0), 1.5)
        drawn = copula.rvs(30_000, random_state=0)
        assert float(rc.cor_kendall(drawn)[0, 1]) == pytest.approx(copula.tau(), abs=0.01)
        assert np.max(np.abs(drawn.mean(axis=0) - 0.5)) < 0.005

    def test_it_round_trips_through_json(self) -> None:
        # Before it was added to the registry this encoded happily and then
        # failed on the way back, which is the worst of both -- a document that
        # looks valid and is not readable.
        from rcopula.serialize import from_json, to_json

        original = rc.opower(rc.ClaytonCopula(2.0), 1.5)
        reloaded = from_json(to_json(original))
        assert reloaded.describe() == original.describe()
        points = np.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.2]])
        assert np.array_equal(np.asarray(original.cdf(points)), np.asarray(reloaded.cdf(points)))

    def test_higher_dimensions_have_a_cdf_but_no_density(self) -> None:
        copula = rc.opower(rc.ClaytonCopula(2.0, dim=4), 1.5)
        points = np.full((5, 4), 0.6)
        assert np.all(np.isfinite(copula.cdf(points)))
        with pytest.raises(NotImplementedError, match="dim=2 only"):
            copula.logpdf(points)

    def test_a_non_archimedean_base_is_refused(self) -> None:
        with pytest.raises(TypeError, match="Archimedean"):
            rc.opower(rc.GaussianCopula(0.5), 2.0)

    def test_alpha_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError):
            rc.opower(rc.ClaytonCopula(2.0), 0.5)
