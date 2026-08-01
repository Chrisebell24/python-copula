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

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.core.measures import rho_by_quadrature, tau_by_quadrature
from rcopula.structural import RotatedCopula, survival

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
