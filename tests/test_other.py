"""Behavioural tests for Plackett, FGM, Marshall-Olkin and the Frechet bounds."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from scipy import stats

import rcopula as rc

#: Families with an ordinary density and sampler, exercised generically below.
CONTINUOUS = [
    (rc.PlackettCopula, (3.0,)),
    (rc.FGMCopula, (0.7,)),
    (rc.IndependenceCopula, ()),
]


def _make(cls, args):
    return cls(*args)


# ----------------------------------------------------------------------
# Copula axioms
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("cls", "args"), CONTINUOUS)
def test_uniform_margins_of_the_cdf(cls, args) -> None:
    cop = _make(cls, args)
    u = np.linspace(0.01, 0.99, 40)
    grid = np.column_stack([u, np.ones_like(u)])
    assert np.allclose(cop.cdf(grid), u, atol=1e-10)


@pytest.mark.parametrize(("cls", "args"), CONTINUOUS)
def test_frechet_hoeffding_bounds(cls, args) -> None:
    cop = _make(cls, args)
    rng = np.random.default_rng(0)
    u = rng.uniform(0.01, 0.99, size=(400, 2))
    c = cop.cdf(u)
    assert np.all(c >= np.maximum(u.sum(axis=1) - 1.0, 0.0) - 1e-12)
    assert np.all(c <= u.min(axis=1) + 1e-12)


@pytest.mark.parametrize(("cls", "args"), CONTINUOUS)
def test_c_volume_is_non_negative(cls, args) -> None:
    cop = _make(cls, args)
    rng = np.random.default_rng(1)
    for _ in range(60):
        a = rng.uniform(0, 0.9, 2)
        b = np.minimum(a + rng.uniform(0.01, 0.1, 2), 1.0)
        assert cop.prob(a, b) >= -1e-12


@pytest.mark.parametrize(("cls", "args"), CONTINUOUS)
def test_pdf_matches_numerical_derivative_of_cdf(cls, args) -> None:
    cop = _make(cls, args)
    h = 1e-5
    for u, v in [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]:
        numerical = (
            cop.cdf([[u + h, v + h]])[0]
            - cop.cdf([[u + h, v - h]])[0]
            - cop.cdf([[u - h, v + h]])[0]
            + cop.cdf([[u - h, v - h]])[0]
        ) / (4 * h * h)
        assert cop.pdf([[u, v]])[0] == pytest.approx(numerical, rel=1e-4)


@pytest.mark.parametrize(("cls", "args"), CONTINUOUS)
def test_density_integrates_to_one(cls, args) -> None:
    rng = np.random.default_rng(2)
    u = rng.uniform(size=(200_000, 2))
    assert _make(cls, args).pdf(u).mean() == pytest.approx(1.0, abs=0.02)


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "args"),
    [*CONTINUOUS, (rc.MarshallOlkinCopula, (0.3, 0.7))],
)
def test_margins_are_uniform(cls, args) -> None:
    u = _make(cls, args).rvs(20_000, random_state=5)
    for j in range(2):
        assert stats.kstest(u[:, j], "uniform").pvalue > 0.001


@pytest.mark.parametrize(
    ("cls", "args"),
    [(rc.PlackettCopula, (3.0,)), (rc.FGMCopula, (0.7,)), (rc.MarshallOlkinCopula, (0.3, 0.7))],
)
def test_sample_kendall_tau_matches_population(cls, args) -> None:
    cop = _make(cls, args)
    u = cop.rvs(20_000, random_state=6)
    empirical = stats.kendalltau(u[:, 0], u[:, 1]).statistic
    assert empirical == pytest.approx(cop.tau(), abs=0.02)


def test_frechet_bounds_sample_degenerately() -> None:
    u = rc.FrechetUpperCopula(3).rvs(100, random_state=0)
    assert np.all(u[:, 0] == u[:, 1]) and np.all(u[:, 1] == u[:, 2])

    w = rc.FrechetLowerCopula().rvs(100, random_state=0)
    assert np.allclose(w[:, 0] + w[:, 1], 1.0)
    assert stats.kendalltau(w[:, 0], w[:, 1]).statistic == pytest.approx(-1.0)


# ----------------------------------------------------------------------
# Plackett
# ----------------------------------------------------------------------


class TestPlackett:
    def test_theta_one_is_independence(self) -> None:
        u = np.array([[0.3, 0.4], [0.7, 0.9]])
        assert np.allclose(rc.PlackettCopula(1.0).cdf(u), np.prod(u, axis=1))
        assert np.allclose(rc.PlackettCopula(1.0).pdf(u), 1.0)

    def test_approaches_the_frechet_bounds(self) -> None:
        u = np.array([[0.3, 0.7]])
        assert rc.PlackettCopula(1e8).cdf(u)[0] == pytest.approx(0.3, abs=1e-4)
        assert rc.PlackettCopula(1e-8).cdf(u)[0] == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("rho", [-0.8, -0.3, 0.0, 0.4, 0.9])
    def test_from_rho_round_trips(self, rho: float) -> None:
        assert rc.PlackettCopula.from_rho(rho).rho() == pytest.approx(rho, abs=1e-10)

    def test_rho_is_continuous_through_theta_one(self) -> None:
        """The closed form is 0/0 at theta = 1 and uses a series expansion."""
        for theta in (1 - 1e-6, 1 - 1e-8, 1.0, 1 + 1e-8, 1 + 1e-6):
            assert abs(rc.PlackettCopula(theta).rho()) < 1e-6

    def test_rejects_non_bivariate(self) -> None:
        with pytest.raises(ValueError, match="bivariate only"):
            rc.PlackettCopula(2.0, dim=3)


# ----------------------------------------------------------------------
# FGM
# ----------------------------------------------------------------------


class TestFGM:
    @pytest.mark.parametrize("theta", [-1.0, -0.4, 0.0, 0.6, 1.0])
    def test_closed_form_dependence(self, theta: float) -> None:
        cop = rc.FGMCopula(theta)
        assert cop.tau() == pytest.approx(2 * theta / 9)
        assert cop.rho() == pytest.approx(theta / 3)

    def test_dependence_range_is_narrow(self) -> None:
        """FGM cannot express strong dependence; the constructors say so."""
        assert rc.FGMCopula(1.0).tau() == pytest.approx(2 / 9)
        with pytest.raises(ValueError, match=r"tau in \[-2/9, 2/9\]"):
            rc.FGMCopula.from_tau(0.5)
        with pytest.raises(ValueError, match=r"rho in \[-1/3, 1/3\]"):
            rc.FGMCopula.from_rho(0.5)

    def test_theta_zero_is_independence(self) -> None:
        u = np.array([[0.3, 0.4], [0.8, 0.2]])
        assert np.allclose(rc.FGMCopula(0.0).cdf(u), np.prod(u, axis=1))

    def test_rejects_out_of_range_theta(self) -> None:
        with pytest.raises(ValueError, match="outside admissible range"):
            rc.FGMCopula(1.5)


# ----------------------------------------------------------------------
# Marshall-Olkin
# ----------------------------------------------------------------------


class TestMarshallOlkin:
    def test_is_asymmetric_unless_alphas_are_equal(self) -> None:
        asym = rc.MarshallOlkinCopula(0.2, 0.8)
        assert float(asym.cdf([[0.3, 0.7]])[0]) != pytest.approx(
            float(asym.cdf([[0.7, 0.3]])[0]), abs=1e-6
        )
        sym = rc.MarshallOlkinCopula(0.5, 0.5)
        assert float(sym.cdf([[0.3, 0.7]])[0]) == pytest.approx(
            float(sym.cdf([[0.7, 0.3]])[0]), abs=1e-12
        )

    def test_upper_tail_dependence_is_the_smaller_alpha(self) -> None:
        assert rc.MarshallOlkinCopula(0.2, 0.8).lambda_() == rc.TailDependence(0.0, 0.2)
        assert rc.MarshallOlkinCopula(0.9, 0.4).lambda_() == rc.TailDependence(0.0, 0.4)

    def test_zero_alphas_give_independence(self) -> None:
        u = np.array([[0.3, 0.4], [0.8, 0.2]])
        cop = rc.MarshallOlkinCopula(0.0, 0.0)
        assert np.allclose(cop.cdf(u), np.prod(u, axis=1))
        assert cop.tau() == 0.0

    def test_singular_component_carries_real_mass(self) -> None:
        """A visible fraction of draws lands exactly on u**a1 == v**a2.

        This is why the density describes only the continuous part.
        """
        a1, a2 = 0.4, 0.6
        u = rc.MarshallOlkinCopula(a1, a2).rvs(20_000, random_state=8)
        on_curve = np.isclose(u[:, 0] ** a1, u[:, 1] ** a2, rtol=1e-9)
        assert on_curve.mean() > 0.1

    def test_accepts_a_pair_or_two_scalars(self) -> None:
        assert np.allclose(
            rc.MarshallOlkinCopula([0.2, 0.8]).alpha, rc.MarshallOlkinCopula(0.2, 0.8).alpha
        )


# ----------------------------------------------------------------------
# Frechet bounds
# ----------------------------------------------------------------------


class TestFrechetBounds:
    def test_they_bracket_every_other_copula(self) -> None:
        rng = np.random.default_rng(3)
        u = rng.uniform(0.05, 0.95, size=(200, 2))
        lower = rc.FrechetLowerCopula().cdf(u)
        upper = rc.FrechetUpperCopula(2).cdf(u)
        for cop in (rc.GaussianCopula(0.5), rc.ClaytonCopula(2.0), rc.FGMCopula(0.8)):
            c = cop.cdf(u)
            assert np.all(c >= lower - 1e-12)
            assert np.all(c <= upper + 1e-12)

    def test_extreme_dependence_measures(self) -> None:
        assert rc.FrechetUpperCopula(2).tau() == 1.0
        assert rc.FrechetLowerCopula().tau() == -1.0

    def test_lower_bound_rejects_higher_dimensions(self) -> None:
        with pytest.raises(ValueError, match="only a copula for dim=2"):
            rc.FrechetLowerCopula(dim=3)

    def test_upper_bound_works_in_any_dimension(self) -> None:
        assert float(rc.FrechetUpperCopula(7).cdf([[0.4] * 7])[0]) == pytest.approx(0.4)


class TestPlackettTauInversion:
    """``from_tau`` was missing, so ``fit(..., method="itau")`` raised.

    Plackett's tau has no closed form in either direction, which is why it was
    left out; now that :meth:`tau` is reliable across the family's whole range
    it can simply be inverted numerically.
    """

    @pytest.mark.parametrize("tau", [-0.9, -0.5, -0.1, 0.1, 0.5, 0.9, 0.95])
    def test_it_round_trips(self, tau: float) -> None:
        assert rc.PlackettCopula.from_tau(tau).tau() == pytest.approx(tau, abs=1e-6)

    def test_independence_is_exact(self) -> None:
        assert rc.PlackettCopula.from_tau(0.0).theta == 1.0

    def test_it_is_monotone(self) -> None:
        thetas = [rc.PlackettCopula.from_tau(t).theta for t in (-0.5, -0.1, 0.1, 0.5)]
        assert all(a < b for a, b in pairwise(thetas))

    def test_it_rejects_unreachable_targets(self) -> None:
        for bad in (-1.0, 1.0, 1.5):
            with pytest.raises(ValueError, match="tau must lie"):
                rc.PlackettCopula.from_tau(bad)

    def test_fitting_by_inversion_now_works(self) -> None:
        u = rc.PlackettCopula(5.0).rvs(3000, random_state=0)
        res = rc.fit(rc.PlackettCopula(), u, method="itau")
        assert res.params[0] == pytest.approx(5.0, rel=0.2)
        assert res.bse is not None


class TestExtremeValueDefaultConstruction:
    """Every other family can be named without inventing a parameter value.

    ``ClaytonCopula()`` means "this family, to be estimated" -- NaN parameters
    that ``fit`` fills in. The extreme-value families required a positional
    argument, so ``fit(GalambosCopula(), u)`` raised ``TypeError`` and the
    natural idiom did not work for them alone.
    """

    @pytest.mark.parametrize(
        "ctor", [rc.GalambosCopula, rc.HuslerReissCopula, rc.TawnCopula, rc.TEVCopula]
    )
    def test_it_constructs_with_no_arguments(self, ctor: type) -> None:
        cop = ctor()
        assert np.isnan(cop.params[0])
        assert cop.dim == 2

    @pytest.mark.parametrize(
        "ctor", [rc.GalambosCopula, rc.HuslerReissCopula, rc.TawnCopula, rc.TEVCopula]
    )
    def test_it_refuses_to_evaluate_until_specified(self, ctor: type) -> None:
        with pytest.raises(ValueError, match="unspecified parameters"):
            ctor().cdf([[0.5, 0.5]])

    @pytest.mark.parametrize(
        ("ctor", "truth"),
        [(rc.GalambosCopula, 1.5), (rc.HuslerReissCopula, 1.5), (rc.TawnCopula, 0.7)],
    )
    def test_fitting_the_bare_family_recovers_the_parameter(self, ctor: type, truth: float) -> None:
        u = ctor(truth).rvs(2000, random_state=0)
        assert rc.fit(ctor(), u, method="mpl").params[0] == pytest.approx(truth, rel=0.15)
