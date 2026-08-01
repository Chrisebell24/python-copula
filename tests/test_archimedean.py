"""Behavioural tests for the Archimedean families.

Complements ``test_golden_archimedean.py``, which pins numeric agreement with R.
This file checks the properties that make something *a copula* at all — uniform
margins, the Frechet-Hoeffding bounds, a non-negative C-volume — plus the
sampler, which cannot be compared to R value-for-value because the RNG streams
differ.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import rcopula as rc

FAMILIES = [rc.ClaytonCopula, rc.GumbelCopula, rc.FrankCopula, rc.JoeCopula, rc.AMHCopula]

# Calibrating every family to the same Kendall's tau makes the cases directly
# comparable. 0.25 rather than 0.5 because AMH cannot reach beyond tau = 1/3.
COMMON_TAU = 0.25


def _cop(cls, dim=2):
    return cls.from_tau(COMMON_TAU, dim=dim)


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("dim", [2, 3, 5])
def test_margins_are_uniform(cls, dim: int) -> None:
    """Every margin of a copula sample must be Uniform(0,1). This is the single
    strongest check on the Marshall-Olkin frailty sampler: get the frailty
    distribution even slightly wrong and the margins stop being uniform."""
    u = _cop(cls, dim).rvs(20_000, random_state=12345)
    for j in range(dim):
        p = stats.kstest(u[:, j], "uniform").pvalue
        assert p > 0.001, f"margin {j} of {cls.__name__} d={dim} is not uniform (p={p:.2g})"


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("dim", [2, 3, 5])
def test_sample_kendall_tau_matches_population(cls, dim: int) -> None:
    """The empirical Kendall's tau of a sample must converge to the population
    value. This validates the *dependence* the sampler produces, not just its
    margins."""
    cop = _cop(cls, dim)
    u = cop.rvs(8000, random_state=999)
    empirical = np.mean(
        [
            stats.kendalltau(u[:, i], u[:, j]).statistic
            for i in range(dim)
            for j in range(i + 1, dim)
        ]
    )
    assert empirical == pytest.approx(cop.tau(), abs=0.02)


@pytest.mark.parametrize("cls", FAMILIES)
def test_sampling_is_reproducible(cls) -> None:
    a = _cop(cls).rvs(50, random_state=7)
    b = _cop(cls).rvs(50, random_state=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, _cop(cls).rvs(50, random_state=8))


@pytest.mark.parametrize("cls", FAMILIES)
def test_accepts_a_generator_as_random_state(cls) -> None:
    rng = np.random.default_rng(3)
    u = _cop(cls).rvs(10, random_state=rng)
    assert u.shape == (10, 2)


# ----------------------------------------------------------------------
# Copula axioms
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("dim", [2, 3])
def test_frechet_hoeffding_bounds(cls, dim: int) -> None:
    """W(u) <= C(u) <= M(u) everywhere."""
    rng = np.random.default_rng(0)
    u = rng.uniform(0.01, 0.99, size=(500, dim))
    c = _cop(cls, dim).cdf(u)
    lower = np.maximum(u.sum(axis=1) - (dim - 1), 0.0)
    upper = u.min(axis=1)
    assert np.all(c >= lower - 1e-12)
    assert np.all(c <= upper + 1e-12)


@pytest.mark.parametrize("cls", FAMILIES)
def test_uniform_margins_of_the_cdf(cls) -> None:
    """C(u, 1, ..., 1) = u — the defining margin property."""
    cop = _cop(cls, 3)
    u = np.linspace(0.01, 0.99, 50)
    grid = np.column_stack([u, np.ones_like(u), np.ones_like(u)])
    assert np.allclose(cop.cdf(grid), u, atol=1e-10)


@pytest.mark.parametrize("cls", FAMILIES)
def test_cdf_is_zero_when_any_coordinate_is_zero(cls) -> None:
    cop = _cop(cls, 3)
    assert cop.cdf([[0.0, 0.5, 0.5]])[0] == 0.0
    assert cop.cdf([[0.5, 0.0, 0.5]])[0] == 0.0


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("dim", [2, 3])
def test_c_volume_is_non_negative(cls, dim: int) -> None:
    """A copula is d-increasing: every box has non-negative probability."""
    cop = _cop(cls, dim)
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.uniform(0, 0.9, dim)
        b = a + rng.uniform(0.01, 0.1, dim)
        assert cop.prob(a, np.minimum(b, 1.0)) >= -1e-12


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("dim", [2, 3])
def test_density_integrates_to_one(cls, dim: int) -> None:
    """Monte-Carlo check that the density is normalised, which catches a wrong
    constant in the d-th generator derivative that ``pdf`` alone would not."""
    cop = _cop(cls, dim)
    rng = np.random.default_rng(4)
    u = rng.uniform(size=(200_000, dim))
    assert cop.pdf(u).mean() == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("cls", FAMILIES)
def test_pdf_matches_numerical_derivative_of_cdf(cls) -> None:
    """c(u,v) = d2C/dudv. An independent check on the density, since ``pdf`` and
    ``cdf`` are computed by completely different code paths."""
    cop = _cop(cls)
    h = 1e-5
    pts = np.array([[0.3, 0.4], [0.5, 0.5], [0.7, 0.2], [0.85, 0.9]])
    for u, v in pts:
        numerical = (
            cop.cdf([[u + h, v + h]])[0]
            - cop.cdf([[u + h, v - h]])[0]
            - cop.cdf([[u - h, v + h]])[0]
            + cop.cdf([[u - h, v - h]])[0]
        ) / (4 * h * h)
        assert cop.pdf([[u, v]])[0] == pytest.approx(numerical, rel=1e-4)


# ----------------------------------------------------------------------
# Generator round-trips and calibration
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FAMILIES)
def test_generator_inverts_itself(cls) -> None:
    cop = _cop(cls)
    u = np.linspace(0.001, 0.999, 200)
    assert np.allclose(cop.psi(cop.ipsi(u)), u, rtol=1e-11)


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("tau", [0.05, 0.2, 0.3])
def test_from_tau_round_trips(cls, tau: float) -> None:
    """Only tau values every family can reach; AMH tops out at 1/3."""
    assert cls.from_tau(tau).tau() == pytest.approx(tau, abs=1e-10)


@pytest.mark.parametrize("cls", [rc.ClaytonCopula, rc.GumbelCopula, rc.FrankCopula, rc.JoeCopula])
@pytest.mark.parametrize("tau", [0.4, 0.6, 0.8, 0.95])
def test_from_tau_round_trips_strong_dependence(cls, tau: float) -> None:
    assert cls.from_tau(tau).tau() == pytest.approx(tau, abs=1e-10)


def test_amh_rejects_unreachable_dependence() -> None:
    """AMH spans only tau in [(5 - 8 log 2)/3, 1/3] ~ [-0.1817, 0.3333]."""
    with pytest.raises(ValueError, match="not attainable"):
        rc.AMHCopula.from_tau(0.5)
    with pytest.raises(ValueError, match="not attainable"):
        rc.AMHCopula.from_tau(-0.5)


@pytest.mark.parametrize("tau", [-0.8, -0.4, -0.1])
def test_frank_and_clayton_support_negative_dependence_in_2d(tau: float) -> None:
    assert rc.FrankCopula.from_tau(tau).tau() == pytest.approx(tau, abs=1e-9)
    assert rc.ClaytonCopula.from_tau(tau).tau() == pytest.approx(tau, abs=1e-12)


def test_amh_supports_only_weak_negative_dependence() -> None:
    assert rc.AMHCopula.from_tau(-0.15).tau() == pytest.approx(-0.15, abs=1e-10)


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("rho", [0.05, 0.15, 0.3])
def test_from_rho_round_trips(cls, rho: float) -> None:
    assert cls.from_rho(rho).rho() == pytest.approx(rho, abs=1e-6)


# ----------------------------------------------------------------------
# Parameter validation and object semantics
# ----------------------------------------------------------------------


def test_clayton_rejects_negative_theta_above_two_dimensions() -> None:
    """Negative theta is only valid in d=2 (McNeil & Neslehova 2009)."""
    rc.ClaytonCopula(-0.5, dim=2)  # fine
    with pytest.raises(ValueError, match="outside admissible range"):
        rc.ClaytonCopula(-0.5, dim=3)


def test_gumbel_rejects_theta_below_one() -> None:
    with pytest.raises(ValueError, match="outside admissible range"):
        rc.GumbelCopula(0.5)


def test_joe_rejects_theta_below_one() -> None:
    with pytest.raises(ValueError, match="outside admissible range"):
        rc.JoeCopula(0.5)


def test_amh_rejects_theta_at_or_above_one() -> None:
    with pytest.raises(ValueError, match="outside admissible range"):
        rc.AMHCopula(1.5)


def test_amh_negative_theta_only_in_two_dimensions() -> None:
    """A deliberate divergence from R, which forbids AMH beyond d = 2 entirely."""
    rc.AMHCopula(-0.5, dim=2)
    rc.AMHCopula(0.5, dim=4)
    with pytest.raises(ValueError, match="outside admissible range"):
        rc.AMHCopula(-0.5, dim=3)


def test_unspecified_parameters_raise_on_evaluation() -> None:
    cop = rc.ClaytonCopula()  # theta = NaN, "to be estimated"
    assert np.isnan(cop.theta)
    with pytest.raises(ValueError, match="unspecified parameters"):
        cop.pdf([[0.5, 0.5]])


def test_dimension_mismatch_is_reported_clearly() -> None:
    with pytest.raises(ValueError, match="has 3 column"):
        rc.ClaytonCopula(2.0, dim=2).pdf([[0.5, 0.5, 0.5]])


def test_copulas_are_immutable_and_with_params_copies() -> None:
    cop = rc.ClaytonCopula(2.0, dim=3)
    other = cop.with_params([4.0])
    assert cop.theta == 2.0 and other.theta == 4.0
    assert other.dim == 3 and type(other) is type(cop)
    with pytest.raises(ValueError):
        cop.params[0] = 99.0


def test_equality_and_hashing() -> None:
    assert rc.ClaytonCopula(2.0) == rc.ClaytonCopula(2.0)
    assert rc.ClaytonCopula(2.0) != rc.ClaytonCopula(3.0)
    assert rc.ClaytonCopula(2.0) != rc.GumbelCopula(2.0)
    assert len({rc.ClaytonCopula(2.0), rc.ClaytonCopula(2.0)}) == 1


def test_out_of_range_input_has_zero_density() -> None:
    cop = rc.ClaytonCopula(2.0)
    assert cop.pdf([[1.5, 0.5]])[0] == 0.0
    assert cop.pdf([[-0.1, 0.5]])[0] == 0.0


def test_describe_mentions_family_dimension_and_parameter() -> None:
    text = rc.ClaytonCopula(2.0, dim=4).describe()
    assert "Clayton" in text and "dim 4" in text and "theta=2" in text


def test_fixed_parameters_are_reported() -> None:
    cop = rc.ClaytonCopula(2.0).fix_params([False])
    assert cop.n_params == 0
    assert "fixed" in cop.describe()


class TestAMHAtZero:
    """theta = 0 sits inside AMH's admissible interval [-1, 1).

    The closed form for the generator derivatives carries a ``1/theta``, so a
    naive implementation raises exactly where an optimiser is most likely to
    step -- and the singularity is removable: at theta = 0 AMH *is* the
    independence copula.
    """

    def test_the_density_is_one(self) -> None:
        u = np.array([[0.2, 0.7], [0.5, 0.5], [0.9, 0.1], [0.01, 0.99]])
        assert np.allclose(rc.AMHCopula(0.0).pdf(u), 1.0, rtol=1e-12)

    def test_the_cdf_is_the_product(self) -> None:
        u = np.array([[0.2, 0.7], [0.5, 0.5], [0.9, 0.1]])
        assert np.allclose(rc.AMHCopula(0.0).cdf(u), u.prod(axis=1), rtol=1e-12)

    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_the_density_is_one_in_every_dimension(self, dim: int) -> None:
        u = np.random.default_rng(0).uniform(0.05, 0.95, size=(20, dim))
        assert np.allclose(rc.AMHCopula(0.0, dim=dim).pdf(u), 1.0, rtol=1e-12)

    def test_the_density_is_continuous_across_zero(self) -> None:
        u = np.array([[0.3, 0.8], [0.6, 0.4]])
        left = rc.AMHCopula(-1e-7).pdf(u)
        here = rc.AMHCopula(0.0).pdf(u)
        right = rc.AMHCopula(1e-7).pdf(u)
        assert np.allclose(left, here, atol=1e-6)
        assert np.allclose(right, here, atol=1e-6)
        # And genuinely monotone through the point, not merely equal at it.
        assert np.all(left > right)

    def test_generator_derivatives_stay_finite(self) -> None:
        gen = rc.AMHCopula(0.0).generator
        t = np.array([1e-8, 0.1, 1.0, 10.0, 40.0])
        for theta in (0.0, 1e-14, -1e-14, 1e-6):
            for order in (1, 2, 3, 4):
                assert np.all(np.isfinite(gen.log_abs_dpsi_d(t, theta, order)))

    def test_the_first_derivative_is_exactly_minus_t_in_logs(self) -> None:
        """psi(t) = e^{-t} at theta = 0, so |psi^(d)(t)| = e^{-t} for every d."""
        gen = rc.AMHCopula(0.0).generator
        t = np.array([0.1, 1.0, 5.0, 40.0])
        for order in (1, 2, 3):
            assert np.allclose(gen.log_abs_dpsi_d(t, 0.0, order), -t, rtol=1e-13)

    def test_fitting_walks_through_zero_without_raising(self) -> None:
        """The failure this guards: an optimiser stepping onto theta = 0."""
        u = rc.AMHCopula(0.05).rvs(600, random_state=0)
        res = rc.fit(rc.AMHCopula(), u, method="mpl")
        assert res.copula.theta == pytest.approx(0.05, abs=0.35)
