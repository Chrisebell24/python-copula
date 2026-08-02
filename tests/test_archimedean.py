"""Behavioural tests for the Archimedean families.

Complements ``test_golden_archimedean.py``, which pins numeric agreement with R.
This file checks the properties that make something *a copula* at all — uniform
margins, the Frechet-Hoeffding bounds, a non-negative C-volume — plus the
sampler, which cannot be compared to R value-for-value because the RNG streams
differ.
"""

from __future__ import annotations

from typing import ClassVar

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


def _mixed_second_difference(cop: rc.Copula, x: float, y: float, h: float = 1e-5) -> float:
    """d2C/dudv by central differences -- an independent check on the density."""
    corners = np.array([[x + h, y + h], [x + h, y - h], [x - h, y + h], [x - h, y - h]])
    c = cop.cdf(corners)
    return float((c[0] - c[1] - c[2] + c[3]) / (4.0 * h * h))


class TestNegativeDependence:
    """Clayton, Frank and AMH admit negative theta in d = 2 -- and it was broken.

    Frank's density was ``nan`` for *every* negative theta, Clayton's for every
    negative theta, and none of the three could be sampled at all. All three
    failures came from the same place: closed forms written assuming the
    positively-dependent branch, where a quantity that is genuinely negative
    (``theta`` itself, the polylogarithm, the generator argument) was passed to
    ``log`` unsigned, or a frailty was drawn from a distribution that does not
    exist there.

    The negative branch is not an edge case: it is the only reason to prefer
    Frank over Gumbel, and the reason Clayton has a lower bound of -1 rather
    than 0.
    """

    NEGATIVE: ClassVar[list[tuple[type, float]]] = [
        (rc.ClaytonCopula, -0.9),
        (rc.ClaytonCopula, -0.5),
        (rc.ClaytonCopula, -0.1),
        (rc.FrankCopula, -0.5),
        (rc.FrankCopula, -2.0),
        (rc.FrankCopula, -8.0),
        (rc.FrankCopula, -25.0),
        (rc.AMHCopula, -0.3),
        (rc.AMHCopula, -0.9),
    ]

    @pytest.mark.parametrize(("family", "theta"), NEGATIVE)
    def test_the_density_is_finite_and_positive(self, family: type, theta: float) -> None:
        u = np.array([[0.2, 0.7], [0.5, 0.5], [0.9, 0.1], [0.05, 0.95], [0.3, 0.3]])
        pdf = family(theta).pdf(u)
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0.0)

    @pytest.mark.parametrize(("family", "theta"), NEGATIVE)
    def test_the_density_is_the_derivative_of_the_cdf(self, family: type, theta: float) -> None:
        """The check that catches a sign error rather than merely a nan."""
        cop = family(theta)
        for x, y in [(0.3, 0.5), (0.6, 0.6), (0.8, 0.4), (0.45, 0.25)]:
            if cop.cdf([[x, y]])[0] <= 0.0:
                continue  # outside the support, where Clayton's density is 0
            assert cop.pdf([[x, y]])[0] == pytest.approx(
                _mixed_second_difference(cop, x, y), abs=1e-5, rel=1e-4
            )

    @pytest.mark.parametrize(("family", "theta"), NEGATIVE)
    def test_sampling_reproduces_the_theoretical_tau(self, family: type, theta: float) -> None:
        """There is no frailty on this branch, so sampling goes by inversion."""
        cop = family(theta)
        sample = cop.rvs(40_000, random_state=0)
        assert stats.kendalltau(sample[:, 0], sample[:, 1]).statistic == pytest.approx(
            cop.tau(), abs=0.01
        )

    @pytest.mark.parametrize(("family", "theta"), NEGATIVE)
    def test_samples_have_uniform_margins(self, family: type, theta: float) -> None:
        sample = family(theta).rvs(20_000, random_state=1)
        for j in range(2):
            assert stats.kstest(sample[:, j], "uniform").pvalue > 0.01

    @pytest.mark.parametrize(("family", "theta"), NEGATIVE)
    def test_estimation_recovers_the_parameter(self, family: type, theta: float) -> None:
        """Within three standard errors, using the fit's own reported error.

        A fixed tolerance would be wrong here: Frank at theta = -0.5 has
        tau = -0.055, so the parameter is barely identified and its standard
        error is an order of magnitude larger than at theta = -8. Judging every
        case by the same absolute gap would either fail on the weak ones or pass
        vacuously on the strong ones.
        """
        sample = family(theta).rvs(6000, random_state=0)
        # The itau standard error is the yardstick because it always exists:
        # its asymptotics rest on a rank statistic, not on differentiating a
        # likelihood whose support may move -- see the Clayton test below.
        reference = rc.fit(family(), sample, method="itau")
        assert reference.bse is not None
        for method in ("mpl", "itau"):
            estimate = rc.fit(family(), sample, method=method).params[0]
            assert abs(estimate - theta) < 3.0 * reference.bse[0]

    @pytest.mark.parametrize("theta", [-0.9, -0.5])
    def test_clayton_below_zero_refuses_a_pseudo_likelihood_standard_error(
        self, theta: float
    ) -> None:
        """A non-regular model, and the code has to say so rather than guess.

        For ``theta < 0`` the Clayton density vanishes outside
        ``u^-theta + v^-theta > 1``, so the *support depends on the parameter* --
        the textbook irregular case, like estimating the endpoint of a uniform.
        Differentiating the likelihood across that moving boundary gives a
        negative "negative Hessian", and the sandwich ``H^-1 S H^-1`` is
        positive whatever the sign of ``H``, so a meaningless number would
        otherwise come back looking perfectly respectable.

        The point estimate is still good; only the standard error is refused.
        """
        sample = rc.ClaytonCopula(theta).rvs(3000, random_state=0)
        mpl = rc.fit(rc.ClaytonCopula(), sample, method="mpl")
        assert mpl.params[0] == pytest.approx(theta, abs=0.05)
        assert mpl.bse is None
        assert "not computed" in mpl.summary()

        inversion = rc.fit(rc.ClaytonCopula(), sample, method="itau")
        assert inversion.bse is not None
        assert inversion.bse[0] > 0.0

    @pytest.mark.parametrize(("family", "theta"), [(rc.FrankCopula, -4.0), (rc.AMHCopula, -0.6)])
    def test_families_with_fixed_support_still_report_one(self, family: type, theta: float) -> None:
        """The refusal above must be specific to the moving boundary, not blanket."""
        sample = family(theta).rvs(3000, random_state=0)
        res = rc.fit(family(), sample, method="mpl")
        assert res.bse is not None
        assert res.bse[0] > 0.0

    @pytest.mark.parametrize(("family", "theta"), NEGATIVE)
    def test_tau_and_rho_are_negative(self, family: type, theta: float) -> None:
        cop = family(theta)
        assert cop.tau() < 0.0
        assert cop.rho() < 0.0

    def test_clayton_has_finite_support_below_zero(self) -> None:
        """psi(t) = 0 once 1 + t <= 0, so C reaches the Frechet lower bound.

        Reached at ordinary (u, v) -- here C(0.2, 0.7) is exactly zero at
        theta = -0.9 -- so it is a branch that has to be handled, not an
        asymptotic curiosity.
        """
        cop = rc.ClaytonCopula(-0.9)
        assert cop.cdf([[0.2, 0.7]])[0] == 0.0
        assert cop.pdf([[0.2, 0.7]])[0] == 0.0
        # W(u, v) = max(u + v - 1, 0) is the bound it is pressed against.
        assert cop.cdf([[0.4, 0.4]])[0] >= max(0.4 + 0.4 - 1.0, 0.0)

    def test_clayton_approaches_the_lower_frechet_bound(self) -> None:
        cop = rc.ClaytonCopula(-1.0 + 1e-9)
        u = np.array([[0.3, 0.9], [0.6, 0.6], [0.8, 0.5]])
        assert np.allclose(cop.cdf(u), np.maximum(u.sum(axis=1) - 1.0, 0.0), atol=1e-6)

    @pytest.mark.parametrize(
        "cop",
        [
            rc.ClaytonCopula(0.0),
            rc.FrankCopula(0.0),
            rc.AMHCopula(0.0),
            rc.GumbelCopula(1.0),
            rc.JoeCopula(1.0),
        ],
    )
    def test_the_independence_point_is_reachable(self, cop: rc.Copula) -> None:
        """Every family's degenerate theta, where the generator divides by zero."""
        u = np.array([[0.2, 0.7], [0.5, 0.5], [0.9, 0.1]])
        assert np.allclose(cop.pdf(u), 1.0, atol=1e-6)
        assert np.allclose(cop.cdf(u), u.prod(axis=1), atol=1e-6)
        sample = cop.rvs(20_000, random_state=0)
        assert abs(stats.kendalltau(sample[:, 0], sample[:, 1]).statistic) < 0.02

    def test_negative_theta_is_rejected_above_two_dimensions(self) -> None:
        """The generator stops being d-monotone; the bounds must say so."""
        with pytest.raises(ValueError, match="outside admissible range"):
            rc.ClaytonCopula(-0.5, dim=3)
        with pytest.raises(ValueError, match="outside admissible range"):
            rc.FrankCopula(-2.0, dim=3)
