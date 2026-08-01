"""Behavioural tests for the elliptical families.

The headline check is the one that matters for risk work: a Gaussian and a
Student-t copula calibrated to the *same* Kendall's tau still disagree
substantially about joint extremes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import rcopula as rc

FAMILIES = [rc.GaussianCopula, rc.StudentCopula]
DISPSTRS = ["ex", "ar1", "toep", "un"]


def _params_for(dispstr: str, dim: int) -> float | list[float]:
    if dispstr in ("ex", "ar1"):
        return 0.5
    if dispstr == "toep":
        return [0.5**k for k in range(1, dim)]
    return [0.4] * (dim * (dim - 1) // 2)


# ----------------------------------------------------------------------
# Correlation structures
# ----------------------------------------------------------------------


@pytest.mark.parametrize("dispstr", DISPSTRS)
@pytest.mark.parametrize("dim", [2, 3, 5])
def test_sigma_is_a_valid_correlation_matrix(dispstr: str, dim: int) -> None:
    cop = rc.GaussianCopula(_params_for(dispstr, dim), dim=dim, dispstr=dispstr)
    sigma = cop.sigma()
    assert sigma.shape == (dim, dim)
    assert np.allclose(np.diag(sigma), 1.0)
    assert np.allclose(sigma, sigma.T)
    assert np.linalg.eigvalsh(sigma).min() > 0


def test_ar1_decays_geometrically() -> None:
    sigma = rc.GaussianCopula(0.5, dim=4, dispstr="ar1").sigma()
    assert np.allclose(sigma[0], [1.0, 0.5, 0.25, 0.125])


def test_unstructured_parameters_map_pairwise() -> None:
    sigma = rc.GaussianCopula([0.6, 0.3, 0.2], dim=3, dispstr="un").sigma()
    assert sigma[0, 1] == 0.6
    assert sigma[0, 2] == 0.3
    assert sigma[1, 2] == 0.2


def test_p2P_and_P2p_round_trip() -> None:
    params = [0.6, 0.3, 0.2, -0.1, 0.4, 0.15]
    assert np.allclose(rc.P2p(rc.p2P(params, 4)), params)


def test_exchangeable_lower_bound_is_enforced() -> None:
    """rho >= -1/(d-1) is required for positive definiteness."""
    rc.GaussianCopula(-0.24, dim=5)  # just inside
    with pytest.raises(ValueError, match="outside admissible range"):
        rc.GaussianCopula(-0.3, dim=5)


def test_non_positive_definite_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="not positive definite"):
        rc.GaussianCopula([0.99, -0.99, 0.99], dim=3, dispstr="un")


def test_unknown_dispstr_is_rejected() -> None:
    with pytest.raises(ValueError, match="dispstr must be one of"):
        rc.GaussianCopula(0.5, dim=3, dispstr="banana")


# ----------------------------------------------------------------------
# Dependence measures
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("rho", [-0.7, -0.2, 0.0, 0.3, 0.9])
def test_kendall_tau_identity(cls, rho: float) -> None:
    """tau = (2/pi) arcsin(rho) holds for every elliptical copula."""
    assert cls(rho, dim=2).tau() == pytest.approx(2 / np.pi * np.arcsin(rho), rel=1e-13)


@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.5, 0.9])
def test_spearman_rho_identity(rho: float) -> None:
    expected = 6 / np.pi * np.arcsin(rho / 2)
    assert rc.GaussianCopula(rho, dim=2).rho() == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("tau", [-0.5, 0.0, 0.25, 0.75])
def test_from_tau_round_trips(cls, tau: float) -> None:
    assert cls.from_tau(tau).tau() == pytest.approx(tau, abs=1e-13)


@pytest.mark.parametrize("rho_s", [-0.5, 0.0, 0.3, 0.8])
def test_from_rho_round_trips(rho_s: float) -> None:
    assert rc.GaussianCopula.from_rho(rho_s).rho() == pytest.approx(rho_s, abs=1e-13)


def test_gaussian_has_no_tail_dependence() -> None:
    for rho in (-0.5, 0.0, 0.5, 0.99):
        assert rc.GaussianCopula(rho).lambda_() == rc.TailDependence(0.0, 0.0)


def test_student_tail_dependence_is_symmetric_and_positive() -> None:
    lam = rc.StudentCopula(0.5, df=4).lambda_()
    assert lam.lower == lam.upper
    assert lam.lower > 0


def test_student_tail_dependence_survives_zero_correlation() -> None:
    """The property that separates t from Gaussian: uncorrelated is not
    independent in the tails."""
    assert rc.StudentCopula(0.0, df=3).lambda_().upper > 0.1


def test_student_converges_to_gaussian_as_df_grows() -> None:
    """df = 1e4, not 1e6+.

    Beyond roughly 1e5 degrees of freedom ``scipy.stats.t.ppf`` itself starts
    losing accuracy, so a tighter test at larger df would be measuring scipy's
    quantile function rather than this package's convergence.
    """
    assert rc.StudentCopula(0.5, df=1e6).lambda_().upper < 1e-4
    u = np.array([[0.3, 0.7], [0.5, 0.5], [0.9, 0.2]])
    assert np.allclose(
        rc.StudentCopula(0.5, df=1e4).cdf(u), rc.GaussianCopula(0.5).cdf(u), atol=1e-4
    )


class TestGaussianVersusStudent:
    """Same tau, very different joint extremes -- the whole point of the t copula."""

    def test_joint_tail_probabilities_differ_materially(self) -> None:
        tau = 0.5
        gauss = rc.GaussianCopula.from_tau(tau)
        t = rc.StudentCopula.from_tau(tau, df=3)

        # Probability that both margins fall below their q-quantile. The gap
        # widens as q shrinks, which is exactly the asymptotic tail-dependence
        # statement: measured ratios at tau = 0.5 are about 1.27x at q = 0.05,
        # 1.72x at q = 0.01 and over 2x at q = 0.001.
        for q, min_ratio in ((0.05, 1.2), (0.01, 1.6), (0.001, 2.0)):
            p_gauss = float(gauss.cdf([[q, q]])[0])
            p_t = float(t.cdf([[q, q]])[0])
            assert p_t > min_ratio * p_gauss, (
                f"at q={q} the t copula joint-tail probability {p_t:.3e} should "
                f"exceed {min_ratio}x the Gaussian's {p_gauss:.3e}"
            )

    def test_tau_alone_cannot_distinguish_them(self) -> None:
        gauss = rc.GaussianCopula.from_tau(0.5)
        t = rc.StudentCopula.from_tau(0.5, df=3)
        assert gauss.tau() == pytest.approx(t.tau(), rel=1e-13)


# ----------------------------------------------------------------------
# Distribution functions and sampling
# ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("dim", [2, 3, 5])
def test_margins_are_uniform(cls, dim: int) -> None:
    u = cls(0.5, dim=dim).rvs(20_000, random_state=7)
    for j in range(dim):
        assert stats.kstest(u[:, j], "uniform").pvalue > 0.001


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("dim", [2, 3])
def test_sample_kendall_tau_matches_population(cls, dim: int) -> None:
    cop = cls(0.5, dim=dim)
    u = cop.rvs(8000, random_state=11)
    empirical = stats.kendalltau(u[:, 0], u[:, 1]).statistic
    assert empirical == pytest.approx(cop.tau(), abs=0.025)


@pytest.mark.parametrize("cls", FAMILIES)
def test_uniform_margins_of_the_cdf(cls) -> None:
    cop = cls(0.5, dim=3)
    u = np.linspace(0.05, 0.95, 15)
    grid = np.column_stack([u, np.ones_like(u), np.ones_like(u)])
    assert np.allclose(cop.cdf(grid), u, atol=1e-5)


@pytest.mark.parametrize("cls", FAMILIES)
def test_independence_case_factorises(cls) -> None:
    cop = cls(0.0, dim=3, df=1e8) if cls is rc.StudentCopula else cls(0.0, dim=3)
    u = np.array([[0.3, 0.5, 0.7], [0.2, 0.9, 0.4]])
    assert np.allclose(cop.cdf(u), np.prod(u, axis=1), atol=1e-4)


@pytest.mark.parametrize("cls", FAMILIES)
def test_pdf_matches_numerical_derivative_of_cdf(cls) -> None:
    cop = cls(0.5, dim=2)
    h = 1e-4
    for u, v in [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2)]:
        numerical = (
            cop.cdf([[u + h, v + h]])[0]
            - cop.cdf([[u + h, v - h]])[0]
            - cop.cdf([[u - h, v + h]])[0]
            + cop.cdf([[u - h, v - h]])[0]
        ) / (4 * h * h)
        assert cop.pdf([[u, v]])[0] == pytest.approx(numerical, rel=1e-4)


@pytest.mark.parametrize("cls", FAMILIES)
@pytest.mark.parametrize("dim", [2, 3])
def test_density_integrates_to_one(cls, dim: int) -> None:
    rng = np.random.default_rng(2)
    u = rng.uniform(size=(200_000, dim))
    assert cls(0.5, dim=dim).pdf(u).mean() == pytest.approx(1.0, abs=0.03)


def test_cdf_is_deterministic() -> None:
    """scipy's multivariate_normal.cdf is not; ours must be, or golden fixtures
    would be meaningless."""
    cop = rc.GaussianCopula(0.5, dim=5)
    u = np.full((3, 5), 0.4)
    assert np.array_equal(cop.cdf(u), cop.cdf(u))


# ----------------------------------------------------------------------
# Student-specific
# ----------------------------------------------------------------------


def test_non_integer_degrees_of_freedom_are_supported() -> None:
    """R's pmvt rejects these; fitted t copulas produce them all the time."""
    cop = rc.StudentCopula(0.5, df=3.5)
    assert cop.df == 3.5
    assert 0.0 < float(cop.cdf([[0.5, 0.5]])[0]) < 1.0


def test_df_fixed_removes_it_from_the_free_parameters() -> None:
    free_df = rc.StudentCopula(0.5, df=4)
    fixed_df = rc.StudentCopula(0.5, df=4, df_fixed=True)
    assert free_df.n_params == 2
    assert fixed_df.n_params == 1
    assert "fixed" in fixed_df.describe()


def test_parameter_names_follow_the_structure() -> None:
    assert rc.GaussianCopula(0.5, dim=3).param_names == ("rho",)
    assert rc.GaussianCopula([0.5, 0.2], dim=3, dispstr="toep").param_names == (
        "rho.1",
        "rho.2",
    )
    assert rc.StudentCopula(0.5, dim=2).param_names == ("rho", "df")


def test_with_params_preserves_structure() -> None:
    cop = rc.GaussianCopula([0.5, 0.2], dim=3, dispstr="toep")
    other = cop.with_params([0.3, 0.1])
    assert other.dispstr == "toep"
    assert other.dim == 3
    assert np.allclose(other.params, [0.3, 0.1])
