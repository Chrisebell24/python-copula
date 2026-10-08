"""Factor copulas: exact identities with the dense elliptical copulas, recovery, scale."""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.core.elliptical import P2p
from rcopula.garch import CopulaGarch
from rcopula.serialize import from_json, to_json

GROUPS_SMALL = np.array([0, 0, 1, 1, 1, 2])


def small(family: str = "student") -> rc.FactorCopula:
    return rc.FactorCopula(
        [0.6, 0.5, 0.7, 0.4, 0.55, 0.65],
        group_loadings=[0.3, 0.4, 0.2, 0.5, 0.35, 0.3],
        groups=GROUPS_SMALL,
        family=family,
        df=4.5 if family == "student" else None,
    )


def simulated(d: int, n_groups: int, family: str, seed: int = 0) -> rc.FactorCopula:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), d // n_groups)
    return rc.FactorCopula(
        rng.uniform(0.4, 0.7, d),
        rng.uniform(0.3, 0.5, d),
        groups,
        family=family,
        df=4.0 if family == "student" else None,
    )


# -- construction ---------------------------------------------------------


def test_parameters_are_laid_out_and_counted() -> None:
    cop = small()
    assert cop.dim == 6
    assert cop.n_groups == 3 and cop.n_factors == 4
    assert cop.n_params == 13  # 6 market + 6 group + df
    assert cop.param_names[0] == "market_0" and cop.param_names[6] == "group_0"
    assert cop.param_names[-1] == "df"
    np.testing.assert_allclose(cop.idiosyncratic, 1 - cop.market**2 - cop.group_loadings**2)
    assert small("gaussian").n_params == 12
    assert rc.FactorCopula([0.5, 0.6, 0.7]).n_params == 3
    assert np.isinf(rc.FactorCopula([0.5, 0.6]).df)


def test_family_is_inferred_from_df() -> None:
    assert rc.FactorCopula([0.5, 0.6], df=6.0).family == "student"
    assert rc.FactorCopula([0.5, 0.6]).family == "gaussian"
    assert rc.FactorCopula([0.5, 0.6], family="student").df == 4.0


def test_string_group_labels_are_recoded() -> None:
    cop = rc.FactorCopula([0.5, 0.6, 0.7], [0.3, 0.3, 0.2], ["tech", "bank", "tech"])
    np.testing.assert_array_equal(cop.groups, [1, 0, 1])
    assert list(cop.group_labels) == ["bank", "tech"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"market": [0.9, 0.5], "group_loadings": [0.5, 0.1], "groups": [0, 0]}, "below 1"),
        ({"market": [0.5, 0.5], "groups": [0, 0]}, "together"),
        ({"market": [0.5, 0.5], "group_loadings": [0.1, 0.1]}, "together"),
        ({"market": [0.5, 0.5], "group_loadings": [0.1], "groups": [0]}, "shape"),
        ({"market": [0.5, 0.5], "family": "gaussian", "df": 4.0}, "df applies"),
        ({"market": [0.5, 0.5], "family": "clayton"}, "family"),
        ({"market": [1.5, 0.5]}, "outside"),
        ({"market": [0.5]}, "dim"),
    ],
)
def test_bad_arguments_are_rejected(kwargs: dict, match: str) -> None:
    market = kwargs.pop("market")
    with pytest.raises(ValueError, match=match):
        rc.FactorCopula(market, **kwargs)


def test_unfitted_template_cannot_be_evaluated() -> None:
    template = rc.FactorCopula.unfitted(4, groups=[0, 0, 1, 1])
    assert template.n_params == 9
    assert "unfitted" in template.describe()
    with pytest.raises(ValueError, match="unspecified"):
        template.logpdf([0.5, 0.5, 0.5, 0.5])


def test_with_params_fix_params_equality_and_hash() -> None:
    cop = small()
    same = cop.with_params(cop.params)
    assert same == cop and hash(same) == hash(cop)
    pinned = cop.fix_params([True] * 12 + [False])
    assert pinned.n_params == 12 and pinned != cop
    assert isinstance(pinned, rc.FactorCopula) and pinned.groups is not None
    other_groups = rc.FactorCopula(cop.market, cop.group_loadings, [0, 0, 1, 1, 2, 2], df=4.5)
    assert other_groups != cop


# -- the density is exactly the dense elliptical one ----------------------


@pytest.mark.parametrize("family", ["gaussian", "student"])
@pytest.mark.parametrize("grouped", [True, False])
def test_logpdf_equals_the_dense_copula_with_the_implied_matrix(family: str, grouped: bool) -> None:
    cop = small(family)
    if not grouped:
        cop = rc.FactorCopula(cop.market, family=family, df=cop.df if family == "student" else None)
    dense = (
        rc.StudentCopula(P2p(cop.sigma()), dim=6, dispstr="un", df=cop.df)
        if family == "student"
        else rc.GaussianCopula(P2p(cop.sigma()), dim=6, dispstr="un")
    )
    u = np.random.default_rng(1).uniform(0.001, 0.999, (200, 6))
    np.testing.assert_allclose(cop.logpdf(u), dense.logpdf(u), rtol=0, atol=1e-9)
    assert type(cop.to_elliptical()) is type(dense)
    np.testing.assert_allclose(cop.to_elliptical().logpdf(u), dense.logpdf(u), atol=1e-12)


def test_sigma_has_the_factor_structure() -> None:
    cop = small()
    a, b, g = cop.market, cop.group_loadings, GROUPS_SMALL
    R = cop.sigma()
    assert R[0, 1] == pytest.approx(a[0] * a[1] + b[0] * b[1])  # same group
    assert R[0, 2] == pytest.approx(a[0] * a[2])  # different groups
    np.testing.assert_allclose(np.diag(R), 1.0)
    assert np.linalg.eigvalsh(R)[0] > 0
    assert g[0] == g[1] != g[2]


def test_logpdf_handles_the_boundary_and_large_batches() -> None:
    cop = small()
    out = cop.logpdf([[0.0, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5] * 6])
    assert out[0] == -np.inf and np.isfinite(out[1])
    # more rows than one internal block, so the blocking is exercised
    big = rc.FactorCopula(np.full(2000, 0.5))
    u = big.rvs(2500, random_state=0)
    whole = big.logpdf(u)
    np.testing.assert_allclose(whole[:10], big.logpdf(u[:10]), atol=1e-10)


# -- sampling and dependence measures -------------------------------------


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_rvs_margins_are_uniform(family: str) -> None:
    u = small(family).rvs(5000, random_state=2)
    assert u.shape == (5000, 6)
    assert np.all((u > 0) & (u < 1))
    for j in range(6):
        assert stats.kstest(u[:, j], "uniform").pvalue > 1e-3


def test_rvs_is_reproducible() -> None:
    cop = small()
    np.testing.assert_array_equal(cop.rvs(50, random_state=7), cop.rvs(50, random_state=7))


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_kendall_tau_matches_two_over_pi_arcsin_r(family: str) -> None:
    cop = small(family)
    expected = 2 / np.pi * np.arcsin(cop.sigma())
    np.testing.assert_allclose(cop.tau_matrix(), expected, atol=1e-14)
    np.testing.assert_allclose(cop.tau(), P2p(expected), atol=1e-14)
    sample = rc.cor_kendall(cop.rvs(4000, random_state=3))
    assert np.max(np.abs(sample - expected)) < 0.03


def test_spearman_rho_matches_the_elliptical_formulas() -> None:
    gauss = small("gaussian")
    np.testing.assert_allclose(gauss.rho_matrix(), 6 / np.pi * np.arcsin(gauss.sigma() / 2))
    pair = rc.FactorCopula([0.8, 0.6], df=4.0)
    assert pair.rho() == pytest.approx(pair.to_elliptical().rho(), abs=1e-10)
    # 21 distinct correlations: more than the interpolation threshold
    many = rc.FactorCopula(np.linspace(0.2, 0.8, 7), df=4.0)
    rho = many.rho_matrix()
    for i, j in [(0, 1), (2, 5), (5, 6)]:
        exact = rc.FactorCopula(many.market[[i, j]], df=4.0).rho()
        assert rho[i, j] == pytest.approx(exact, abs=1e-6)


def test_student_pairs_have_tail_dependence_and_gaussian_pairs_do_not() -> None:
    student = small("student")
    lam = student.lambda_matrix()
    off = ~np.eye(6, dtype=bool)
    assert np.all(lam[off] > 0.05)
    pair = rc.marginal_copula(student.to_elliptical(), [0, 1])
    assert lam[0, 1] == pytest.approx(pair.lambda_().upper, abs=1e-12)
    np.testing.assert_array_equal(small("gaussian").lambda_matrix()[off], 0.0)
    assert small("gaussian").lambda_() == (0.0, 0.0)
    with pytest.raises(ValueError, match="lambda_matrix"):
        student.lambda_()
    # and it shows: joint 1% crashes are far commoner under the Student-t
    u_t = student.rvs(100_000, random_state=4)
    u_g = small("gaussian").rvs(100_000, random_state=4)
    assert np.mean(np.all(u_t < 0.01, axis=1)) > 3 * np.mean(np.all(u_g < 0.01, axis=1))


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_cdf_agrees_with_the_dense_copula(family: str) -> None:
    cop = small(family)
    pts = np.random.default_rng(5).uniform(0.1, 0.95, (3, 6))
    np.testing.assert_allclose(cop.cdf(pts), cop.to_elliptical().cdf(pts), atol=5e-4)
    # margins: C(u, 1, ..., 1) = u
    assert cop.cdf([0.3, 1, 1, 1, 1, 1])[0] == pytest.approx(0.3, abs=1e-5)
    assert cop.cdf([0.0, 0.5, 0.5, 0.5, 0.5, 0.5])[0] == 0.0


def test_describe_summarises() -> None:
    text = small().describe()
    assert text.startswith("Factor copula (Student-t, df=4.5), dim 6, 1 market + 3 group factors")
    assert text.endswith("13 parameters")
    assert repr(small()) == f"<{text}>"


# -- estimation -----------------------------------------------------------


@pytest.mark.parametrize("family", ["gaussian", "student"])
def test_fit_factor_recovers_the_loadings_of_50_stocks_in_5_groups(family: str) -> None:
    truth = simulated(50, 5, family)
    u = rc.pseudo_obs(truth.rvs(1000, random_state=11))
    fitted = rc.fit_factor(u, groups=truth.groups, family=family)
    assert fitted.family == family and fitted.n_groups == 5
    assert np.mean(np.abs(fitted.market - truth.market)) < 0.04
    assert np.mean(np.abs(fitted.group_loadings - truth.group_loadings)) < 0.05
    if family == "student":
        assert 3.0 < fitted.df < 5.5
    # the fit beats the truth's own likelihood only by sampling noise
    assert fitted.loglik(u) > truth.loglik(u) - 0.05 * abs(truth.loglik(u))


def test_fit_factor_one_factor_and_fixed_df() -> None:
    truth = rc.FactorCopula(np.linspace(0.3, 0.8, 20), df=6.0)
    u = truth.rvs(1500, random_state=12)
    fitted = rc.fit_factor(u, df=6.0)
    assert fitted.df == 6.0 and fitted.groups is None
    assert np.mean(np.abs(fitted.market - truth.market)) < 0.03
    normal = rc.fit_factor(u, family="gaussian", method="normal")
    assert normal.family == "gaussian"
    assert np.mean(np.abs(normal.market - truth.market)) < 0.06


def test_fit_factor_prefers_student_on_student_data() -> None:
    truth = simulated(30, 3, "student", seed=3)
    u = rc.pseudo_obs(truth.rvs(1500, random_state=13))
    t_fit = rc.fit_factor(u, groups=truth.groups)
    g_fit = rc.fit_factor(u, groups=truth.groups, family="gaussian")
    assert t_fit.loglik(u) > g_fit.loglik(u) + 50


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"family": "clayton"}, "family"),
        ({"method": "spearman"}, "method"),
        ({"family": "gaussian", "df": 4.0}, "df applies"),
        ({"max_communality": 1.0}, "max_communality"),
        ({"groups": [0, 1]}, "shape"),
    ],
)
def test_fit_factor_rejects_bad_arguments(kwargs: dict, match: str) -> None:
    u = small().rvs(100, random_state=0)
    with pytest.raises(ValueError, match=match):
        rc.fit_factor(u, **kwargs)


def test_rcopula_fit_dispatches_to_fit_factor() -> None:
    truth = small()
    u = truth.rvs(800, random_state=14)
    result = rc.fit(rc.FactorCopula.unfitted(6, groups=GROUPS_SMALL), u)
    assert isinstance(result.copula, rc.FactorCopula)
    assert len(result.params) == 13 and result.bse is None
    assert result.loglik == pytest.approx(result.copula.loglik(u))
    pinned = rc.FactorCopula.unfitted(6, groups=GROUPS_SMALL).with_params(truth.params)
    pinned = pinned.fix_params([True] * 12 + [False])
    held = rc.fit(pinned, u)
    assert held.copula.df == truth.df and len(held.params) == 12
    with pytest.raises(ValueError, match="pinned"):
        rc.fit(truth.fix_params([False] + [True] * 12), u)


# -- the rest of the package ----------------------------------------------


def test_works_inside_copula_distribution() -> None:
    cop = small()
    dist = rc.CopulaDistribution(cop, [stats.norm(0, 0.02)] * 6)
    x = dist.rvs(500, random_state=15)
    assert x.shape == (500, 6)
    u = np.column_stack([stats.norm(0, 0.02).cdf(x[:, j]) for j in range(6)])
    expected = cop.logpdf(u) + stats.norm(0, 0.02).logpdf(x).sum(axis=1)
    np.testing.assert_allclose(dist.logpdf(x), expected, atol=1e-8)


def test_works_inside_copula_garch() -> None:
    truth = small()
    u = truth.rvs(600, random_state=16)
    returns = 0.01 * stats.t.ppf(u, 6)
    model = CopulaGarch.fit(returns, rc.FactorCopula.unfitted(6, groups=GROUPS_SMALL))
    assert isinstance(model.copula, rc.FactorCopula)
    assert np.mean(np.abs(model.copula.market - truth.market)) < 0.1
    assert model.simulate(horizon=5, n=200, random_state=0).shape == (200, 5, 6)


def test_serialisation_round_trips_exactly() -> None:
    cop = rc.FactorCopula([0.5, 0.6, 0.7, 0.4], [0.3, 0.2, 0.4, 0.1], ["x", "x", "y", "y"], df=5.5)
    back = from_json(to_json(cop))
    assert back == cop
    u = cop.rvs(20, random_state=0)
    np.testing.assert_array_equal(back.logpdf(u), cop.logpdf(u))
    for other in (rc.FactorCopula([0.5, 0.6]), cop.fix_params([True] * 8 + [False])):
        assert from_json(to_json(other)) == other


# -- scale ----------------------------------------------------------------


def test_800_stocks_density_and_sampler_are_fast() -> None:
    truth = simulated(800, 10, "student", seed=32)
    start = time.perf_counter()
    u = truth.rvs(1000, random_state=1)
    ll = truth.logpdf(u)
    elapsed = time.perf_counter() - start
    assert np.all(np.isfinite(ll))
    assert elapsed < 10.0


@pytest.mark.slow
def test_800_stocks_fit_in_well_under_a_minute() -> None:
    truth = simulated(800, 10, "student", seed=32)
    u = rc.pseudo_obs(truth.rvs(1000, random_state=1))
    start = time.perf_counter()
    fitted = rc.fit_factor(u, groups=truth.groups)
    elapsed = time.perf_counter() - start
    assert elapsed < 45.0
    assert np.mean(np.abs(fitted.market - truth.market)) < 0.03
    assert np.mean(np.abs(fitted.group_loadings - truth.group_loadings)) < 0.04
    assert 3.0 < fitted.df < 5.0
