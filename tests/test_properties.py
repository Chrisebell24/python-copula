"""Property-based tests: invariants every copula must satisfy, whatever its parameters.

The 0.3.0 review found ~80 edge-case bugs by hand -- parameters a hair away from
independence, ties, out-of-range inputs, keyword arguments silently ignored.
Each was a violation of something that holds for *every* copula: the Fréchet-
Hoeffding bounds, uniform margins, ``from_tau(tau(c)) == c``, a lossless save and
load. Here those statements are checked across the families and across their
admissible parameter ranges, near-independence included, so the next such bug is
found by the machine.

Runs are derandomized and small by default, so the suite is deterministic and
fast. To search harder (and with fresh randomness) set, e.g.,
``RCOPULA_PROPERTY_EXAMPLES=500``; anything found should become an explicit
``@example`` or a focused regression test.

Where a property genuinely does not hold numerically, its domain is restricted
and the reason given next to the restriction -- never loosened to make it pass.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from scipy import stats

import rcopula as rc
from rcopula import serialize
from rcopula.structural import (
    KhoudrajiCopula,
    MixtureCopula,
    NestedArchimedean,
    RotatedCopula,
)

_EXPLORE = "RCOPULA_PROPERTY_EXAMPLES" in os.environ
_EXAMPLES = int(os.environ.get("RCOPULA_PROPERTY_EXAMPLES", "12"))


def _settings(scale: float = 1.0) -> settings:
    return settings(
        max_examples=max(2, int(_EXAMPLES * scale)),
        deadline=None,
        derandomize=not _EXPLORE,
        database=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )


# ======================================================================
# Families and their parameter strategies
# ======================================================================


@dataclass(frozen=True)
class Family:
    """A one-parameter family, its admissible range, and its independence point."""

    name: str
    build: Callable[[float, int], rc.Copula]
    lo: float
    hi: float
    indep: float | None  # parameter value giving independence (None: only a limit)
    dims: tuple[int, ...] = (2,)
    # Part of the range where inverting tau is numerically well-posed (see uses).
    tau_lo: float | None = None
    tau_hi: float | None = None
    log_scale: bool = False


ONE_PARAMETER = [
    # Clayton admits theta in [-1, 0) in two dimensions only.
    Family("clayton", lambda t, d: rc.ClaytonCopula(t, dim=d), -0.95, 20.0, 0.0, (2,)),
    Family("clayton_d", lambda t, d: rc.ClaytonCopula(t, dim=d), 0.0, 20.0, 0.0, (3, 4)),
    Family("gumbel", lambda t, d: rc.GumbelCopula(t, dim=d), 1.0, 20.0, 1.0, (2, 3, 4)),
    Family("frank", lambda t, d: rc.FrankCopula(t, dim=d), -35.0, 35.0, 0.0, (2,)),
    Family("frank_d", lambda t, d: rc.FrankCopula(t, dim=d), 0.0, 35.0, 0.0, (3, 4)),
    Family("joe", lambda t, d: rc.JoeCopula(t, dim=d), 1.0, 20.0, 1.0, (2, 3, 4)),
    Family("amh", lambda t, d: rc.AMHCopula(t, dim=d), -1.0, 0.99, 0.0, (2,)),
    Family("amh_d", lambda t, d: rc.AMHCopula(t, dim=d), 0.0, 0.99, 0.0, (3,)),
    Family("plackett", lambda t, d: rc.PlackettCopula(t), -5.0, 5.0, 0.0, log_scale=True),
    Family("fgm", lambda t, d: rc.FGMCopula(t), -1.0, 1.0, 0.0),
    Family("galambos", lambda t, d: rc.GalambosCopula(t), 0.02, 8.0, None),
    Family("husler_reiss", lambda t, d: rc.HuslerReissCopula(t), 0.05, 8.0, None),
    Family("tawn", lambda t, d: rc.TawnCopula(t), 0.01, 1.0, None),
    Family("gaussian", lambda t, d: rc.GaussianCopula(t, dim=d), -0.98, 0.98, 0.0, (2,)),
    Family("gaussian_ex", lambda t, d: rc.GaussianCopula(t, dim=d), -0.3, 0.98, 0.0, (3,)),
]
BY_NAME = {f.name: f for f in ONE_PARAMETER}


def _parameter(family: Family) -> st.SearchStrategy[float]:
    """Anywhere in the range, plus the edges and the independence neighbourhood."""
    interior = st.floats(family.lo, family.hi, allow_nan=False, exclude_min=True)
    picks = [interior]
    if family.indep is not None:
        # Within 1e-8 of independence: where 0.3.0's cancellation bugs lived.
        near = st.floats(-1e-8, 1e-8).map(lambda e, f=family: f.indep + e)
        picks.append(near.filter(lambda t, f=family: f.lo < t <= f.hi or t == f.lo == f.indep))
        picks.append(st.just(family.indep))
    out = st.one_of(*picks)
    if family.log_scale:
        out = out.map(np.exp)
    return out


@st.composite
def one_parameter_copula(draw: st.DrawFn, families: list[Family] = ONE_PARAMETER) -> rc.Copula:
    family = draw(st.sampled_from(families))
    theta = draw(_parameter(family))
    dim = draw(st.sampled_from(family.dims))
    try:
        return family.build(float(theta), dim)
    except ValueError:
        # The edge of an *open* range (e.g. Gumbel's theta = 1 is admissible,
        # AMH's theta = 1 is not) -- reject rather than test.
        assume(False)
        raise  # pragma: no cover


@st.composite
def correlation_matrix(draw: st.DrawFn, dim: int) -> np.ndarray:
    """A random, comfortably positive-definite correlation matrix."""
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim + 2))
    s = a @ a.T
    d = np.sqrt(np.diag(s))
    return s / np.outer(d, d)


@st.composite
def elliptical_copula(draw: st.DrawFn) -> rc.Copula:
    dim = draw(st.sampled_from([2, 3, 4]))
    sigma = draw(correlation_matrix(dim))
    params = rc.P2p(sigma)
    if draw(st.booleans()):
        return rc.GaussianCopula(params, dim=dim, dispstr="un")
    df = draw(st.floats(1.5, 40.0))
    return rc.StudentCopula(params, dim=dim, dispstr="un", df=df)


@st.composite
def structural_copula(draw: st.DrawFn) -> rc.Copula:
    kind = draw(st.sampled_from(["rotated", "mixture", "khoudraji", "nested", "factor"]))
    if kind == "rotated":
        base = draw(st.sampled_from([rc.ClaytonCopula, rc.GumbelCopula, rc.JoeCopula]))
        theta = draw(st.floats(1.05, 8.0))
        flip = draw(st.sampled_from([[True, False], [False, True], [True, True]]))
        return RotatedCopula(base(theta), flip)
    if kind == "mixture":
        w = draw(st.floats(0.05, 0.95))
        return MixtureCopula(
            [
                rc.ClaytonCopula(draw(st.floats(0.1, 8.0))),
                rc.GumbelCopula(draw(st.floats(1.0, 8.0))),
            ],
            [w, 1.0 - w],
        )
    if kind == "khoudraji":
        shapes = [draw(st.floats(0.05, 1.0)), draw(st.floats(0.05, 1.0))]
        return KhoudrajiCopula(
            rc.GumbelCopula(draw(st.floats(1.0, 8.0))), rc.IndependenceCopula(), shapes
        )
    if kind == "nested":
        gen = draw(st.sampled_from([rc.ClaytonCopula, rc.GumbelCopula]))
        lo = 0.2 if gen is rc.ClaytonCopula else 1.05
        outer = draw(st.floats(lo, 4.0))
        inner = outer + draw(st.floats(0.0, 4.0))
        return NestedArchimedean(gen(outer), [0], children=[NestedArchimedean(gen(inner), [1, 2])])
    dim = draw(st.sampled_from([3, 4]))
    loadings = draw(st.lists(st.floats(-0.9, 0.9), min_size=dim, max_size=dim))
    if draw(st.booleans()):
        return rc.FactorCopula(loadings)
    return rc.FactorCopula(loadings, df=draw(st.floats(2.0, 30.0)))


any_copula = st.one_of(one_parameter_copula(), elliptical_copula(), structural_copula())


def _cdf_tolerance(cop: rc.Copula) -> float:
    """Elliptical CDFs above two dimensions are quasi-Monte-Carlo (~1e-6)."""
    if isinstance(cop, (rc.EllipticalCopula, rc.FactorCopula)) and cop.dim > 2:
        return 2e-5
    if isinstance(cop, rc.StudentCopula) and cop.df < 3.0:
        # The radial quadrature in special.mvtnorm.mvt_cdf integrates a density
        # with an s**(df - 1) factor at the origin; below df = 2 it switches to
        # the probability scale, whose documented accuracy is ~1e-7, and just
        # above 2 the density form carries ~2e-10 (C(u, 1) - u at df = 2.5).
        return 2e-7
    return 1e-10


def _points(seed: int, n: int, d: int, eps: float) -> np.ndarray:
    return np.random.default_rng(seed).uniform(eps, 1.0 - eps, size=(n, d))


# ======================================================================
# Sampling
# ======================================================================


@_settings(0.75)
@given(cop=any_copula, seed=st.integers(0, 2**32 - 1))
def test_rvs_margins_are_uniform(cop: rc.Copula, seed: int) -> None:
    n = 1500
    u = cop.rvs(n, random_state=seed)
    assert u.shape == (n, cop.dim)
    assert np.all((u >= 0.0) & (u <= 1.0))
    for j in range(cop.dim):
        # A deterministic run cannot be "unlucky" twice, and 1e-6 over a few
        # hundred margins is a false-alarm rate of ~1e-4 for exploratory runs.
        assert stats.kstest(u[:, j], "uniform").pvalue > 1e-6, (cop, j)


# ======================================================================
# Distribution function
# ======================================================================


@_settings()
@given(cop=any_copula, seed=st.integers(0, 2**32 - 1))
def test_cdf_lies_within_frechet_hoeffding_bounds(cop: rc.Copula, seed: int) -> None:
    u = _points(seed, 25, cop.dim, 0.0)
    c = cop.cdf(u)
    tol = _cdf_tolerance(cop)
    lower = np.maximum(u.sum(axis=1) - cop.dim + 1.0, 0.0)
    upper = u.min(axis=1)
    assert np.all(c >= lower - tol), cop
    assert np.all(c <= upper + tol), cop


@_settings()
@given(cop=any_copula, seed=st.integers(0, 2**32 - 1))
def test_cdf_is_monotone_in_each_argument(cop: rc.Copula, seed: int) -> None:
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(15, cop.dim))
    j = int(rng.integers(cop.dim))
    v = u.copy()
    v[:, j] = u[:, j] + (1.0 - u[:, j]) * rng.uniform(size=15)
    assert np.all(cop.cdf(v) >= cop.cdf(u) - _cdf_tolerance(cop)), cop


@_settings()
@given(cop=any_copula, seed=st.integers(0, 2**32 - 1))
def test_cdf_has_uniform_margins_and_vanishes_on_the_lower_faces(cop: rc.Copula, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=10)
    tol = _cdf_tolerance(cop)
    for j in range(cop.dim):
        u = np.ones((10, cop.dim))
        u[:, j] = x
        np.testing.assert_allclose(cop.cdf(u), x, atol=tol, err_msg=repr(cop))
        w = rng.uniform(size=(10, cop.dim))
        w[:, j] = 0.0
        np.testing.assert_allclose(cop.cdf(w), 0.0, atol=tol, err_msg=repr(cop))


# ======================================================================
# Density
# ======================================================================


def _has_density(cop: rc.Copula) -> bool:
    return not isinstance(cop, NestedArchimedean)  # not implemented (nor in R)


@_settings()
@given(cop=any_copula, seed=st.integers(0, 2**32 - 1))
def test_logpdf_is_finite_inside_the_cube_and_matches_pdf(cop: rc.Copula, seed: int) -> None:
    assume(_has_density(cop))
    u = _points(seed, 30, cop.dim, 1e-6)
    logpdf = cop.logpdf(u)
    assert np.all(np.isfinite(logpdf)), (cop, u[~np.isfinite(logpdf)])
    pdf = cop.pdf(u)
    np.testing.assert_allclose(pdf, np.exp(logpdf), rtol=1e-10, err_msg=repr(cop))


# ======================================================================
# Rosenblatt transform
# ======================================================================


def _supports_rosenblatt(cop: rc.Copula) -> bool:
    if isinstance(cop, (rc.ArchimedeanCopula, rc.EllipticalCopula)):
        return True
    return cop.dim == 2 and _has_density(cop)


@_settings()
@given(cop=any_copula, seed=st.integers(0, 2**32 - 1))
def test_rosenblatt_and_its_inverse_round_trip(cop: rc.Copula, seed: int) -> None:
    assume(_supports_rosenblatt(cop))
    z = _points(seed, 12, cop.dim, 1e-3)
    u = rc.inverse_rosenblatt(cop, z)
    assert np.all((u >= 0.0) & (u <= 1.0))
    np.testing.assert_allclose(rc.rosenblatt(cop, u), z, atol=1e-7, err_msg=repr(cop))


# ======================================================================
# Calibration: from_tau / from_rho invert tau / rho
# ======================================================================


@_settings(1.5)
@given(data=st.data())
def test_from_tau_inverts_tau(data: st.DataObject) -> None:
    family = data.draw(st.sampled_from(ONE_PARAMETER))
    theta = float(data.draw(_parameter(family)))
    try:
        cop = family.build(theta, 2)
    except ValueError:
        assume(False)
        return
    tau = cop.tau()
    assume(abs(tau) < 0.97)  # tau -> +-1 has a vanishing derivative; theta is undetermined
    back = type(cop).from_tau(tau)
    assert back.tau() == pytest.approx(tau, abs=1e-9), (cop, back)
    # On the parameter scale, error is amplified by dtheta/dtau; compare with it.
    h = 1e-6
    slope = abs(_tau_slope(cop, h))
    assert abs(float(back.params[0]) - theta) <= 1e-8 + 1e-7 * (1.0 + slope), (cop, back)


def _tau_slope(cop: rc.Copula, h: float) -> float:
    """dtheta/dtau at cop, by central differences of from_tau."""
    tau = cop.tau()
    cls = type(cop)
    try:
        hi = float(cls.from_tau(min(tau + h, 0.999)).params[0])
        lo = float(cls.from_tau(max(tau - h, -0.999)).params[0])
    except ValueError:
        return np.inf
    return (hi - lo) / (2 * h)


@_settings()
@given(data=st.data())
def test_from_rho_inverts_rho(data: st.DataObject) -> None:
    family = data.draw(
        st.sampled_from(
            [BY_NAME[n] for n in ("clayton", "gumbel", "frank", "plackett", "fgm", "gaussian")]
        )
    )
    theta = float(data.draw(_parameter(family)))
    cop = family.build(theta, 2)
    rho = cop.rho()
    assume(abs(rho) < 0.97)
    back = type(cop).from_rho(rho)
    assert back.rho() == pytest.approx(rho, abs=1e-8), (cop, back)


@_settings()
@given(
    name=st.sampled_from(
        ["clayton", "gumbel", "frank", "joe", "amh", "plackett", "fgm", "gaussian"]
    ),
    eps=st.floats(0.0, 1e-8),
)
def test_independence_neighbourhood_is_continuous(name: str, eps: float) -> None:
    """tau, rho, density and CDF are continuous through the independence point."""
    family = BY_NAME[name]
    assert family.indep is not None
    theta = family.indep + eps
    if family.log_scale:
        theta = float(np.exp(theta))
    cop = family.build(theta, 2)
    u = _points(7, 20, 2, 1e-4)
    assert abs(cop.tau()) < 1e-6
    assert abs(cop.rho()) < 1e-6
    np.testing.assert_allclose(cop.cdf(u), u.prod(axis=1), atol=1e-7)
    np.testing.assert_allclose(cop.logpdf(u), 0.0, atol=1e-6)


# ======================================================================
# Estimation
# ======================================================================


@pytest.mark.slow
@_settings(0.5)
@given(data=st.data(), seed=st.integers(0, 2**32 - 1))
def test_fit_recovers_the_parameter_on_a_large_sample(data: st.DataObject, seed: int) -> None:
    family = data.draw(
        st.sampled_from(
            [BY_NAME[n] for n in ("clayton", "gumbel", "frank", "joe", "plackett", "gaussian")]
        )
    )
    theta = float(data.draw(_parameter(family)))
    truth = family.build(theta, 2)
    assume(abs(truth.tau()) < 0.9)
    u = rc.pseudo_obs(truth.rvs(3000, random_state=seed))
    res = rc.fit(type(truth)(), u, method="mpl", estimate_variance=False)
    # Compare on the tau scale: theta itself is ill-determined where tau is flat.
    assert res.copula.tau() == pytest.approx(truth.tau(), abs=0.04), (truth, res.copula)


@_settings(0.5)
@given(cop=elliptical_copula(), seed=st.integers(0, 2**32 - 1))
def test_itau_recovers_unstructured_correlations(cop: rc.Copula, seed: int) -> None:
    """Fast (closed-form) estimation check, in up to four dimensions."""
    u = cop.rvs(4000, random_state=seed)
    family = (
        rc.GaussianCopula(dim=cop.dim, dispstr="un")
        if isinstance(cop, rc.GaussianCopula)
        else rc.StudentCopula(dim=cop.dim, dispstr="un", df=cop.df, df_fixed=True)
    )
    res = rc.fit(family, u, method="itau", estimate_variance=False)
    # Six standard errors of a correlation estimate, (1 - rho^2) / sqrt(n),
    # with 25% for itau's lower efficiency: a fixed 0.06 was a ~1-in-1500 event
    # per matrix (checked against 3000 NumPy reference samples), which a
    # 200-example exploratory run hits.
    sigma = cop.sigma()
    tol = 6.0 * 1.25 * (1.0 - sigma**2) / np.sqrt(u.shape[0]) + 1e-3
    assert np.all(np.abs(res.copula.sigma() - sigma) <= tol), (cop, res.copula)


# ======================================================================
# Serialization
# ======================================================================


@_settings()
@given(cop=any_copula, seed=st.integers(0, 2**32 - 1))
def test_json_round_trip_is_lossless(cop: rc.Copula, seed: int) -> None:
    back = serialize.from_json(serialize.to_json(cop))
    assert type(back) is type(cop)
    u = _points(seed, 10, cop.dim, 1e-3)
    values = (lambda c: c.logpdf(u)) if _has_density(cop) else (lambda c: c.cdf(u))
    np.testing.assert_array_equal(back.params, cop.params)
    np.testing.assert_array_equal(values(back), values(cop))


# ======================================================================
# Ranks and rank correlations
# ======================================================================


TIES = ["average", "min", "max", "dense", "ordinal"]


@st.composite
def tied_data(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(2, 40))
    d = draw(st.integers(1, 4))
    levels = draw(st.integers(1, 6))  # few levels: plenty of ties, maybe constant columns
    values = draw(st.lists(st.integers(0, levels), min_size=n * d, max_size=n * d))
    return np.asarray(values, dtype=float).reshape(n, d)


@_settings(2.0)
@given(x=tied_data(), method=st.sampled_from(TIES))
def test_pseudo_obs_are_scaled_ranks(x: np.ndarray, method: str) -> None:
    u = np.asarray(rc.pseudo_obs(x, ties_method=method))
    n = x.shape[0]
    assert u.shape == x.shape
    assert np.all((u > 0.0) & (u < 1.0))
    for j in range(x.shape[1]):
        np.testing.assert_allclose(u[:, j] * (n + 1), stats.rankdata(x[:, j], method=method))


@_settings(2.0)
@given(x=tied_data(), seed=st.integers(0, 2**32 - 1))
def test_random_tie_breaking_gives_a_valid_ranking(x: np.ndarray, seed: int) -> None:
    u = np.asarray(rc.pseudo_obs(x, ties_method="random", random_state=seed))
    n = x.shape[0]
    for j in range(x.shape[1]):
        ranks = u[:, j] * (n + 1)
        np.testing.assert_allclose(np.sort(ranks), np.arange(1, n + 1))
        # Distinct values keep their order; only ties are permuted.
        lo, hi = stats.rankdata(x[:, j], "min"), stats.rankdata(x[:, j], "max")
        assert np.all((ranks >= lo - 1e-9) & (ranks <= hi + 1e-9))


@_settings(2.0)
@given(x=tied_data())
def test_cor_kendall_matches_scipy_pairwise(x: np.ndarray) -> None:
    assume(x.shape[1] >= 2)
    got = rc.cor_kendall(x)
    d = x.shape[1]
    for i in range(d):
        for j in range(d):
            expected = 1.0 if i == j else stats.kendalltau(x[:, i], x[:, j]).statistic
            if np.isnan(expected):
                assert np.isnan(got[i, j]) or i == j, (i, j)
            else:
                assert got[i, j] == pytest.approx(expected, abs=1e-12), (i, j)


# ======================================================================
# Constructors reject what they would otherwise ignore
# ======================================================================


CONSTRUCTORS: list[Callable[..., object]] = [
    lambda **kw: rc.ClaytonCopula(2.0, **kw),
    lambda **kw: rc.GumbelCopula(2.0, **kw),
    lambda **kw: rc.FrankCopula(2.0, **kw),
    lambda **kw: rc.JoeCopula(2.0, **kw),
    lambda **kw: rc.AMHCopula(0.5, **kw),
    lambda **kw: rc.GaussianCopula(0.5, **kw),
    lambda **kw: rc.StudentCopula(0.5, **kw),
    lambda **kw: rc.PlackettCopula(2.0, **kw),
    lambda **kw: rc.FGMCopula(0.5, **kw),
    lambda **kw: rc.GalambosCopula(1.0, **kw),
    lambda **kw: rc.HuslerReissCopula(1.0, **kw),
    lambda **kw: rc.TawnCopula(0.5, **kw),
    lambda **kw: rc.TEVCopula(0.5, **kw),
    lambda **kw: rc.MarshallOlkinCopula(0.3, 0.6, **kw),
    lambda **kw: rc.IndependenceCopula(2, **kw),
    lambda **kw: rc.FrechetUpperCopula(2, **kw),
    lambda **kw: rc.FrechetLowerCopula(**kw),
    lambda **kw: rc.EmpiricalCopula(np.random.default_rng(0).uniform(size=(20, 2)), **kw),
    lambda **kw: RotatedCopula(rc.ClaytonCopula(2.0), **kw),
    lambda **kw: MixtureCopula([rc.ClaytonCopula(2.0), rc.GumbelCopula(2.0)], **kw),
    lambda **kw: KhoudrajiCopula(rc.GumbelCopula(2.0), rc.IndependenceCopula(), [0.5, 0.5], **kw),
    lambda **kw: NestedArchimedean(rc.ClaytonCopula(1.0), [0, 1], **kw),
    lambda **kw: rc.FactorCopula([0.5, 0.4, 0.3], **kw),
]


@pytest.mark.parametrize("ctor", CONSTRUCTORS)
@pytest.mark.parametrize("kwarg", ["thetaa", "degrees_of_freedom", "random_state"])
def test_constructors_reject_unknown_keywords(ctor: Callable[..., object], kwarg: str) -> None:
    ctor()  # the call itself is fine ...
    with pytest.raises(TypeError):
        ctor(**{kwarg: 1})  # ... and a misspelt or foreign keyword is not ignored
