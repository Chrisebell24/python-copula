"""Parity tests in four and five dimensions against R's ``copula`` package.

Fixtures come from ``tools/rgolden/11_highdim.R``; regenerate with
``make golden``.

**Why these exist.** Every other multi-parameter fixture stops at ``d <= 3``,
where the row-major and column-major orders of the lower triangle coincide --
``(2,1), (3,1), (3,2)`` either way. A correlation vector read in the wrong order
passes all of them, which is how the ``p2P``/``P2p`` ordering bug fixed in 0.2.0
went unnoticed. Here every correlation vector has distinct entries and R's
``getSigma()`` is recorded alongside, so an ordering error is a hard failure.

Tolerances follow the existing golden tests: densities are deterministic on both
sides (``~1e-10`` relative); elliptical CDFs are quasi-Monte-Carlo on both sides
(R's ``GenzBretz(abseps=1e-8)``), so ``1e-5`` absolute; Archimedean CDFs are
closed form, so tight. Fits use the sample R drew, written into the fixture, so
the estimates are deterministic: ``1e-6`` relative where both sides solve the
same closed-form problem, looser for optimiser output. Inversion standard errors
use a slightly different (asymptotically equivalent) empirical influence
function from R's, hence 5% as in ``test_fitting.py``.

Two things R does differently, pinned rather than hidden:

* **One-parameter inversion in ``d > 2``** (Archimedean, and elliptical
  ``"ex"``). R inverts each pairwise statistic and averages the *parameters*;
  this package averages the *statistics* and inverts once. Both are consistent;
  they differ by O(1/n). ``test_one_parameter_inversion_differs_from_r_by_design``
  pins both relations exactly.
* **R's inverse Rosenblatt transform** for Gumbel, Frank and Joe is a loose
  root find (round trip off by up to 1.7e-3); there the round trip, not R, is
  the oracle.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.structural import NestedArchimedean

pytestmark = pytest.mark.golden

GOLDEN = Path(__file__).parent / "golden" / "highdim.json"

TOL_PDF = 5e-10  # relative
TOL_LOGPDF = 5e-8  # relative, on log densities that pass near zero
TOL_CDF_QMC = 1e-5  # absolute; Genz-Bretz on both sides
TOL_CDF_CLOSED = 1e-9  # closed form; as test_golden_archimedean (Joe near 1 reaches 1e-11)

ARCHIMEDEAN = {
    "clayton": rc.ClaytonCopula,
    "gumbel": rc.GumbelCopula,
    "frank": rc.FrankCopula,
    "joe": rc.JoeCopula,
    "amh": rc.AMHCopula,
}


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run `make golden`")
    return json.loads(GOLDEN.read_text())


def _numeric(value: object) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=object))
    return np.array([np.nan if (v is None or v == "NA") else float(v) for v in arr.ravel()])


def _matrix(value: object) -> np.ndarray:
    return np.atleast_2d(np.asarray(value, dtype=float))


def _cases(blob: dict, kind: str) -> list[str]:
    return sorted(k for k in blob if not k.startswith("_") and blob[k]["kind"] == kind)


def _elliptical(family: str, rho: object, dim: int, dispstr: str, df: object) -> rc.Copula:
    params = _numeric(rho)
    if family == "normal":
        return rc.GaussianCopula(params, dim=dim, dispstr=dispstr)
    return rc.StudentCopula(params, dim=dim, dispstr=dispstr, df=float(_numeric(df)[0]))


def _rel(got: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(got - expected) / np.maximum(np.abs(expected), 1e-12)))


# ----------------------------------------------------------------------
# Elliptical copulas at fixed points
# ----------------------------------------------------------------------


def _ell(blk: dict) -> rc.Copula:
    return _elliptical(blk["family"], blk["rho"], blk["dim"], blk["dispstr"], blk["df"])


def test_correlation_matrix_layout_matches_r(golden: dict) -> None:
    """The ordering guard: distinct entries, so any transposition fails."""
    for key in _cases(golden, "elliptical"):
        blk = golden[key]
        np.testing.assert_allclose(
            _ell(blk).sigma(), _matrix(blk["sigma"]), atol=1e-15, err_msg=key
        )
        if blk["dispstr"] == "un":
            # And P2p reads it back in R's order.
            np.testing.assert_array_equal(rc.P2p(_matrix(blk["sigma"])), _numeric(blk["rho"]))


@pytest.mark.parametrize("quantity", ["pdf", "logpdf"])
def test_elliptical_density_matches_r(golden: dict, quantity: str) -> None:
    worst, worst_case = 0.0, ""
    for key in _cases(golden, "elliptical"):
        blk = golden[key]
        got = getattr(_ell(blk), quantity)(_matrix(blk["u"]))
        rel = _rel(got, _numeric(blk[quantity]))
        if rel > worst:
            worst, worst_case = rel, key
    tol = TOL_PDF if quantity == "pdf" else TOL_LOGPDF
    assert worst < tol, f"{quantity}: worst rel dev {worst:.3e} at {worst_case}"


def test_elliptical_cdf_matches_r(golden: dict) -> None:
    for key in _cases(golden, "elliptical"):
        blk = golden[key]
        got = _ell(blk).cdf(_matrix(blk["u"]))
        err = float(np.max(np.abs(got - _numeric(blk["cdf"]))))
        assert err < TOL_CDF_QMC, f"{key}: max abs cdf deviation {err:.3e}"


def _per_pair(blk: dict, per_parameter: np.ndarray) -> np.ndarray:
    """Spread R's per-parameter measure over the pairs, in P2p order.

    For ``"toep"`` and ``"ar1"`` R returns one value per *parameter* (per lag
    for Toeplitz, the lag-1 value for AR(1)); rcopula returns one per pair.
    """
    d = blk["dim"]
    lags = rc.P2p(np.abs(np.subtract.outer(np.arange(d), np.arange(d)))).astype(int)
    if blk["dispstr"] == "toep":
        return per_parameter[lags - 1]
    return per_parameter


def test_elliptical_dependence_measures_match_r(golden: dict) -> None:
    compared_rho = 0
    for key in _cases(golden, "elliptical"):
        blk = golden[key]
        cop = _ell(blk)
        pairs = rc.P2p(cop.sigma())
        tau = np.atleast_1d(cop.tau())
        expected_tau = _numeric(blk["tau"])
        if blk["dispstr"] == "ar1":
            # R reports tau of the lag-1 correlation only.
            assert expected_tau.size == 1
            assert tau[0] == pytest.approx(expected_tau[0], rel=1e-12), key
            np.testing.assert_allclose(tau, 2.0 / np.pi * np.arcsin(pairs), rtol=1e-12)
        else:
            np.testing.assert_allclose(
                tau, _per_pair(blk, expected_tau), rtol=1e-12, atol=1e-15, err_msg=key
            )
        expected_rho = _numeric(blk["rho_s"])
        if np.isnan(expected_rho).any():
            assert blk["family"] == "t"  # R has no Spearman's rho for the t copula
            continue
        rho = np.atleast_1d(cop.rho())
        if blk["dispstr"] == "ar1":
            assert rho[0] == pytest.approx(expected_rho[0], rel=1e-12), key
        else:
            np.testing.assert_allclose(
                rho, _per_pair(blk, expected_rho), rtol=1e-12, atol=1e-15, err_msg=key
            )
        compared_rho += 1
    assert compared_rho == 6


# ----------------------------------------------------------------------
# Fits on R's exported draws
# ----------------------------------------------------------------------


def _family_for_fit(blk: dict) -> rc.Copula:
    family, dim = blk["family"], blk["dim"]
    if family == "normal":
        return rc.GaussianCopula(dim=dim, dispstr=blk["dispstr"])
    if family == "t":
        return rc.StudentCopula(dim=dim, dispstr=blk["dispstr"])
    if family == "t_dffixed":
        df = float(_numeric(blk["df"])[0])
        return rc.StudentCopula(dim=dim, dispstr=blk["dispstr"], df=df, df_fixed=True)
    return ARCHIMEDEAN[family](dim=dim)


_FIT_CACHE: dict[tuple[int, str], rc.CopulaFitResult] = {}


def _fit(blk: dict, method: str) -> rc.CopulaFitResult:
    """Fit once per (fixture block, method); several tests read the same fit."""
    cache_key = (id(blk), method)
    if cache_key not in _FIT_CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _FIT_CACHE[cache_key] = rc.fit(_family_for_fit(blk), _matrix(blk["u"]), method=method)
    return _FIT_CACHE[cache_key]


def _one_parameter_by_inversion(blk: dict) -> bool:
    return blk["family"] in ARCHIMEDEAN or blk.get("dispstr") == "ex"


@pytest.mark.parametrize("method", ["itau", "irho"])
def test_inversion_estimates_match_r(golden: dict, method: str) -> None:
    compared = 0
    for key in _cases(golden, "fit"):
        blk = golden[key]
        expected = blk.get(f"est_{method}")
        if expected is None or _one_parameter_by_inversion(blk):
            continue
        got = _fit(blk, method).params
        np.testing.assert_allclose(got, _numeric(expected), rtol=1e-6, atol=1e-9, err_msg=key)
        compared += 1
    assert compared >= (5 if method == "itau" else 4)


def test_one_parameter_inversion_differs_from_r_by_design(golden: dict) -> None:
    """R averages pairwise *inverses*; rcopula inverts the averaged statistic.

    Both relations are checked exactly, so a change on either side is noticed.
    """
    for key in _cases(golden, "fit"):
        blk = golden[key]
        if not _one_parameter_by_inversion(blk):
            continue
        u = _matrix(blk["u"])
        d = u.shape[1]
        pairs = [(i, j) for j in range(d) for i in range(j + 1, d)]
        taus = np.array([stats.kendalltau(u[:, i], u[:, j]).statistic for i, j in pairs])
        family = _family_for_fit(blk)
        if isinstance(family, rc.GaussianCopula):
            r_value = np.mean(np.sin(np.pi * taus / 2.0))
            ours = np.sin(np.pi * np.mean(taus) / 2.0)
        else:
            cls = type(family)
            r_value = np.mean([cls.from_tau(t).params[0] for t in taus])
            ours = cls.from_tau(float(np.mean(taus))).params[0]
        expected_r = float(_numeric(blk["est_itau"])[0])
        assert r_value == pytest.approx(expected_r, rel=1e-10), key
        assert float(_fit(blk, "itau").params[0]) == pytest.approx(ours, rel=1e-10), key
        # ... and the two estimators are within sampling noise of each other.
        se = float(_numeric(blk["se_itau"])[0])
        assert abs(ours - expected_r) < 0.25 * se, key


@pytest.mark.parametrize("method", ["itau", "irho"])
def test_inversion_standard_errors_match_r(golden: dict, method: str) -> None:
    compared = 0
    for key in _cases(golden, "fit"):
        blk = golden[key]
        expected = blk.get(f"se_{method}")
        if expected is None:
            continue
        got = _fit(blk, method).bse
        assert got is not None, key
        np.testing.assert_allclose(got, _numeric(expected), rtol=0.05, err_msg=key)
        compared += 1
    assert compared >= (7 if method == "itau" else 5)


def test_mpl_estimates_match_r(golden: dict) -> None:
    """Both maximise the same pseudo-likelihood; R's optimiser is the looser one.

    rcopula's maximised log-likelihood is never below R's (it is up to 1.7e-7
    *above* it on the 6-11 parameter unstructured fits), so the remaining
    difference in the estimates -- up to 5e-5 relative, a few thousandths of a
    standard error, on a surface that flat -- is R stopping early. One-parameter
    fits agree to 1e-7.
    """
    compared = 0
    for key in _cases(golden, "fit"):
        blk = golden[key]
        expected = _numeric(blk["est_mpl"])
        res = _fit(blk, "mpl")
        rtol = 1e-6 if expected.size == 1 else 2e-4
        np.testing.assert_allclose(res.params, expected, rtol=rtol, atol=1e-5, err_msg=key)
        loglik = float(_numeric(blk["loglik_mpl"])[0])
        assert res.loglik >= loglik - 1e-9, key
        assert res.loglik == pytest.approx(loglik, rel=1e-8, abs=1e-6), key
        compared += 1
    assert compared == 10


def test_itau_mpl_estimates_match_r(golden: dict) -> None:
    compared = 0
    for key in _cases(golden, "fit"):
        blk = golden[key]
        expected = blk.get("est_itau.mpl")
        if expected is None:
            continue
        got = _fit(blk, "itau.mpl").params
        expected_arr = _numeric(expected)
        # Correlations come from tau: closed form on both sides.
        np.testing.assert_allclose(got[:-1], expected_arr[:-1], rtol=1e-6, atol=1e-9, err_msg=key)
        # df from a one-dimensional bounded search.
        assert got[-1] == pytest.approx(expected_arr[-1], rel=1e-4), key
        compared += 1
    assert compared == 2


def test_mpl_standard_errors(golden: dict) -> None:
    """Agree with R to the precision two different plug-ins allow -- 20%.

    Both sides estimate the Genest-Ghoudi-Rivest sandwich, but with different
    empirical ingredients: R (``Jscore``) uses the outer product of scores as
    the information and the product form ``dlogc/dtheta * dlogc/du`` in the rank
    correction; rcopula uses the observed Hessian and the mixed derivative
    ``d2 logc / dtheta du``. These are asymptotically equivalent and differ
    sample by sample (``var_mpl``'s docstring records a 0.91-1.12 paired range
    on Clayton). On these samples: up to 14.8% (5-d Gaussian ``un``), 13.4%
    (4-d Gaussian ``un``), 10.5% (Gumbel d = 5), under 7% elsewhere.

    Averaged over samples the two agree with each other and with the truth.
    4-d Gaussian ``un``, per correlation: R's mean over 25 samples
    (0.0199, 0.0310, 0.0283, 0.0250, 0.0321, 0.0296), rcopula's over 30
    (0.0191, 0.0307, 0.0273, 0.0248, 0.0315, 0.0292), Monte-Carlo SD over 300
    (0.0209, 0.0311, 0.0271, 0.0241, 0.0327, 0.0297). Gumbel d = 5: R 0.0397,
    rcopula 0.0386, Monte-Carlo 0.043.

    For the t copula with ``df`` free R computes the correlations' variance "as
    if df were fixed" and reports none for ``df``; only the correlations are
    compared.
    """
    compared = 0
    for key in _cases(golden, "fit"):
        blk = golden[key]
        res = _fit(blk, "mpl")
        expected = _numeric(blk["se_mpl"])
        assert res.bse is not None, key
        got = np.asarray(res.bse, dtype=float)
        if blk["family"] == "t":
            assert np.isnan(expected[-1]) and np.isfinite(got[-1]), key
            got, expected = got[:-1], expected[:-1]
        np.testing.assert_allclose(got, expected, rtol=0.2, err_msg=key)
        compared += 1
    assert compared == 10


# ----------------------------------------------------------------------
# Archimedean and nested Archimedean
# ----------------------------------------------------------------------


@pytest.mark.parametrize("quantity", ["pdf", "logpdf", "cdf"])
def test_archimedean_matches_r(golden: dict, quantity: str) -> None:
    worst, worst_case = 0.0, ""
    for key in _cases(golden, "archimedean"):
        blk = golden[key]
        cop = ARCHIMEDEAN[blk["family"]](blk["theta"], dim=blk["dim"])
        got = getattr(cop, quantity)(_matrix(blk["u"]))
        expected = _numeric(blk[quantity])
        rel = _rel(got, expected)
        if rel > worst:
            worst, worst_case = rel, key
    tol = {"pdf": TOL_PDF, "logpdf": TOL_LOGPDF, "cdf": TOL_CDF_CLOSED}[quantity]
    assert worst < tol, f"{quantity}: worst rel dev {worst:.3e} at {worst_case}"


def _nested(blk: dict) -> NestedArchimedean:
    gen = ARCHIMEDEAN[blk["family"]]
    children = [
        NestedArchimedean(gen(float(ch["theta"])), [int(j) for j in np.atleast_1d(ch["comp"])])
        for ch in blk["children"]
    ]
    # jsonlite unboxes a length-one vector, so `root_comp` is [], an int, or a list.
    comps = [int(j) for j in np.atleast_1d(np.asarray(blk["root_comp"], dtype=int))] or None
    return NestedArchimedean(gen(float(blk["root"])), comps, children=children)


def test_nested_archimedean_cdf_matches_r(golden: dict) -> None:
    cases = _cases(golden, "nested")
    assert {golden[k]["dim"] for k in cases} == {4, 5}
    for key in cases:
        blk = golden[key]
        cop = _nested(blk)
        assert cop.dim == blk["dim"]
        got = cop.cdf(_matrix(blk["u"]))
        expected = _numeric(blk["cdf"])
        assert _rel(got, expected) < TOL_CDF_CLOSED, key


def test_nested_archimedean_density_is_missing_on_both_sides(golden: dict) -> None:
    """R's dCopula refuses nested copulas; so does rcopula. Pin the gap."""
    for key in _cases(golden, "nested"):
        blk = golden[key]
        assert np.isnan(_numeric(blk["pdf"])).all(), "R now has a nested density; compare it"
        with pytest.raises(NotImplementedError):
            _nested(blk).pdf(_matrix(blk["u"])[:2])


# ----------------------------------------------------------------------
# Marginal copulas and the Rosenblatt transform
# ----------------------------------------------------------------------


def test_marginal_copula_matches_r(golden: dict) -> None:
    cases = _cases(golden, "marginal")
    assert len(cases) == 6
    for key in cases:
        blk = golden[key]
        full = _elliptical(blk["family"], blk["rho"], 5, "un", blk["df"])
        keep = [int(j) for j in np.atleast_1d(blk["keep"])]
        margin = rc.marginal_copula(full, keep)
        np.testing.assert_allclose(margin.params, _numeric(blk["params"]), rtol=0, atol=0)
        np.testing.assert_allclose(margin.sigma(), _matrix(blk["sigma"]), atol=1e-15)
        u = _matrix(blk["u"])
        assert _rel(margin.pdf(u), _numeric(blk["pdf"])) < TOL_PDF, key
        assert _rel(margin.logpdf(u), _numeric(blk["logpdf"])) < TOL_LOGPDF, key


def _rosenblatt_copula(blk: dict) -> rc.Copula:
    params = _numeric(blk["params"])
    if blk["family"] in ("normal", "t"):
        rho = params if blk["family"] == "normal" else params[:-1]
        return _elliptical(blk["family"], rho, 4, blk["dispstr"], blk["df"])
    return ARCHIMEDEAN[blk["family"]](float(params[0]), dim=4)


def test_rosenblatt_matches_r(golden: dict) -> None:
    cases = _cases(golden, "rosenblatt")
    assert len(cases) == 7
    for key in cases:
        blk = golden[key]
        got = rc.rosenblatt(_rosenblatt_copula(blk), _matrix(blk["u"]))
        np.testing.assert_allclose(got, _matrix(blk["forward"]), rtol=0, atol=1e-10, err_msg=key)


#: Families whose conditional distributions R inverts in closed form (the
#: elliptical ones through t/normal quantiles, Clayton algebraically).
_CLOSED_FORM_INVERSE = ("normal", "t", "clayton")


def test_inverse_rosenblatt_matches_r_where_r_is_closed_form(golden: dict) -> None:
    compared = 0
    for key in _cases(golden, "rosenblatt"):
        blk = golden[key]
        if blk["family"] not in _CLOSED_FORM_INVERSE:
            continue
        got = rc.inverse_rosenblatt(_rosenblatt_copula(blk), _matrix(blk["u"]))
        np.testing.assert_allclose(got, _matrix(blk["inverse"]), rtol=0, atol=1e-10, err_msg=key)
        compared += 1
    assert compared == 4


def test_inverse_rosenblatt_is_more_accurate_than_r_elsewhere(golden: dict) -> None:
    """For Gumbel, Frank and Joe R's ``cCopula(inverse=TRUE)`` is a loose root find.

    Feeding R's inverse back through the (agreed, 1e-10) forward transform
    misses the input by up to 1.7e-3; rcopula's inverse round-trips to 1e-12.
    So R is checked only to its own accuracy, and the round trip is the oracle.
    """
    compared = 0
    for key in _cases(golden, "rosenblatt"):
        blk = golden[key]
        if blk["family"] in _CLOSED_FORM_INVERSE:
            continue
        cop = _rosenblatt_copula(blk)
        u = _matrix(blk["u"])
        ours, theirs = rc.inverse_rosenblatt(cop, u), _matrix(blk["inverse"])
        assert np.max(np.abs(rc.rosenblatt(cop, ours) - u)) < 1e-12, key
        assert np.max(np.abs(ours - theirs)) < 1e-4, key
        # Pin R's error, so a future fix in R is noticed and the bound tightened.
        assert np.max(np.abs(rc.rosenblatt(cop, theirs) - u)) > 1e-6, key
        compared += 1
    assert compared == 3


def test_coverage(golden: dict) -> None:
    seen = {
        (golden[k]["family"], golden[k]["dim"], golden[k]["dispstr"])
        for k in _cases(golden, "elliptical")
    }
    for family in ("normal", "t"):
        for dim in (4, 5):
            for dispstr in ("un", "toep", "ar1"):
                assert (family, dim, dispstr) in seen
    arch = {(golden[k]["family"], golden[k]["dim"]) for k in _cases(golden, "archimedean")}
    assert arch == {(f, d) for f in ARCHIMEDEAN for d in (4, 5)}
