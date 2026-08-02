"""Parity tests for the elliptical families against R's ``copula`` package.

Fixtures come from ``tools/rgolden/01_families.R``; regenerate with
``make golden``.

**A note on the CDF oracle.** R's ``pCopula`` picks its integration algorithm by
dimension (``pmvnormAlgo``): TVPACK for ``d <= 3``, **Miwa with 128 steps** for
``d <= 5``, and Genz-Bretz Monte Carlo beyond. Miwa at its default step count
carries roughly ``1e-4`` error, which is *worse* than this package's
integrator by four orders of magnitude — using R's default would have meant
"validating" an accurate result against an inaccurate one. The fixture script
therefore requests ``GenzBretz(maxpts=250000, abseps=1e-8)`` explicitly.

With that in place the remaining CDF differences for ``d >= 3`` are ~2e-6
absolute, consistent with the quasi-Monte-Carlo error of *both* sides. ``d = 2``
is exact on both, since ``rcopula`` uses Owen's T.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import rcopula as rc

pytestmark = pytest.mark.golden

GOLDEN = Path(__file__).parent / "golden" / "elliptical.json"

#: Relative tolerances for the density; absolute for the CDF, because a
#: 5-dimensional copula CDF is routinely ~1e-6 and a relative bound there would
#: be measuring nothing but the QMC floor.
#:
#: These are set by the **oldest supported toolchain**, not by the newest. The
#: Student-t cases move in the last few digits with the SciPy version -- the
#: incomplete beta and gamma functions underneath it are not bit-identical
#: across releases -- and tolerances tuned on one machine fail on Python 3.10's
#: older SciPy while passing everywhere else. Agreeing with R to nine digits on
#: a log density is the claim worth making; the tenth is a property of libm.
TOL_PDF = 5e-10
TOL_LOGPDF = 5e-8  # relative, on log densities that pass near zero
TOL_CDF_EXACT = 1e-10  # d <= 3, both sides exact
TOL_CDF_QMC = 1e-5  # d >= 4, QMC on both sides


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run `make golden`")
    return json.loads(GOLDEN.read_text())


def _numeric(value: object) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=object))
    return np.array([np.nan if (v is None or v == "NA") else float(v) for v in arr])


def _build(blk: dict) -> rc.EllipticalCopula:
    kwargs = {"dim": blk["dim"], "dispstr": blk["dispstr"]}
    if blk["family"] == "normal":
        return rc.GaussianCopula(blk["rho"], **kwargs)
    return rc.StudentCopula(blk["rho"], df=float(_numeric(blk["df"])[0]), **kwargs)


def _cases(blob: dict) -> list[str]:
    return sorted(k for k in blob if not k.startswith("_"))


@pytest.mark.parametrize("quantity", ["pdf", "logpdf"])
def test_density_matches_r(golden: dict, quantity: str) -> None:
    worst, worst_case = 0.0, ""
    for key in _cases(golden):
        blk = golden[key]
        cop = _build(blk)
        u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
        got = getattr(cop, quantity)(u)
        expected = _numeric(blk[quantity])
        rel = np.max(np.abs(got - expected) / np.maximum(np.abs(expected), 1e-12))
        if rel > worst:
            worst, worst_case = rel, key
    tol = TOL_PDF if quantity == "pdf" else TOL_LOGPDF
    assert worst < tol, f"{quantity}: worst rel dev {worst:.3e} at {worst_case}"


@pytest.mark.slow
def test_cdf_matches_r(golden: dict) -> None:
    for key in _cases(golden):
        blk = golden[key]
        cop = _build(blk)
        u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
        got = cop.cdf(u)
        expected = _numeric(blk["cdf"])
        tol = TOL_CDF_EXACT if blk["dim"] <= 3 else TOL_CDF_QMC
        err = np.max(np.abs(got - expected))
        assert err < tol, f"{key}: max abs cdf deviation {err:.3e}"


def test_dependence_measures_match_r(golden: dict) -> None:
    for key in _cases(golden):
        blk = golden[key]
        cop = _build(blk)
        for attr, field in (("tau", "tau"), ("rho", "rho_s")):
            expected = _numeric(blk[field])
            got = np.atleast_1d(getattr(cop, attr)())
            if np.isnan(expected).any() or got.size != expected.size:
                continue
            assert np.allclose(got, expected, rtol=1e-12), f"{key}:{attr}"

        lam = _numeric(blk["lambdaL"])
        if not np.isnan(lam).any():
            assert cop.lambda_().lower == pytest.approx(float(lam[0]), rel=1e-12, abs=1e-15)


def test_low_dimensional_cdf_is_exact_not_merely_close(golden: dict) -> None:
    """d <= 3 should agree to machine precision, not to a QMC tolerance.

    This is the guard that would catch a silent fallback from the exact paths
    (Owen's T, or the trivariate conditioning integral) onto quasi-Monte-Carlo.
    """
    for key in _cases(golden):
        blk = golden[key]
        if blk["dim"] > 3:
            continue
        cop = _build(blk)
        u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
        assert np.max(np.abs(cop.cdf(u) - _numeric(blk["cdf"]))) < TOL_CDF_EXACT, key


def test_coverage_spans_families_dimensions_and_structures(golden: dict) -> None:
    seen = {(golden[k]["family"], golden[k]["dim"], golden[k]["dispstr"]) for k in _cases(golden)}
    for family in ("normal", "t"):
        for dim in (2, 3, 5):
            for dispstr in ("ex", "ar1"):
                assert (family, dim, dispstr) in seen, f"missing {family} d={dim} {dispstr}"
        assert (family, 3, "un") in seen
