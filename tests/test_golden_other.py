"""Parity tests for Plackett, FGM, Marshall-Olkin and the Frechet bounds.

Fixtures come from ``tools/rgolden/02_other.R``.

**Kendall's tau for Plackett is checked against quadrature, not against R.**
R's value is wrong by up to 1.3e-3: at ``theta = 1`` the copula *is* the
independence copula and tau must be exactly zero, yet R returns -2.57e-4. This
package integrates ``tau = 4 * int int C c du dv - 1`` on a Gauss-Legendre grid
and returns 1.6e-15 there, agreeing with adaptive quadrature to 12 digits and
being stable across 100 to 1600 nodes.

Three quantities exist here that R does not provide at all, so they have no
fixture to compare against: ``pdf``/``logpdf`` for Marshall-Olkin (R's
``moCopula`` implements no density) and ``lambda`` for FGM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

import rcopula as rc

pytestmark = pytest.mark.golden

GOLDEN = Path(__file__).parent / "golden" / "other.json"

BUILDERS = {
    "plackett": lambda b: rc.PlackettCopula(b["theta"]),
    "fgm": lambda b: rc.FGMCopula(b["theta"]),
    "mo": lambda b: rc.MarshallOlkinCopula(np.asarray(b["alpha"], dtype=float)),
    "indep": lambda b: rc.IndependenceCopula(2),
    "fh_upper": lambda b: rc.FrechetUpperCopula(2),
    "fh_lower": lambda b: rc.FrechetLowerCopula(),
}


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run `make golden`")
    return json.loads(GOLDEN.read_text())


def _numeric(value: object) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=object))
    return np.array([np.nan if (v is None or v == "NA") else float(v) for v in arr])


def _cases(blob: dict) -> list[str]:
    return sorted(k for k in blob if not k.startswith("_"))


@pytest.mark.parametrize("quantity", ["pdf", "logpdf", "cdf"])
def test_density_and_cdf_match_r(golden: dict, quantity: str) -> None:
    compared = 0
    for key in _cases(golden):
        blk = golden[key]
        expected = _numeric(blk[quantity])
        if np.isnan(expected).any():
            continue  # R provides no method here
        cop = BUILDERS[blk["family"]](blk)
        u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
        got = getattr(cop, quantity)(u)
        assert np.allclose(got, expected, rtol=1e-11, atol=1e-13), key
        compared += 1
    assert compared > 0, f"no fixture exercised {quantity}"


@pytest.mark.parametrize("quantity", ["rho", "lambdaL", "lambdaU"])
def test_dependence_measures_match_r(golden: dict, quantity: str) -> None:
    for key in _cases(golden):
        blk = golden[key]
        expected = _numeric(blk[quantity])
        if np.isnan(expected).any():
            continue
        cop = BUILDERS[blk["family"]](blk)
        got = {
            "rho": cop.rho,
            "lambdaL": lambda c=cop: c.lambda_().lower,
            "lambdaU": lambda c=cop: c.lambda_().upper,
        }[quantity]()
        assert got == pytest.approx(float(expected[0]), rel=1e-11, abs=1e-13), key


def test_kendall_tau_matches_r_where_r_is_reliable(golden: dict) -> None:
    """FGM and Marshall-Olkin have closed forms in both implementations."""
    for key in _cases(golden):
        blk = golden[key]
        if blk["family"] not in ("fgm", "mo", "indep", "fh_upper", "fh_lower"):
            continue
        expected = _numeric(blk["tau"])
        if np.isnan(expected).any():
            continue
        cop = BUILDERS[blk["family"]](blk)
        assert cop.tau() == pytest.approx(float(expected[0]), rel=1e-12, abs=1e-14), key


class TestPlackettKendallTau:
    """R is the unreliable side here, so quadrature is the reference."""

    @pytest.mark.parametrize("theta", [0.1, 0.5, 2.0, 3.0, 10.0, 50.0])
    def test_matches_adaptive_quadrature(self, theta: float) -> None:
        cop = rc.PlackettCopula(theta)

        def integrand(v: float, u: float) -> float:
            return float(cop.cdf([[u, v]])[0] * cop.pdf([[u, v]])[0])

        value, _ = integrate.dblquad(integrand, 0, 1, 0, 1, epsabs=1e-12, epsrel=1e-12)
        assert cop.tau() == pytest.approx(4.0 * value - 1.0, abs=1e-9)

    def test_independence_gives_exactly_zero(self) -> None:
        """theta = 1 *is* the independence copula. R returns -2.57e-4 here."""
        assert rc.PlackettCopula(1.0).tau() == pytest.approx(0.0, abs=1e-12)

    def test_is_antisymmetric_in_log_theta(self) -> None:
        """tau(1/theta) = -tau(theta), exactly."""
        for theta in (0.2, 0.5, 3.0, 10.0):
            assert rc.PlackettCopula(1.0 / theta).tau() == pytest.approx(
                -rc.PlackettCopula(theta).tau(), abs=1e-10
            )

    def test_r_is_the_less_accurate_side(self, golden: dict) -> None:
        """Pin the discrepancy so a future R fix is noticed rather than missed."""
        r_value = float(_numeric(golden["plackett_1"]["tau"])[0])
        assert abs(r_value) > 1e-5, (
            "R's Plackett tau at theta=1 now looks correct; if R has been fixed, "
            "compare against R directly instead of quadrature."
        )
        assert abs(rc.PlackettCopula(1.0).tau()) < 1e-12


def test_families_r_does_not_implement_are_still_provided(golden: dict) -> None:
    """Guard the gaps we fill, so they are not silently lost.

    R has no ``dCopula`` method for ``moCopula`` and no ``lambda`` for
    ``fgmCopula``; both work here.
    """
    assert np.isnan(_numeric(golden["mo_0.2_0.8"]["pdf"])).any()
    assert np.all(np.isfinite(rc.MarshallOlkinCopula(0.2, 0.8).pdf([[0.3, 0.6]])))

    assert np.isnan(_numeric(golden["fgm_0.3"]["lambdaU"])).any()
    assert rc.FGMCopula(0.3).lambda_() == rc.TailDependence(0.0, 0.0)


def test_singular_copulas_report_having_no_density(golden: dict) -> None:
    """R returns an error for these; we raise an informative one."""
    for cls in (rc.FrechetUpperCopula, rc.FrechetLowerCopula):
        with pytest.raises(NotImplementedError, match="singular"):
            cls().pdf([[0.3, 0.4]])


def test_coverage_spans_every_family(golden: dict) -> None:
    seen = {golden[k]["family"] for k in _cases(golden)}
    assert seen == set(BUILDERS)
