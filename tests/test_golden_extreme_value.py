"""Parity tests for the extreme-value families against R's ``copula`` package.

Fixtures come from ``tools/rgolden/03_extreme_value.R``.

**Densities are compared in absolute, not relative, terms.** These copulas
concentrate sharply under strong dependence: at Galambos ``theta = 5`` the R
fixture contains a density of ``4.8e-16``, and at Husler-Reiss ``theta = 3`` one
of ``1.3e-34``. A relative bound on values like those measures floating-point
noise, not agreement. The absolute agreement is ~5e-14 throughout.

**R's t-EV Kendall tau is wrong for negative rho** and is excluded from the
comparison. At ``rho = -0.3, df = 4`` R reports 0.1225 where the true value is
about 0.0211. Two independent routes confirm ours: the Pickands integral, and a
Monte-Carlo Kendall tau computed from samples drawn by conditional inversion of
the *CDF* (which matches R to 1e-16 and never touches ``A''``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import rcopula as rc
from rcopula.core.extreme_value import gumbel_pickands

pytestmark = pytest.mark.golden

GOLDEN = Path(__file__).parent / "golden" / "extreme_value.json"

BUILDERS = {
    "galambos": lambda b: rc.GalambosCopula(b["theta"]),
    "huslerreiss": lambda b: rc.HuslerReissCopula(b["theta"]),
    "tawn": lambda b: rc.TawnCopula(b["theta"]),
    "tev": lambda b: rc.TEVCopula(b["rho_par"], df=b["df"]),
    "gumbel_ev": lambda b: rc.GumbelCopula(b["theta"]),
}


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run `make golden`")
    return json.loads(GOLDEN.read_text())


def _cases(blob: dict) -> list[str]:
    return sorted(k for k in blob if not k.startswith("_"))


def test_pickands_function_matches_r(golden: dict) -> None:
    for key in _cases(golden):
        blk = golden[key]
        w = np.asarray(blk["w"], dtype=float)
        expected = np.asarray(blk["A"], dtype=float)
        got = (
            gumbel_pickands(w, blk["theta"])
            if blk["family"] == "gumbel_ev"
            else BUILDERS[blk["family"]](blk).A(w)
        )
        assert np.allclose(got, expected, rtol=0, atol=1e-14), key


@pytest.mark.parametrize("quantity", ["pdf", "cdf"])
def test_density_and_cdf_match_r(golden: dict, quantity: str) -> None:
    for key in _cases(golden):
        blk = golden[key]
        cop = BUILDERS[blk["family"]](blk)
        u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
        expected = np.asarray(blk[quantity], dtype=float)
        got = getattr(cop, quantity)(u)
        assert np.max(np.abs(got - expected)) < 1e-12, key


def test_tail_dependence_matches_r(golden: dict) -> None:
    for key in _cases(golden):
        blk = golden[key]
        cop = BUILDERS[blk["family"]](blk)
        assert cop.lambda_().upper == pytest.approx(float(blk["lambdaU"]), abs=1e-13), key
        assert cop.lambda_().lower == pytest.approx(float(blk["lambdaL"]), abs=1e-13), key


def test_kendall_tau_matches_r(golden: dict) -> None:
    """Excludes t-EV, where R is unreliable -- see the module docstring."""
    for key in _cases(golden):
        blk = golden[key]
        if blk["family"] == "tev":
            continue
        cop = BUILDERS[blk["family"]](blk)
        assert cop.tau() == pytest.approx(float(blk["tau"]), abs=1e-7), key


def test_spearman_rho_matches_r(golden: dict) -> None:
    for key in _cases(golden):
        blk = golden[key]
        if blk["family"] in ("tev", "gumbel_ev"):
            continue  # both sides quadrature; gumbel_ev uses the Archimedean path
        cop = BUILDERS[blk["family"]](blk)
        assert cop.rho() == pytest.approx(float(blk["rho"]), abs=1e-5), key


class TestTEVKendallTau:
    """R disagrees for negative rho; confirm ours by an independent route."""

    @pytest.mark.slow
    @pytest.mark.parametrize("rho", [-0.3, 0.5, 0.8])
    def test_matches_monte_carlo_from_the_cdf(self, rho: float) -> None:
        """Sampling is conditional inversion of the CDF, which matches R to
        1e-16 and never uses the Pickands second derivative. So this is a
        genuinely independent check on ``tau``."""
        cop = rc.TEVCopula(rho, df=4)
        u = cop.rvs(200_000, random_state=0)
        empirical = stats.kendalltau(u[:, 0], u[:, 1]).statistic
        assert cop.tau() == pytest.approx(empirical, abs=0.005)

    def test_r_is_wrong_at_negative_rho(self, golden: dict) -> None:
        """Pin the discrepancy so a future R fix is noticed rather than missed."""
        r_value = float(golden["tev_-0.3"]["tau"])
        assert abs(r_value - 0.1225) < 1e-3, (
            "R's t-EV tau at rho=-0.3 has changed; if R has been fixed, "
            "include t-EV in test_kendall_tau_matches_r."
        )
        assert rc.TEVCopula(-0.3, df=4).tau() == pytest.approx(0.0211, abs=1e-3)


def test_analytic_derivatives_beat_finite_differences(golden: dict) -> None:
    """The t-EV density was wrong by ~2e-4 while ``A'``/``A''`` were
    finite-differenced. Guard the analytic implementations."""
    blk = golden["tev_0.5"]
    cop = rc.TEVCopula(blk["rho_par"], df=blk["df"])
    u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
    assert np.max(np.abs(cop.pdf(u) - np.asarray(blk["pdf"], dtype=float))) < 1e-12


def test_coverage_spans_every_family(golden: dict) -> None:
    assert {golden[k]["family"] for k in _cases(golden)} == set(BUILDERS)
