"""Parity tests for the Archimedean families against R's ``copula`` package.

Fixtures in ``tests/golden/archimedean.json`` are produced by
``tools/rgolden/01_families.R`` and committed, so this file needs no R.
Regenerate with ``make golden``.

Two things are worth knowing before reading the tolerances.

**Sampling is not compared here.** R's Mersenne-Twister stream cannot be
reproduced by NumPy, so ``rvs`` is validated by distributional properties in
``test_archimedean.py`` instead. Everything in this file is deterministic given
``u``.

**R is not always the more accurate side.** For ``rho`` on Clayton and Gumbel,
R's numerical approximation carries an error of order 1e-3, while the
Gauss-Legendre quadrature used here agrees with 30-digit ``mpmath`` to ~1e-11.
Those cases are therefore checked against ``mpmath`` at full precision and
against R only loosely — with a regression test that pins the discrepancy, so
that if R ever tightens its implementation we find out.
"""

from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest

import rcopula as rc

pytestmark = pytest.mark.golden

GOLDEN = Path(__file__).parent / "golden" / "archimedean.json"

FAMILIES = {
    "clayton": rc.ClaytonCopula,
    "gumbel": rc.GumbelCopula,
    "frank": rc.FrankCopula,
    "joe": rc.JoeCopula,
    "amh": rc.AMHCopula,
}

# Measured worst-case relative deviations from R, with headroom. Keeping these
# tight is the point: a regression shows up as a test failure, not a shrug.
TOL = {
    "pdf": 1e-12,
    "logpdf": 1e-11,
    "cdf": 1e-9,  # Frank at theta=20 reaches 1.9e-10
    "tau": 1e-12,
    "lambdaL": 1e-14,
    "lambdaU": 1e-14,
    "psi": 1e-13,
    "ipsi": 1e-13,
}


def _numeric(value: object) -> np.ndarray:
    """Coerce a fixture entry to floats, mapping R's ``NA`` to ``nan``.

    R does not implement every method for every family -- ``rho()`` has no
    ``joeCopula`` method at all -- so the generator script records ``NA`` there
    and comparisons against it are skipped.
    """
    arr = np.atleast_1d(np.asarray(value, dtype=object))
    return np.array([np.nan if (v is None or v == "NA") else float(v) for v in arr])


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover - only if fixtures are missing
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run `make golden`")
    return json.loads(GOLDEN.read_text())


def _cases(blob: dict) -> list[str]:
    return sorted(k for k in blob if not k.startswith("_"))


def test_fixtures_record_their_provenance(golden: dict) -> None:
    meta = golden["_meta"]
    assert meta["copula_version"].startswith("1.1")
    assert "R version" in meta["r_version"]


@pytest.mark.parametrize("quantity", ["pdf", "logpdf", "cdf"])
def test_density_and_cdf_match_r(golden: dict, quantity: str) -> None:
    worst, worst_case = 0.0, ""
    for key in _cases(golden):
        blk = golden[key]
        cop = FAMILIES[blk["family"]](blk["theta"], dim=blk["dim"])
        u = np.atleast_2d(np.asarray(blk["u"], dtype=float))
        got = getattr(cop, quantity)(u)
        expected = _numeric(blk[quantity])
        if np.isnan(expected).any():
            continue

        rel = np.max(np.abs(got - expected) / np.maximum(np.abs(expected), 1e-12))
        if rel > worst:
            worst, worst_case = rel, key

    assert worst < TOL[quantity], f"{quantity}: worst rel dev {worst:.3e} at {worst_case}"


@pytest.mark.parametrize("quantity", ["tau", "lambdaL", "lambdaU"])
def test_dependence_measures_match_r(golden: dict, quantity: str) -> None:
    for key in _cases(golden):
        blk = golden[key]
        cop = FAMILIES[blk["family"]](blk["theta"], dim=blk["dim"])
        if quantity == "tau":
            got = cop.tau()
        elif quantity == "lambdaL":
            got = cop.lambda_().lower
        else:
            got = cop.lambda_().upper
        expected = _numeric(blk[quantity])
        if np.isnan(expected).any():
            continue
        assert got == pytest.approx(float(expected[0]), rel=TOL[quantity], abs=1e-15), key


@pytest.mark.parametrize("quantity", ["psi", "ipsi"])
def test_generators_match_r(golden: dict, quantity: str) -> None:
    for key in _cases(golden):
        blk = golden[key]
        cop = FAMILIES[blk["family"]](blk["theta"], dim=blk["dim"])
        arg = np.asarray(blk[f"{quantity}_{'t' if quantity == 'psi' else 'u'}"], dtype=float)
        got = getattr(cop, quantity)(arg)
        expected = _numeric(blk[quantity])
        if np.isnan(expected).any():
            continue
        assert np.allclose(got, expected, rtol=TOL[quantity], atol=1e-15), key


def test_itau_matches_r(golden: dict) -> None:
    """Kendall's tau inversion.

    Frank's ``iTau`` is a root-find on both sides; R's own positive- and
    negative-tau answers disagree with each other in the 10th digit, so 1e-8 is
    as tight as a comparison against R can honestly be.
    """
    for name, value in golden["_inversions"].items():
        parts = name.split("_")
        family, kind = parts[0], parts[1]
        if kind != "itau":
            continue
        target = float(parts[-1]) * (-1.0 if "neg" in name else 1.0)
        got = FAMILIES[family].from_tau(target).theta
        assert got == pytest.approx(value, rel=1e-8), name


def test_irho_matches_r(golden: dict) -> None:
    for name, value in golden["_inversions"].items():
        if "irho" not in name:
            continue
        target = float(name.split("_")[-1])
        got = FAMILIES[name.split("_")[0]].from_rho(target).theta
        assert got == pytest.approx(value, rel=1e-7), name


class TestSpearmanRho:
    """Frank has a closed form; Clayton and Gumbel need quadrature."""

    def test_frank_matches_r(self, golden: dict) -> None:
        for key in _cases(golden):
            blk = golden[key]
            if blk["family"] != "frank":
                continue
            cop = rc.FrankCopula(blk["theta"], dim=blk["dim"])
            assert cop.rho() == pytest.approx(blk["rho"], rel=1e-11), key

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("family", "theta"),
        [
            ("clayton", 0.5),
            ("clayton", 2.0),
            ("clayton", 10.0),
            ("gumbel", 1.2),
            ("gumbel", 2.0),
            ("gumbel", 8.0),
        ],
    )
    def test_clayton_gumbel_match_mpmath_not_r(self, family: str, theta: float) -> None:
        """Our quadrature beats R's approximation by ~8 orders of magnitude.

        R is not the authority for this quantity, so the reference is a 30-digit
        ``mpmath`` evaluation of ``rho = 12 * int int C(u,v) du dv - 3``.
        """
        mp.mp.dps = 30
        if family == "clayton":

            def cdf(u, v):
                return (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta)
        else:

            def cdf(u, v):
                return mp.e ** (-(((-mp.log(u)) ** theta + (-mp.log(v)) ** theta) ** (1 / theta)))

        reference = float(12 * mp.quad(lambda u: mp.quad(lambda v: cdf(u, v), [0, 1]), [0, 1]) - 3)
        got = FAMILIES[family](theta).rho()
        assert got == pytest.approx(reference, rel=1e-9)

    @pytest.mark.slow
    def test_r_is_the_less_accurate_side(self, golden: dict) -> None:
        """Pin the known R discrepancy so a future R fix is noticed, not missed."""
        mp.mp.dps = 30
        theta = 2.0
        reference = float(
            12
            * mp.quad(
                lambda u: mp.quad(lambda v: (u**-theta + v**-theta - 1) ** (-1 / theta), [0, 1]),
                [0, 1],
            )
            - 3
        )
        r_value = golden["clayton_d2_theta2"]["rho"]
        ours = rc.ClaytonCopula(theta).rho()

        assert abs(ours - reference) < 1e-11
        assert abs(r_value - reference) > 1e-5, (
            "R's Spearman rho for Clayton now looks accurate; if R has been "
            "fixed, tighten this test and compare against R directly."
        )


def test_every_family_and_dimension_is_covered(golden: dict) -> None:
    """Guard against silently shrinking the fixture set."""
    seen = {(golden[k]["family"], golden[k]["dim"]) for k in _cases(golden)}
    for family in ("clayton", "gumbel", "frank", "joe"):
        for dim in (2, 3, 5):
            assert (family, dim) in seen, f"missing coverage: {family} d={dim}"
    # R restricts amhCopula to d = 2, so that is all we can cross-check.
    assert ("amh", 2) in seen
