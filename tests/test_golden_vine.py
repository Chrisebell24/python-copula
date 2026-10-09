"""Parity tests for regular vines, BB1/BB7 and rotated pair-copulas against R.

Fixtures come from ``tools/rgolden/09_vine.R``, which calls R's ``VineCopula``
package. Three things are compared:

* every bivariate quantity of BB1 (family 7), BB7 (9) and the 90/180/270-degree
  rotations of Clayton, Gumbel, Joe, BB1 and BB7: density, CDF, both
  h-functions, Kendall's tau and the tail-dependence coefficients;
* the density, the per-row log-likelihood and the Rosenblatt transform
  (``RVinePIT``) of a five-dimensional R-vine with mixed and rotated families,
  using the example matrix of ``VineCopula``'s ``RVineMatrix`` help page;
* Dissmann's structure selection on 600 draws from that vine: the first tree
  (a maximum spanning tree on absolute Kendall's tau, which has no tuning
  choices) must be the same edges R chose.

``VineCopula``'s family codes put the rotation in the tens digit: 1x is the
survival (180-degree) copula, 2x reflects the *first* argument and 3x the
*second*, with negated parameters for 2x and 3x. Those are built here with an
explicit reflection mask, so the comparison does not depend on how either
library names its degrees.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import rcopula as rc
from rcopula.core.base import Copula
from rcopula.vine import VineCopula, fit_vine

pytestmark = pytest.mark.golden

GOLDEN = Path(__file__).parent / "golden" / "vine.json"

#: R ``VineCopula`` rotation digit -> which arguments are reflected.
FLIPS = {1: [True, True], 2: [True, False], 3: [False, True]}

#: rcopula family name for each R family code in the selection fixture.
NAMES = {
    1: "gaussian",
    2: "student",
    3: "clayton",
    4: "gumbel",
    5: "frank",
    6: "joe",
    13: "clayton180",
    14: "gumbel180",
    16: "joe180",
    # rcopula's 270 reflects the first argument, which is R's 2x ...
    23: "clayton270",
    24: "gumbel270",
    26: "joe270",
    # ... and its 90 the second, R's 3x.
    33: "clayton90",
    34: "gumbel90",
    36: "joe90",
}


def from_r(family: int, par: float, par2: float) -> Copula:
    """The rcopula pair-copula for one ``VineCopula`` family code and parameters."""
    family = int(family)
    if family == 0:
        return rc.IndependenceCopula(2)
    if family == 1:
        return rc.GaussianCopula(par)
    if family == 2:
        return rc.StudentCopula(par, df=par2)
    if family == 5:
        return rc.FrankCopula(par)
    base_code, rotation = family % 10, family // 10
    theta, delta = abs(par), abs(par2)
    base = {
        3: lambda: rc.ClaytonCopula(theta),
        4: lambda: rc.GumbelCopula(theta),
        6: lambda: rc.JoeCopula(theta),
        7: lambda: rc.BB1Copula(theta, delta),
        9: lambda: rc.BB7Copula(theta, delta),
    }[base_code]()
    return rc.RotatedCopula(base, FLIPS[rotation]) if rotation else base


def vine_from_r(blob: dict) -> VineCopula:
    """An rcopula R-vine from an R ``RVineMatrix`` (1-based labels, row-major)."""
    m = np.asarray(blob["matrix"], dtype=np.int64)
    fam = np.asarray(blob["family"])
    par = np.asarray(blob["par"], dtype=float)
    par2 = np.asarray(blob["par2"], dtype=float)
    d = m.shape[0]
    trees = [
        [from_r(fam[d - 1 - t, i], par[d - 1 - t, i], par2[d - 1 - t, i]) for i in range(d - 1 - t)]
        for t in range(d - 1)
    ]
    return VineCopula(trees, structure="R", matrix=np.tril(m - 1))


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover
        pytest.skip(f"golden fixtures not found at {GOLDEN}; run tools/rgolden/09_vine.R")
    return json.loads(GOLDEN.read_text())


def _pair_keys() -> list[str]:
    return sorted(json.loads(GOLDEN.read_text())["pairs"]) if GOLDEN.exists() else []


@pytest.mark.parametrize("key", _pair_keys())
class TestPairCopulas:
    def test_density_cdf_and_h_functions(self, golden: dict, key: str) -> None:
        blk = golden["pairs"][key]
        cop = from_r(blk["family"], blk["par"], blk["par2"])
        u = np.asarray(blk["u"], dtype=float)
        np.testing.assert_allclose(cop.pdf(u), blk["pdf"], rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(cop.cdf(u), blk["cdf"], rtol=1e-8, atol=1e-12)
        # hfunc1 conditions on the first argument, hfunc2 on the second.
        from rcopula.vine import _pair_h

        np.testing.assert_allclose(_pair_h(cop, u, 0), blk["hfunc1"], rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(_pair_h(cop, u, 1), blk["hfunc2"], rtol=1e-7, atol=1e-10)

    def test_tau_and_tail_dependence(self, golden: dict, key: str) -> None:
        blk = golden["pairs"][key]
        cop = from_r(blk["family"], blk["par"], blk["par2"])
        # R integrates BB7's tau numerically (to about 1e-9); BB1's is closed form.
        assert cop.tau() == pytest.approx(blk["tau"], abs=1e-7)
        lam = cop.lambda_()
        assert [lam.lower, lam.upper] == pytest.approx(blk["lambda"], abs=1e-10)


class TestRegularVineDensity:
    def test_density_matches_rvinepdf(self, golden: dict) -> None:
        blob = golden["rvine_density"]
        vine = vine_from_r(blob)
        u = np.asarray(blob["u"], dtype=float)
        np.testing.assert_allclose(vine.pdf(u), blob["pdf"], rtol=1e-8)
        np.testing.assert_allclose(vine.logpdf(u), blob["loglik"], rtol=1e-8, atol=1e-10)

    def test_rosenblatt_matches_rvinepit(self, golden: dict) -> None:
        blob = golden["rvine_density"]
        vine = vine_from_r(blob)
        u = np.asarray(blob["u"], dtype=float)
        np.testing.assert_allclose(vine.rosenblatt(u), blob["pit"], rtol=1e-7, atol=1e-10)

    def test_the_c_and_d_paths_agree_on_the_r_vine_matrix(self, golden: dict) -> None:
        """The fixture vine is neither C nor D: it exercises the general recursion."""
        vine = vine_from_r(golden["rvine_density"])
        first_tree = {frozenset((a, b)) for t, a, b, _, _ in vine.edges if t == 0}
        degrees = np.bincount([v for edge in first_tree for v in edge], minlength=5)
        assert degrees.max() < 4  # not a star (C-vine)
        assert degrees.max() > 2  # not a path (D-vine)


class TestDissmannSelection:
    def test_first_tree_matches_r(self, golden: dict) -> None:
        sel = golden["selection"]
        u = np.asarray(sel["u"], dtype=float)
        families = [NAMES[code] for code in sel["familyset"]]
        fitted = fit_vine(u, structure="R", families=families, truncate=1)
        ours = sorted(tuple(sorted((a + 1, b + 1))) for t, a, b, _, _ in fitted.edges if t == 0)
        theirs = sorted(tuple(int(v) for v in edge) for edge in sel["tree1"])
        assert ours == theirs

    def test_full_selection_reaches_rs_likelihood(self, golden: dict) -> None:
        """Same trees, same families where AIC is decisive, so the fitted
        log-likelihoods agree to within estimation noise (R uses its own
        optimiser and pseudo-observations, so exact equality is not expected)."""
        sel = golden["selection"]
        u = np.asarray(sel["u"], dtype=float)
        families = [NAMES[code] for code in sel["familyset"]]
        fitted = fit_vine(u, structure="R", families=families)
        assert fitted.loglik(u) == pytest.approx(sel["loglik"], rel=0.01)
        # R's selected vine, rebuilt here, has R's log-likelihood exactly.
        assert vine_from_r(sel).loglik(u) == pytest.approx(sel["loglik"], rel=1e-8)
