"""The documentation must build, and must not silently lose the API.

`mkdocs build --strict` fails on broken links and unresolved references, so
running it is a real check that the navigation and every `:::` reference still
point at something. The content assertions guard against the quieter failure:
mkdocstrings resolving a module but rendering nothing, which produces a valid
site with an empty API reference.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "mkdocs.yml"

pytestmark = pytest.mark.skipif(
    shutil.which("mkdocs") is None and not (Path(sys.executable).parent / "mkdocs").exists(),
    reason="mkdocs is not installed; `pip install -e '.[docs]'`",
)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("site")
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "mkdocs"),
            "build",
            "--strict",
            "-f",
            str(CONFIG),
            "-d",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"mkdocs build --strict failed\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
        )
    return out


def test_every_navigation_page_is_produced(built: Path) -> None:
    expected = [
        "index.html",
        "concepts/index.html",
        "choosing/index.html",
        "vines/index.html",
        "r-translation/index.html",
        "parity/index.html",
        "examples/index.html",
        "api/families/index.html",
        "api/structural/index.html",
        "api/inference/index.html",
        "api/gof/index.html",
        "api/transforms/index.html",
        "api/plots/index.html",
        "api/applications/index.html",
        "api/special/index.html",
    ]
    missing = [name for name in expected if not (built / name).exists()]
    assert not missing, f"pages not built: {missing}"


@pytest.mark.parametrize(
    ("page", "symbols"),
    [
        (
            "api/families",
            [
                "ClaytonCopula",
                "StudentCopula",
                "GalambosCopula",
                "MarshallOlkinCopula",
                "CopulaDistribution",
                "to_emp_margins",
                "fit_lambda",
            ],
        ),
        (
            "api/structural",
            [
                "RotatedCopula",
                "KhoudrajiCopula",
                "MixtureCopula",
                "NestedArchimedean",
                "fit_nested",
                "marginal_copula",
                "opower",
            ],
        ),
        (
            "api/inference",
            [
                "fit",
                "CopulaFitResult",
                "select_copula",
                "cross_validate",
                "bootstrap_measure",
                "fit_joint",
                "to_json",
            ],
        ),
        ("api/gof", ["gof_test", "exch_test", "rad_sym_test", "dependogram", "serial_indep_test"]),
        (
            "api/transforms",
            [
                "rosenblatt",
                "inverse_rosenblatt",
                "kendall_cdf",
                "kendall_return_period",
                "htrafo",
                "radial_simplex",
                "quasi_rvs",
                "variance_ratio",
            ],
        ),
        (
            "api/plots",
            [
                "tail_concentration",
                "kendall_plot",
                "pickands_plot",
                "dependogram_plot",
                "pairs_rosenblatt",
            ],
        ),
        (
            "api/applications",
            [
                "value_at_risk",
                "tranche_spread",
                "cms_spread_option",
                "backtest_pairs",
                "select_pairs",
                "select_partners",
                "CopulaGarch",
                "DynamicCopula",
                "fit_dcc",
                "discrete_pmf",
                "tau_upper_bound",
                "operational_risk_capital",
            ],
        ),
        ("api/special", ["retstable", "mvt_cdf", "debye1", "log1mexp"]),
    ],
)
def test_the_api_reference_actually_documents_things(
    built: Path, page: str, symbols: list[str]
) -> None:
    """mkdocstrings can resolve a module and render nothing; that would still
    build cleanly and leave an empty reference behind."""
    html = (built / page / "index.html").read_text()
    missing = [s for s in symbols if s not in html]
    assert not missing, f"{page} does not document: {missing}"
    assert html.count("doc-heading") > 5, f"{page} rendered almost nothing"


def test_the_home_page_states_the_licence_position(built: Path) -> None:
    """The GPL relationship is the kind of thing that must not quietly vanish."""
    html = (built / "index.html").read_text()
    assert "MIT" in html
    assert "behavioural test oracle" in html or "behavioural" in html
