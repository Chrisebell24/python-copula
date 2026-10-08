"""matplotlib is an optional dependency (the ``plots`` extra).

``import rcopula`` and ``import rcopula.plots`` must work without it, and drawing
a plot must fail with an ImportError that says how to install it. Each check runs
in a fresh interpreter with matplotlib blocked (``sys.modules['matplotlib'] = None``
makes any import of it raise ImportError), so it cannot leak into, or be masked
by, the matplotlib already imported by the rest of the suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MESSAGE = "rcopula.plots needs matplotlib: pip install 'rcopula[plots]'"


def _run_without_matplotlib(code: str) -> subprocess.CompletedProcess[str]:
    prelude = (
        "import sys\nfor _m in ('matplotlib', 'matplotlib.pyplot'):\n    sys.modules[_m] = None\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=120,
    )


def test_import_rcopula_and_plots_without_matplotlib() -> None:
    result = _run_without_matplotlib(
        """
        import rcopula
        import rcopula.plots
        assert 'matplotlib' not in {k for k, v in sys.modules.items() if v is not None}
        print('ok')
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_plot_without_matplotlib_raises_a_helpful_import_error() -> None:
    result = _run_without_matplotlib(
        f"""
        import rcopula as rc
        from rcopula.plots import contour, pairs_rosenblatt, scatter_matrix
        cop = rc.ClaytonCopula(2.0)
        u = cop.rvs(50, random_state=0)
        for call in (lambda: contour(cop), lambda: scatter_matrix(u),
                     lambda: pairs_rosenblatt(cop, u)):
            try:
                call()
            except ImportError as exc:
                assert str(exc) == {MESSAGE!r}, str(exc)
            else:
                raise AssertionError('no ImportError raised')
        print('ok')
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
