"""Run every example script and require it to succeed.

The examples are documentation, and documentation rots. Each one asserts its own
claims, so running it is a real test: if a refactor changes a number the example
depends on, the example fails rather than quietly printing something wrong.

They are marked ``slow`` because several run large simulations. ``pytest -m
"not slow"`` skips them; CI does not.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(p for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))


def test_the_gallery_is_not_empty() -> None:
    assert len(SCRIPTS) >= 14, f"only found {len(SCRIPTS)} examples in {EXAMPLES}"


def test_every_example_is_listed_in_the_readme() -> None:
    """An example nobody can find is an example nobody runs."""
    index = (EXAMPLES / "README.md").read_text()
    missing = [p.name for p in SCRIPTS if p.name not in index]
    assert not missing, f"not linked from examples/README.md: {missing}"


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_runs_and_its_assertions_hold(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=script.parent,
        env={"PYTHONPATH": str(EXAMPLES.parent), "MPLBACKEND": "Agg", "PATH": ""},
        timeout=1200,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{script.name} exited {result.returncode}\n"
            f"--- stdout (tail) ---\n{result.stdout[-2500:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-2500:]}"
        )
    # Every script reports what it verified; a run with no checks is a script
    # that asserts nothing.
    assert "[ok]" in result.stdout, f"{script.name} verified nothing"
