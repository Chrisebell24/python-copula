"""Shared helpers for the example scripts.

Kept deliberately small: an example should read as the thing it demonstrates,
not as a framework.
"""

from __future__ import annotations

import numpy as np


def heading(text: str) -> None:
    print(f"\n{text}\n{'=' * len(text)}")


def show(label: str, value: object, width: int = 46) -> None:
    if isinstance(value, float | np.floating):
        print(f"  {label:<{width}} {value:>12.6f}")
    else:
        print(f"  {label:<{width}} {value}")


def check(description: str, condition: bool) -> None:
    """Assert, and say what was checked -- so the output is a record of what
    the script actually verified rather than a claim that it did."""
    if not condition:
        raise AssertionError(description)
    print(f"  [ok] {description}")
