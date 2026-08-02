"""Shared pytest configuration.

Network-dependent tests are opt-in. A suite that needs the internet fails for
reasons unrelated to the code, so ``network``-marked tests are skipped unless
``--run-network`` is passed -- at which point they check the loader against the
real upstream sources.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that download from the real data sources",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs the network; pass --run-network to include")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
