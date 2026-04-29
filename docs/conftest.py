# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.

"""Pytest configuration for tests collected under docs/.

Registers the `slow` marker (reused from neuralset-repo) and skips slow
tests unless `--slow` is passed.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run end-to-end (slow) tests in docs/test_code_builder.py",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: end-to-end pipeline test (opt-in via `--slow`)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="slow test (opt-in via --slow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
