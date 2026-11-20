# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for mhc_pre golden tests.

Registry-model golden tests carry no marker vocabulary — cells are
identified by parametrize ids, not pytest markers. Only `numerics` needs
registration (used by test_regression.py) so pytest doesn't warn about
unknown marks.

Do NOT define a local `device` fixture here. The root tt-metal/conftest.py
already provides one; combined with the `use_module_device` marker applied
below to every collected test, that's how we get a module-scoped device.
"""


import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "numerics: numerical-stability / data-distribution regression tests — "
        "not registry-driven, run unconditionally in the full suite",
    )


def pytest_collection_modifyitems(items):
    # Open the device once per module for this single-device op's tests.
    # Applies to every test in the dir (test_golden, test_regression,
    # test_translated) so no per-file `pytestmark` is needed — see PR #56.
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
