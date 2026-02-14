# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration for onboarding tests.
"""


def pytest_configure(config):
    """Filter out harmless nanobind type registration warnings."""
    config.addinivalue_line("filterwarnings", "ignore:nanobind.*was already registered:RuntimeWarning")
