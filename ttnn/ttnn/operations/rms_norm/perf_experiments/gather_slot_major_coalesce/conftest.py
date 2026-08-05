# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open the device ONCE per module for this bench, so every launch of a sweep lands in one
profiler CSV in launch order (the root `device` fixture is function-scoped; the marker
switches it to module scope)."""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
