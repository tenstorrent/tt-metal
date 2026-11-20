# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# The fill bench reuses one device per module; the e2e test keeps the function-scoped device (its device_params).
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.fspath.basename == "test_fill_bench.py":
            item.add_marker(pytest.mark.use_module_device)
