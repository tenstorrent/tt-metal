# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Same device reuse as tests/ttnn/unit_tests/operations/conv2d_3x3/conftest.py, whose tests these files re-export.
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
