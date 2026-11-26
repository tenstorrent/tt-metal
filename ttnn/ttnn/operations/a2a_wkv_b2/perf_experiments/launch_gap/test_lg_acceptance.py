# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""launch_gap: the op's acceptance test (tests/ttnn/unit_tests/operations/a2a_wkv_b2/test_a2a_wkv_b2.py), run
unchanged against the lg_descriptor variant named by LG_ACC_VARIANT (default real_ktable_sem)."""
import os

import pytest

import tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 as T
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import *  # noqa: F401,F403 (collect its tests)
from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap.lg_op import lg_op
from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap.test_lg_timing import apply

VARIANT = os.environ.get("LG_ACC_VARIANT", "real_ktable_sem")


def _variant_op(*args, **kwargs):
    apply(VARIANT)
    return lg_op(*args, **kwargs)


T.a2a_wkv_b2 = _variant_op


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
