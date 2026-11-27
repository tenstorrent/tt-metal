"""launch_gap2: the op's acceptance test (tests/ttnn/unit_tests/operations/a2a_wkv_b2/test_a2a_wkv_b2.py), run
unchanged against the v_descriptor variant named by V_ACC_VARIANT (default merge_tx)."""
import os

import pytest

import tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 as T
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import *  # noqa: F401,F403 (collect its tests)
from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap2.v_op import v_op, apply

VARIANT = os.environ.get("V_ACC_VARIANT", "merge_tx")


def _variant_op(*args, **kwargs):
    apply(VARIANT)
    return v_op(*args, **kwargs)


T.a2a_wkv_b2 = _variant_op


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
