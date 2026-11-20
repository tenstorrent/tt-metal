# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The op's own correctness tests on this experiment's op copy (BRF_BANK_RUN=1 by default): ttnn.operations.conv2d_3x3
is redirected to the copy before each test module is imported by path.

  scripts/run_safe_pytest.sh --run-all <this file> --import-mode=prepend
"""

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import brf_load  # noqa: E402

opc, _pd, _op = brf_load.load()
import ttnn.operations.conv2d_3x3  # noqa: E402,F401  (the real package, imported by ttnn anyway)

sys.modules["ttnn.operations.conv2d_3x3"] = opc

REPO = Path(__file__).resolve().parents[6]
MODULES = {
    "unit": "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3.py",
    "layouts": "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_layouts.py",
    "asym": "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_asymmetric_padding.py",
    "regression": "eval/golden_tests/conv2d_3x3/test_regression.py",
}
for tag, rel in MODULES.items():
    spec = importlib.util.spec_from_file_location(f"brf_unit_{tag}", REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("test_"):
            globals()[f"{name}__{tag}"] = getattr(mod, name)
        elif name.startswith(("fixture", "_fixture")) or (
            callable(getattr(mod, name)) and hasattr(getattr(mod, name), "_pytestfixturefunction")
        ):
            globals()[name] = getattr(mod, name)


def pytest_collection_modifyitems(items):  # noqa: D103 (module devices, as the op's conftests)
    for item in items:
        item.add_marker(pytest.mark.use_module_device)
