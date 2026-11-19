# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: run a test suite against this experiment's grad/ copy of the op instead of the real op, without
touching the real op's files (in-memory only: the real package's exports and the op submodules in sys.modules are
redirected to grad/ before collection).

    scripts/run_safe_pytest.sh eval/golden_tests/conv2d_3x3/ \\
        -p ttnn.operations.conv2d_3x3.perf_experiments.kwf_n_planes.golden_redirect
"""
import importlib
import os
import sys

_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.kwf_n_planes." + os.environ.get("KWP_REDIRECT", "grad")


def pytest_configure(config):
    real = importlib.import_module("ttnn.operations.conv2d_3x3")
    op = importlib.import_module(_PKG + ".conv2d_3x3")
    pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
    for name in real.__all__:
        setattr(real, name, getattr(op, name))
    sys.modules["ttnn.operations.conv2d_3x3.conv2d_3x3"] = op
    sys.modules["ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor"] = pd
    assert real.conv2d_3x3 is op.conv2d_3x3
    print(f"GOLDEN_REDIRECT: ttnn.operations.conv2d_3x3 -> {_PKG} (kernels {pd.KERNEL_DIR})")
