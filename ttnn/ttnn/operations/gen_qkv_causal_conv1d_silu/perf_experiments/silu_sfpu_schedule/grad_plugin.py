# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: run any gen_qkv_causal_conv1d_silu test suite with the PROPOSED compute kernel
(graduate/gen_qkv_causal_conv1d_silu_compute.cpp) swapped in for the op's own, without touching the op.

    scripts/run_safe_pytest.sh --run-all <suite> -p \
        ttnn.operations.gen_qkv_causal_conv1d_silu.perf_experiments.silu_sfpu_schedule.grad_plugin
"""
import importlib
from pathlib import Path

import pytest

GRAD_KERNEL = Path(__file__).parent / "graduate" / "gen_qkv_causal_conv1d_silu_compute.cpp"
COMPUTE = 2


@pytest.fixture(autouse=True)
def _grad_compute_kernel(monkeypatch):
    op_module = importlib.import_module("ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu")
    real = op_module.create_program_descriptor

    def create(*args, **kwargs):
        descriptor, work = real(*args, **kwargs)
        kernels = list(descriptor.kernels)
        assert kernels[COMPUTE].kernel_source.endswith("gen_qkv_causal_conv1d_silu_compute.cpp")
        kernels[COMPUTE].kernel_source = str(GRAD_KERNEL)
        descriptor.kernels = kernels
        return descriptor, work

    monkeypatch.setattr(op_module, "create_program_descriptor", create)
    op_module._PROGRAM_CACHE.clear()
    yield
    op_module._PROGRAM_CACHE.clear()
