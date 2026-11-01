# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-op A/B: LN_VARIANT=<base|v2|..> scripts/run_safe_pytest.sh --run-all --profile <this file>::<test>.

Points the op's KERNEL_DIR at op_kernels/<variant> (copies of the op kernels at HEAD; only the compute differs).
One variant per pytest process. Correctness vs torch at the golden tolerance on every row (run_layer_norm).
"""

import importlib
import os
from pathlib import Path

import pytest

from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_perf_bench import GUARD_CASES, IDS, PERF_CASES

_pd = importlib.import_module("ttnn.operations.layer_norm_tile.layer_norm_tile_program_descriptor")
VARIANT = os.environ.get("LN_VARIANT", "base")
_pd.KERNEL_DIR = Path(__file__).parent / "op_kernels" / VARIANT

ALL = {**dict(zip(IDS, PERF_CASES)), **GUARD_CASES}


def _run(device, case):
    case = dict(case)
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)


@pytest.mark.parametrize("name", list(ALL))
def test_row(device, name):
    _run(device, ALL[name])


@pytest.mark.parametrize("offset", [0.0, 100.0], ids=["zero_mean", "mean100"])
@pytest.mark.parametrize("fp32_dest", [True, False], ids=["fp32dest", "bf16dest"])
def test_precision(device, offset, fp32_dest):
    """Focus-shape output PCC / rel-RMS vs float64 torch."""
    import torch
    import ttnn
    from eval.metrics import compute_metrics_torch
    from ttnn.operations.layer_norm_tile import layer_norm

    torch.manual_seed(0)
    shape = (8, 1, 512, 768)
    x = (torch.randn(shape) * 2.0 + offset).to(torch.bfloat16)
    res = torch.randn(shape).to(torch.bfloat16)
    gamma = torch.randn(768).to(torch.bfloat16)
    beta = torch.randn(768).to(torch.bfloat16)
    z = x.double() + res.double()
    expected = torch.nn.functional.layer_norm(z, (768,), gamma.double(), beta.double(), eps=1e-5)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = False
    l1 = ttnn.L1_MEMORY_CONFIG
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1)
    tr = ttnn.from_torch(res, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    rm = dict(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=l1)
    tg = ttnn.from_torch(gamma.reshape(1, 1, 24, 32), **rm)
    tb = ttnn.from_torch(beta.reshape(1, 1, 24, 32), **rm)
    out = layer_norm(tx, tg, tb, tr, epsilon=1e-5, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    m = compute_metrics_torch(ttnn.to_torch(out).float(), expected.float(), readback_dtype=torch.bfloat16)
    print(f"\nWHOLE_PREC variant={VARIANT} fp32_dest={fp32_dest} offset={offset} pcc={m.pcc:.6f} rel_rms={m.rms:.6f}")
    assert m.pcc > 0.99
