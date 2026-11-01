# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""compute_retest whole-op A/B: LN_VARIANT=<base|stats|affine|combo> scripts/run_safe_pytest.sh --profile <this file>.

Points the op's KERNEL_DIR at kernels/<variant> (reader/writer/stream symlinked to the real op, compute =
the variant). One variant per pytest process. Correctness vs torch at the golden tolerance on every row.
"""

import os
from pathlib import Path

import pytest

from eval.golden_tests.layer_norm_tile.helpers import run_layer_norm
from tests.ttnn.unit_tests.operations.layer_norm_tile.test_perf_bench import GUARD_CASES, IDS, PERF_CASES
from ttnn.operations.layer_norm_tile import layer_norm_tile_program_descriptor as _pd

VARIANT = os.environ.get("LN_VARIANT", "base")
_pd.KERNEL_DIR = Path(__file__).parent / "kernels" / VARIANT

ALL = dict(zip(IDS, PERF_CASES))
PEEL = {name: case for name, case in ALL.items() if name.startswith(("focus", "8x384", "32x32"))}
REST = {name: case for name, case in ALL.items() if name not in PEEL}


def _run(device, case):
    case = dict(case)
    inputs = case.pop("inputs")
    run_layer_norm(inputs, device=device, **case)


@pytest.mark.parametrize("name", list(PEEL))
def test_peel(device, name):
    _run(device, PEEL[name])


@pytest.mark.parametrize("name", list(REST) + list(GUARD_CASES))
def test_guard(device, name):
    _run(device, {**REST, **GUARD_CASES}[name])


@pytest.mark.parametrize("offset", [0.0, 100.0], ids=["zero_mean", "mean100"])
@pytest.mark.parametrize("fp32_dest", [True, False], ids=["fp32dest", "bf16dest"])
def test_precision(device, offset, fp32_dest):
    """Focus shape metrics vs float64 torch (output rel-RMS / PCC)."""
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
    actual = ttnn.to_torch(out).float()
    m = compute_metrics_torch(actual, expected.float(), readback_dtype=torch.bfloat16)
    print(f"\nRETEST_PREC variant={VARIANT} fp32_dest={fp32_dest} offset={offset} pcc={m.pcc:.6f} rel_rms={m.rms:.6f}")
    assert m.pcc > 0.99
