# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op A/B bench for the x_round0_fill experiment (run with scripts/run_safe_pytest.sh --profile).

Dispatches this dir's copy of the wq_b_linear program (pd.py + kernels/) 1 warmup + 5 timed times.
XRF_KNOBS="NAME=value,..." overrides pd.py knobs. Prints PCC vs float matmul of the device-quantized operands.
Analyze with: python3 analyze.py <profile_log_device.csv>
"""
import ast
import importlib.util
import os
from pathlib import Path

import ttnn

_spec = importlib.util.spec_from_file_location("xrf_pd", Path(__file__).parent / "pd.py")
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)


def _apply_knobs():
    for kv in filter(None, os.environ.get("XRF_KNOBS", "").split(";")):
        k, v = kv.split("=", 1)
        assert hasattr(pd, k), k
        setattr(pd, k, ast.literal_eval(v))
        print("knob", k, "=", getattr(pd, k))


def test_bench(device):
    import torch  # local: no global torch import under ttnn/

    _apply_knobs()
    torch.manual_seed(0)
    x = torch.randn(1, 1, 160, 1536).bfloat16()
    w = (torch.randn(1536, 4096) * 0.02).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )
    ref = (x.float() @ ttnn.to_torch(tw).float()).flatten()
    for i in range(6):
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 160, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        y = ttnn.generic_op([tx, tw, y], pd.create_program_descriptor(tx, tw, y, cfg))
        got = ttnn.to_torch(y).float().flatten()
        pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
        print("PCC", i, pcc)
        assert pcc >= 0.9998 or os.environ.get("XRF_KNOBS", "").find("ABLATE") >= 0, pcc
        ttnn.deallocate(y)
    ttnn.synchronize_device(device)
