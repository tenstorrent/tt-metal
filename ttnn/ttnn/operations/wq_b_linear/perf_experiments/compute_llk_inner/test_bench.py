# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""compute_llk_inner bench: the wq_b_linear op with a swappable compute kernel (local copy of descriptor + kernels).

Run: WQB_KNOBS="COMPUTE_KERNEL='cand.cpp',ABLATE='w_read+x_read+x_mcast+y_store'" \
     scripts/run_safe_pytest.sh --profile <this file> -s
then: python3 run.py parse <report dir>. ABLATE of all DM = compute alone (x/W data is whatever sits in L1:
PCC is only meaningful with ABLATE empty). Dispatches 1 warmup + 5.
"""
import ast
import importlib.util
import os
from pathlib import Path

import ttnn

_spec = importlib.util.spec_from_file_location("cli_pd", Path(__file__).parent / "pd.py")
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)


def _apply_knobs():
    for kv in filter(None, os.environ.get("WQB_KNOBS", "").split(",")):
        k, v = kv.split("=", 1)
        assert hasattr(pd, k), k
        setattr(pd, k, ast.literal_eval(v))
        print("knob", k, "=", getattr(pd, k))


def _run(tx, tw, cfg):
    device = tx.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 160, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    desc = pd.create_program_descriptor(tx, tw, out, cfg)
    return ttnn.generic_op([tx, tw, out], desc)


def test_bench(device):
    import torch  # local: no global torch import under ttnn/

    _apply_knobs()
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )
    torch.manual_seed(0)
    x = torch.randn(1, 1, 160, 1536).bfloat16()
    w = (torch.randn(1536, 4096) * 0.02).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    for i in range(6):
        y = _run(tx, tw, cfg)
        if i == 0:
            ref = (x.float() @ ttnn.to_torch(tw).float()).flatten()
            got = ttnn.to_torch(y).float().flatten()
            pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
            print("PCC", pcc, "nan_got", got.isnan().sum().item())
        ttnn.deallocate(y)
    ttnn.synchronize_device(device)
