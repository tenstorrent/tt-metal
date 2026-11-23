# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""tail_store2 whole-op A/B bench: wq_b_linear with this dir's descriptor copy (pd.py) + kernels.
WQB_KNOBS="NAME=value,..." overrides pd knobs (python literals), e.g. TAIL='TAIL_PRECOMP;TAIL_LAST=1',
ABLATE='x_mcast+x_read'. 1 warmup + 5 dispatches; PCC (vs torch) printed for every dispatch.
Run under scripts/run_safe_pytest.sh --profile; then tail.py <csv>."""
import ast
import importlib.util
import os
from pathlib import Path

import ttnn

_spec = importlib.util.spec_from_file_location("tail2_pd", Path(__file__).parent / "pd.py")
pd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pd)


def _apply_knobs():
    for kv in filter(None, os.environ.get("WQB_KNOBS", "").split(",")):
        k, v = kv.split("=", 1)
        assert hasattr(pd, k), k
        setattr(pd, k, ast.literal_eval(v))
        print("knob", k, "=", getattr(pd, k))


def test_tail2(device):
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
    ref = (x.float() @ ttnn.to_torch(tw).float()).flatten()
    pccs = []
    for i in range(6):
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 160, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        y = ttnn.generic_op([tx, tw, y], pd.create_program_descriptor(tx, tw, y, cfg))
        got = ttnn.to_torch(y).float().flatten()
        pccs.append(torch.corrcoef(torch.stack([ref, got]))[0, 1].item())
        ttnn.deallocate(y)
    ttnn.synchronize_device(device)
    print("PCC", min(pccs), [round(p, 7) for p in pccs])
    if not pd.ABLATE:
        assert min(pccs) >= 0.9998
