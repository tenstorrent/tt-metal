# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""compute_pack bake-off: whole op (copy in this dir) with the compute DEST window decoupled from the 2-k-tile x round.

Run via run.sh "<variant>-<mode>,...". Modes: full (PCC-gated 0.9998), comp (all DM payloads stubbed),
xstub (x read + x mcast stubbed). 1 warmup + 5 dispatches per run; parse.py reads them in CP_RUNS order.
"""
import importlib.util
import os
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cp_pd", HERE / "pd.py")
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)

VARIANTS = {
    "base": dict(),
    "based8": dict(W_DEPTH_BLOCKS=8),
    "w1": dict(WINDOWS=(1,) * 24),
    "w2": dict(WINDOWS=(2,) * 12),
    "g2": dict(WINDOWS=(1, 1) + (2,) * 10 + (1, 1)),
    "w4": dict(WINDOWS=(4,) * 6),
    "w4d8": dict(WINDOWS=(4,) * 6, W_DEPTH_BLOCKS=8),
    "g4d8": dict(WINDOWS=(1, 1, 2, 4, 4, 4, 4, 2, 1, 1), W_DEPTH_BLOCKS=8),
    "w8d8": dict(WINDOWS=(8,) * 3, W_DEPTH_BLOCKS=8),
    "w1l": dict(WINDOWS=(1,) * 24, LAST_ROWS=True),
    "g2l": dict(WINDOWS=(1, 1) + (2,) * 10 + (1, 1), LAST_ROWS=True),
}
MODES = {"full": frozenset(), "comp": "w_read+x_read+x_mcast+y_store", "xstub": "x_read+x_mcast"}
DEFAULTS = {k: getattr(pd, k) for k in ("WINDOWS", "W_DEPTH_BLOCKS", "ABLATE", "LAST_ROWS")}
RUNS = [tuple(r.split("-")) for r in os.environ.get("CP_RUNS", "").split(",") if r]


@pytest.mark.parametrize("variant,mode", RUNS)
def test_variant(device, variant, mode):
    import torch

    for k, v in {**DEFAULTS, **VARIANTS[variant], "ABLATE": MODES[mode]}.items():
        setattr(pd, k, v)
    torch.manual_seed(0)
    x = torch.randn(1, 1, 160, 1536).bfloat16()
    w = (torch.randn(1536, 4096) * 0.02).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )
    pcc = None
    for i in range(6):
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 160, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        y = ttnn.generic_op([tx, tw, y], pd.create_program_descriptor(tx, tw, y, cfg))
        if i == 0:
            ref = (x.float() @ ttnn.to_torch(tw).float()).flatten()
            got = ttnn.to_torch(y).float().flatten()
            pcc = torch.corrcoef(torch.stack([ref, got]))[0, 1].item()
        ttnn.deallocate(y)
    ttnn.synchronize_device(device)
    print(f"RESULT {variant}-{mode} PCC {pcc:.6f}")
    if mode == "full":
        assert pcc >= 0.9998, f"{variant}: PCC {pcc}"
