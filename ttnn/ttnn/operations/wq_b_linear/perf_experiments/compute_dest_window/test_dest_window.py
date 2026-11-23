# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""compute_dest_window bake-off: whole op (copy in this dir) with the DEST window decoupled from the x round.

Run: scripts/run_safe_pytest.sh --profile <this file> -k "<variant ids>"
Each test = one variant x mode ("full" or "comp" = all DM payloads stubbed). 1 warmup + 5 dispatches;
the report's profile_log_device.csv holds 6 dispatches per run in CDW_RUNS order (run.sh parses it). PCC (full mode only) vs float matmul of the device-quantized operands, gate 0.9998.
"""
import importlib.util
import os
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cdw_pd", HERE / "pd.py")
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)

VARIANTS = {
    "base": dict(WINDOWS=None),
    "w1": dict(WINDOWS=(1,) * 12),
    "w2": dict(WINDOWS=(2,) * 6),
    "g2": dict(WINDOWS=(1, 1, 2, 2, 2, 2, 2)),
    "w3d6": dict(WINDOWS=(3,) * 4, W_DEPTH_BLOCKS=6),
    "g3d6": dict(WINDOWS=(1, 2, 3, 3, 3), W_DEPTH_BLOCKS=6),
    "w4": dict(WINDOWS=(4,) * 3),
    "w4d8": dict(WINDOWS=(4,) * 3, W_DEPTH_BLOCKS=8),
    "g4d8": dict(WINDOWS=(1, 3, 4, 4), W_DEPTH_BLOCKS=8),
    "based6": dict(WINDOWS=None, W_DEPTH_BLOCKS=6),
    "based8": dict(WINDOWS=None, W_DEPTH_BLOCKS=8),
    "w1r": dict(WINDOWS=(1,) * 12, RECONFIG_EACH=True),
    "hco": dict(HELPER_CALLER_OWNS=True, OUT_SUBBLOCK_MAJOR=False),
    "w1d": dict(WINDOWS=(1,) * 12, DRAIN_PROBE=True),
}
MODES = {"full": frozenset(), "comp": "w_read+x_read+x_mcast+y_store"}
DEFAULTS = {
    k: getattr(pd, k)
    for k in (
        "WINDOWS",
        "W_DEPTH_BLOCKS",
        "ABLATE",
        "RECONFIG_EACH",
        "DRAIN_PROBE",
        "HELPER_CALLER_OWNS",
        "OUT_SUBBLOCK_MAJOR",
    )
}


# CDW_RUNS="base-full,w2-comp,..." selects runs (in that order); default all.
RUNS = [tuple(r.split("-")) for r in os.environ.get("CDW_RUNS", "").split(",") if r] or [
    (v, m) for v in VARIANTS for m in MODES
]


@pytest.mark.parametrize("variant,mode", RUNS)
def test_variant(device, variant, mode):
    import torch  # local: no global torch import under ttnn/

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
            slope = (got @ ref / (ref @ ref)).item()
        ttnn.deallocate(y)
    ttnn.synchronize_device(device)
    print(f"RESULT {variant}-{mode} PCC {pcc:.6f} slope {slope:.5f}")
    if mode == "full":
        assert pcc >= 0.9998, f"{variant}: PCC {pcc}"
