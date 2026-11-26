# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B bench for the prologue_off_critical_path idea: the full groupnorm_sc_N_1_HW_C_b op with its kernel source
dir swapped per variant (descriptor module global KERNEL_DIR; CT/RT args and the CB layout unchanged).

Run under the profiler, one dispatch per (variant, case):
    scripts/run_safe_pytest.sh --profile --no-py-zones <this file> -k <token>
Env:
    PL_VARIANTS=base,cand   variant dirs (kernels_<name>) in run order (default base,cand)
    PL_CASES=a,b            case names (default: the three focus shapes); "sweep" = the domain sweep, "all" = both
Every run is correctness-gated against the fp32 torch reference (perf contract PCC >= 0.999, rel-RMS <= 0.01) and
appends one line to runs.jsonl (variant, case, pcc, rms) so parse.py can align the GenericOp rows of the ops CSV.
"""

import json
import os
from pathlib import Path

import pytest
import ttnn

from eval.metrics import compute_metrics_torch
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config, pytorch_groupnorm_sc_N_1_HW_C
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

HERE = Path(__file__).parent
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

# name: (shape, G, grid or None (DRAM), layout, eps, in_place)
FOCUS = {
    "unet_rm_ip_1024x1280": ((1, 1, 1024, 1280), 32, (8, 8), RM, 1e-5, True),
    "unet_rm_ip_16384x960": ((1, 1, 16384, 960), 32, (8, 8), RM, 1e-5, True),
    "vae_tile_16384x512": ((1, 1, 16384, 512), 32, (8, 8), TL, 1e-6, False),
}
SWEEP = {
    "unet_tile_1024x1280": ((1, 1, 1024, 1280), 32, (8, 8), TL, 1e-6, False),
    "unet_tile_4096x640_4x8": ((1, 1, 4096, 640), 32, (4, 8), TL, 1e-6, False),
    "unet_rm_oop_1024x1280": ((1, 1, 1024, 1280), 32, (8, 8), RM, 1e-5, False),
    "unet_rm_oop_16384x320": ((1, 1, 16384, 320), 32, (8, 8), RM, 1e-5, False),
    "unet_rm_oop_4096x320": ((1, 1, 4096, 320), 32, (8, 8), RM, 1e-5, False),
    "unet_rm_oop_4096x640": ((1, 1, 4096, 640), 32, (8, 8), RM, 1e-5, False),
    "unet_rm_oop_1024x640": ((1, 1, 1024, 640), 32, (8, 8), RM, 1e-5, False),
    "refiner_rm_oop_256x1536": ((1, 1, 256, 1536), 32, (8, 8), RM, 1e-5, False),
    "refiner_rm_oop_16384x384": ((1, 1, 16384, 384), 32, (8, 8), RM, 1e-5, False),
    "unet_rm_ip_16384x320": ((1, 1, 16384, 320), 32, (8, 8), RM, 1e-5, True),
    "unet_rm_ip_4096x320": ((1, 1, 4096, 320), 32, (8, 8), RM, 1e-5, True),
    "unet_rm_ip_4096x640": ((1, 1, 4096, 640), 32, (8, 8), RM, 1e-5, True),
    "unet_rm_ip_1024x640": ((1, 1, 1024, 640), 32, (8, 8), RM, 1e-5, True),
    "refiner_rm_ip_256x1536": ((1, 1, 256, 1536), 32, (8, 8), RM, 1e-5, True),
    "refiner_rm_ip_16384x384": ((1, 1, 16384, 384), 32, (8, 8), RM, 1e-5, True),
    "unet_rm_ip_16384x640": ((1, 1, 16384, 640), 32, (8, 8), RM, 1e-5, True),
    "unet_tile_ip_1024x1280": ((1, 1, 1024, 1280), 32, (8, 8), TL, 1e-6, True),
    "unet_tile_ip_4096x640_4x8": ((1, 1, 4096, 640), 32, (4, 8), TL, 1e-6, True),
    "vae_tile_ip_16384x512": ((1, 1, 16384, 512), 32, (8, 8), TL, 1e-6, True),
    "vae_dram_262144x256": ((1, 1, 262144, 256), 32, None, TL, 1e-6, False),
    "flux1_dram_16384x256": ((1, 1, 16384, 256), 16, None, TL, 1e-6, False),
}
# Extra regimes (not in the timed sweep): DRAM interleaved ROW_MAJOR input (RM tilize path off the shard).
EXTRA = {
    "mid_4096x640_rm_dram": ((1, 1, 4096, 640), 32, None, RM, 1e-5, False),
    "mid_16384x320_rm_dram": ((1, 1, 16384, 320), 32, None, RM, 1e-5, False),
    # Odd channels per group (halfword pilot stores): DRAM TILE / DRAM RM Cg=3, sharded RM Cg=3, sharded TILE Cg=3.
    "oddcg3_dram_tile_1024x96": ((1, 1, 1024, 96), 32, None, TL, 1e-5, False),
    "oddcg3_dram_rm_1024x96": ((1, 1, 1024, 96), 32, None, RM, 1e-5, False),
    # (2048x480 G=32 Cg=15 on 8x8 is a ragged shard -> rejected by the op's SUPPORTED; use G=256 Cg=3 instead.)
    "oddcg3_rm_ip_1024x768_g256": ((1, 1, 1024, 768), 256, (8, 8), RM, 1e-5, True),
    "oddcg3_tile_1024x768_g256": ((1, 1, 1024, 768), 256, (8, 8), TL, 1e-5, False),
}
ALL = {**FOCUS, **SWEEP, **EXTRA}


def _case_names():
    sel = os.environ.get("PL_CASES", "focus")
    if sel == "focus":
        return list(FOCUS)
    if sel == "sweep":
        return list(SWEEP)
    if sel == "all":
        return list(ALL)
    return sel.split(",")


VARIANTS = os.environ.get("PL_VARIANTS", "base,cand").split(",")
PARAMS = [(v, c) for c in _case_names() for v in VARIANTS]
RUNS = HERE / "runs.jsonl"


@pytest.fixture(scope="module", autouse=True)
def _fresh_log():
    RUNS.write_text("")
    yield


@pytest.mark.parametrize("variant, case", PARAMS, ids=[f"{v}-{c}" for v, c in PARAMS])
def test_pl(device, variant, case):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports

    shape, G, grid, layout, eps, in_place = ALL[case]
    saved = pd.KERNEL_DIR
    pd.KERNEL_DIR = HERE / f"kernels_{variant}"
    try:
        torch.manual_seed(0)
        C = shape[-1]
        xt = torch.randn(shape, dtype=torch.bfloat16)
        gt = torch.randn((1, 1, 1, C), dtype=torch.bfloat16)
        bt = torch.randn((1, 1, 1, C), dtype=torch.bfloat16)
        mem = (
            ttnn.DRAM_MEMORY_CONFIG
            if grid is None
            else block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
        )
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)
        g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=RM, device=device)
        b = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=RM, device=device)
        out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps, in_place=in_place)
        ttnn.synchronize_device(device)
        y = ttnn.to_torch(out)
        ref = pytorch_groupnorm_sc_N_1_HW_C(xt, G, gamma=gt, beta=bt, eps=eps)
        m = compute_metrics_torch(y.reshape(shape), ref, readback_dtype=torch.bfloat16)
        with RUNS.open("a") as f:
            f.write(json.dumps(dict(variant=variant, case=case, pcc=m.pcc, rms=m.rms)) + "\n")
        if in_place:
            ttnn.deallocate(out, force=True)
        else:
            out.deallocate()
            x.deallocate()
        assert m.pcc >= 0.999 and m.rms <= 0.01, f"{variant}/{case}: pcc {m.pcc:.6f} rms {m.rms:.5f}"
    finally:
        pd.KERNEL_DIR = saved
