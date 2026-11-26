# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""combine_critical_path bake-off: the full groupnorm_sc_N_1_HW_C_b op with its kernel source dir swapped per
variant (the descriptor module global KERNEL_DIR -> kernels_<variant>/; CT/RT args and the CB layout unchanged).

Run under the profiler, one fresh dispatch per (variant, case):
    scripts/run_safe_pytest.sh --profile --no-py-zones <this file>
Env:
    CCP_VARIANTS=base,both   variant dirs in run order (default base,both); every case runs them in this order
    CCP_CASES=a,b            case names, or "focus" (default) / "sweep" / "all"
    CCP_CHUNK_ZONES=1        diagnosis: request the op's per-chunk zones (TUNING_OVERRIDES["chunk_zones"])
Correctness gate per run: the fp32 torch reference at the perf contract (PCC >= 0.999, rel-RMS <= 0.01). Each run also
records whether its output is bit-identical to the first variant's output of the same case (same inputs, same
session). One line per run goes to runs.jsonl, aligned with the ops CSV's GenericOp rows by parse.py.
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
import importlib

opmod = importlib.import_module("ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b")

HERE = Path(__file__).parent
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

# name: (shape, G, grid or None (DRAM), layout, eps, in_place)
FOCUS = {
    "unet_rm_ip_1024x1280": ((1, 1, 1024, 1280), 32, (8, 8), RM, 1e-5, True),
    "unet_tile_1024x1280": ((1, 1, 1024, 1280), 32, (8, 8), TL, 1e-6, False),
    "unet_rm_ip_16384x960": ((1, 1, 16384, 960), 32, (8, 8), RM, 1e-5, True),
}
SWEEP = {
    "unet_rm_ip_16384x320": ((1, 1, 16384, 320), 32, (8, 8), RM, 1e-5, True),
    "unet_rm_ip_4096x320": ((1, 1, 4096, 320), 32, (8, 8), RM, 1e-5, True),
    "unet_rm_ip_4096x640": ((1, 1, 4096, 640), 32, (8, 8), RM, 1e-5, True),
    "unet_rm_ip_1024x640": ((1, 1, 1024, 640), 32, (8, 8), RM, 1e-5, True),
    "refiner_rm_ip_256x1536": ((1, 1, 256, 1536), 32, (8, 8), RM, 1e-5, True),
    "refiner_rm_ip_16384x384": ((1, 1, 16384, 384), 32, (8, 8), RM, 1e-5, True),
    "unet_rm_ip_16384x640": ((1, 1, 16384, 640), 32, (8, 8), RM, 1e-5, True),
    "unet_tile_4096x640_4x8": ((1, 1, 4096, 640), 32, (4, 8), TL, 1e-6, False),
    "vae_tile_16384x512": ((1, 1, 16384, 512), 32, (8, 8), TL, 1e-6, False),
    "flux1_dram_16384x256": ((1, 1, 16384, 256), 16, None, TL, 1e-6, False),
    "vae_dram_65536x512": ((1, 1, 65536, 512), 32, None, TL, 1e-6, False),
    "refiner_dram_16384x1152": ((1, 1, 16384, 1152), 32, None, TL, 1e-5, False),
}
ALL = {**FOCUS, **SWEEP}


def _case_names():
    sel = os.environ.get("CCP_CASES", "focus")
    if sel == "focus":
        return list(FOCUS)
    if sel == "sweep":
        return list(SWEEP)
    if sel == "all":
        return list(ALL)
    return sel.split(",")


VARIANTS = os.environ.get("CCP_VARIANTS", "base,both").split(",")
PARAMS = [(v, c) for c in _case_names() for v in VARIANTS]
RUNS = HERE / "runs.jsonl"
_first_out = {}  # case -> output of the first variant run on it (bit-identity reference)


@pytest.fixture(scope="module", autouse=True)
def _fresh_log():
    RUNS.write_text("")
    yield


@pytest.mark.parametrize("variant, case", PARAMS, ids=[f"{v}-{c}" for v, c in PARAMS])
def test_ccp(device, variant, case):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports

    shape, G, grid, layout, eps, in_place = ALL[case]
    saved = pd.KERNEL_DIR
    saved_tuning = dict(opmod.TUNING_OVERRIDES)
    pd.KERNEL_DIR = HERE / f"kernels_{variant}"
    if os.environ.get("CCP_CHUNK_ZONES") == "1":
        opmod.TUNING_OVERRIDES["chunk_zones"] = True
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
        y = ttnn.to_torch(out).reshape(shape)
        ref = pytorch_groupnorm_sc_N_1_HW_C(xt, G, gamma=gt, beta=bt, eps=eps)
        m = compute_metrics_torch(y, ref, readback_dtype=torch.bfloat16)
        first = _first_out.setdefault(case, (variant, y))
        ndiff = int((first[1].view(torch.int16) != y.view(torch.int16)).sum())
        with RUNS.open("a") as f:
            f.write(json.dumps(dict(variant=variant, case=case, pcc=m.pcc, rms=m.rms, vs=first[0], ndiff=ndiff)) + "\n")
        if in_place:
            ttnn.deallocate(out, force=True)
        else:
            out.deallocate()
            x.deallocate()
        assert m.pcc >= 0.999 and m.rms <= 0.01, f"{variant}/{case}: pcc {m.pcc:.6f} rms {m.rms:.5f}"
    finally:
        pd.KERNEL_DIR = saved
        opmod.TUNING_OVERRIDES.clear()
        opmod.TUNING_OVERRIDES.update(saved_tuning)
