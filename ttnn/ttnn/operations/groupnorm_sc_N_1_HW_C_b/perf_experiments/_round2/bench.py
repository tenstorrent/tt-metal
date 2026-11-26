# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 coordinator bench: the full op with the descriptor's KERNEL_DIR swapped per variant.

    R2_VARIANTS=dirA,dirB  kernel dirs (relative to this file's dir, or absolute); 'op' = the op's own kernels
    R2_CASES=a,b           case names (see CASES) or a group: focus_sh, focus_dram, small, all
    R2_CHECK=0             skip the correctness gate (ablation stubs produce garbage)
    R2_CHUNK_ZONES=1       request the op's per-chunk zones
One line per run -> runs.jsonl (variant, case, pcc, rms); the ops CSV rows follow the same order.
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
G88, G48 = (8, 8), (4, 8)
# name: (shape, G, grid or None (DRAM), layout, eps, in_place)
CASES = {
    # PERF FOCUS, sharded (RM in place, TILE out of place: the model forms)
    "unet_rm_ip_1024x1280": ((1, 1, 1024, 1280), 32, G88, RM, 1e-5, True),
    "unet_rm_ip_16384x320": ((1, 1, 16384, 320), 32, G88, RM, 1e-5, True),
    "unet_rm_ip_4096x320": ((1, 1, 4096, 320), 32, G88, RM, 1e-5, True),
    "unet_rm_ip_4096x640": ((1, 1, 4096, 640), 32, G88, RM, 1e-5, True),
    "unet_rm_ip_1024x640": ((1, 1, 1024, 640), 32, G88, RM, 1e-5, True),
    "refiner_rm_ip_256x1536": ((1, 1, 256, 1536), 32, G88, RM, 1e-5, True),
    "refiner_rm_ip_16384x384": ((1, 1, 16384, 384), 32, G88, RM, 1e-5, True),
    "unet_rm_ip_16384x640": ((1, 1, 16384, 640), 32, G88, RM, 1e-5, True),
    "unet_rm_ip_16384x960": ((1, 1, 16384, 960), 32, G88, RM, 1e-5, True),
    "unet_tile_1024x1280": ((1, 1, 1024, 1280), 32, G88, TL, 1e-6, False),
    "unet_tile_4096x640_4x8": ((1, 1, 4096, 640), 32, G48, TL, 1e-6, False),
    "vae_tile_16384x512": ((1, 1, 16384, 512), 32, G88, TL, 1e-6, False),
    # PERF FOCUS, DRAM interleaved TILE
    "vae_dram_1048576x128": ((1, 1, 1048576, 128), 32, None, TL, 1e-6, False),
    "vae_dram_65536x512": ((1, 1, 65536, 512), 32, None, TL, 1e-6, False),
    "vae_dram_262144x256": ((1, 1, 262144, 256), 32, None, TL, 1e-6, False),
    "vae_dram_262144x512": ((1, 1, 262144, 512), 32, None, TL, 1e-6, False),
    "vae_dram_1048576x256": ((1, 1, 1048576, 256), 32, None, TL, 1e-6, False),
    "refiner_dram_16384x1152": ((1, 1, 16384, 1152), 32, None, TL, 1e-5, False),
    "flux1_dram_16384x256": ((1, 1, 16384, 256), 16, None, TL, 1e-6, False),
    "flux1_dram_1048576x64": ((1, 1, 1048576, 64), 16, None, TL, 1e-6, False),
    "mochi_dram_8x407040x128": ((8, 1, 407040, 128), 32, None, TL, 1e-6, False),
    # mid / guard
    "mid_4096x640_rm": ((1, 1, 4096, 640), 32, None, RM, 1e-5, False),
    "mid_16384x320_tile": ((1, 1, 16384, 320), 32, None, TL, 1e-5, False),
    "unet_tile_ip_1024x1280": ((1, 1, 1024, 1280), 32, G88, TL, 1e-6, True),
    "unet_rm_oop_1024x1280": ((1, 1, 1024, 1280), 32, G88, RM, 1e-5, False),
    # Refinement 3: ragged shard geometries (grid = (grid_x, grid_y) as in the golden "bsXxY" ids)
    "r3_tile_4096x960_padcol": ((1, 1, 4096, 960), 32, G88, TL, 1e-6, False),
    "r3_tile_1024x1920_padcol": ((1, 1, 1024, 1920), 32, G88, TL, 1e-6, False),
    "r3_tile_4096x1152_padcol": ((1, 1, 4096, 1152), 32, G88, TL, 1e-6, False),
    "r3_rm_ip_4096x320_8x7_padrow": ((1, 1, 4096, 320), 32, (8, 7), RM, 1e-5, True),
    "r3_tile_1024x1280_8x7_padrow": ((1, 1, 1024, 1280), 32, (8, 7), TL, 1e-6, False),
    "r3_tile_2x256x1280_8x3_straddle": ((2, 1, 256, 1280), 32, (8, 3), TL, 1e-6, False),
    "r3_rm_ip_2x256x1280_8x3_straddle": ((2, 1, 256, 1280), 32, (8, 3), RM, 1e-5, True),
    "r3_tile_1024x1280_7x8_grpcol": ((1, 1, 1024, 1280), 32, (7, 8), TL, 1e-6, False),
    "r3_rm_ip_1024x1280_7x8_grpcol": ((1, 1, 1024, 1280), 32, (7, 8), RM, 1e-5, True),
    "r3_rm_ip_4096x640_6x8_padcol": ((1, 1, 4096, 640), 32, (6, 8), RM, 1e-5, True),
    # the same ragged shapes on the DRAM interleaved path (the only route they had before R3)
    "r3_dram_tile_4096x960": ((1, 1, 4096, 960), 32, None, TL, 1e-6, False),
    "r3_dram_tile_1024x1920": ((1, 1, 1024, 1920), 32, None, TL, 1e-6, False),
    "r3_dram_tile_4096x1152": ((1, 1, 4096, 1152), 32, None, TL, 1e-6, False),
    "r3_dram_rm_4096x320": ((1, 1, 4096, 320), 32, None, RM, 1e-5, False),
    "r3_dram_tile_2x256x1280": ((2, 1, 256, 1280), 32, None, TL, 1e-6, False),
    "r3_dram_rm_1024x1280": ((1, 1, 1024, 1280), 32, None, RM, 1e-5, False),
    "r3_dram_rm_4096x640": ((1, 1, 4096, 640), 32, None, RM, 1e-5, False),
}
GROUPS = {
    "focus_sh": [
        k for k, v in CASES.items() if v[2] is not None and not k.startswith(("unet_tile_ip", "unet_rm_oop", "r3_"))
    ],
    "focus_dram": [k for k, v in CASES.items() if k.split("_")[1] == "dram" or k.startswith("mochi")],
    "r3": [k for k in CASES if k.startswith("r3_")],
    "sh_all": [k for k, v in CASES.items() if v[2] is not None and not k.startswith("r3_")],
    "dram_small": [
        "flux1_dram_16384x256",
        "refiner_dram_16384x1152",
        "vae_dram_65536x512",
        "mid_4096x640_rm",
        "mid_16384x320_tile",
    ],
    "all": list(CASES),
}


def _case_names():
    sel = os.environ.get("R2_CASES", "focus_sh")
    out = []
    for s in sel.split(","):
        out += GROUPS.get(s, [s])
    return out


VARIANTS = os.environ.get("R2_VARIANTS", "op").split(",")
PARAMS = [(v, c) for c in _case_names() for v in VARIANTS]
RUNS = HERE / os.environ.get("R2_RUNS", "runs.jsonl")


@pytest.fixture(scope="module", autouse=True)
def _fresh_log():
    RUNS.write_text("")
    yield


@pytest.mark.parametrize("variant, case", PARAMS, ids=[f"{v}-{c}" for v, c in PARAMS])
def test_r2(device, variant, case):
    import torch

    shape, G, grid, layout, eps, in_place = CASES[case]
    saved = pd.KERNEL_DIR
    saved_tuning = dict(opmod.TUNING_OVERRIDES)
    # variant = <kernel dir>[@<flags>]: flags '_'-joined -> GN_ABL_<FLAG>=1 defines on every kernel (one source dir,
    # so the profiler's 16-bit zone-location hashes do not multiply per variant and collide)
    vdir, _, flags = variant.partition("@")
    if vdir != "op":
        p = Path(vdir)
        pd.KERNEL_DIR = p if p.is_absolute() else HERE / p
    saved_create = pd.create_program_descriptor
    if flags:
        defs = [(f"GN_ABL_{f.upper()}", "1") for f in flags.split("_")]

        def _create(*a, **kw):
            desc = saved_create(*a, **kw)
            for k in desc.kernels:
                k.defines = list(k.defines) + defs
            return desc

        pd.create_program_descriptor = _create
    if os.environ.get("R2_CHUNK_ZONES") == "1":
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
        check = os.environ.get("R2_CHECK", "1") != "0"
        rec = dict(variant=variant, case=case)
        if check:
            y = ttnn.to_torch(out).reshape(shape)
            ref = pytorch_groupnorm_sc_N_1_HW_C(xt, G, gamma=gt, beta=bt, eps=eps)
            m = compute_metrics_torch(y, ref, readback_dtype=torch.bfloat16)
            rec.update(pcc=m.pcc, rms=m.rms)
        with RUNS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if in_place:
            ttnn.deallocate(out, force=True)
        else:
            out.deallocate()
            x.deallocate()
        if check:
            assert rec["pcc"] >= 0.999 and rec["rms"] <= 0.01, rec
    finally:
        pd.KERNEL_DIR = saved
        pd.create_program_descriptor = saved_create
        opmod.TUNING_OVERRIDES.clear()
        opmod.TUNING_OVERRIDES.update(saved_tuning)
