# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""combine_tail bench: the full op with the combine-tail options selected per variant (one kernel source dir, src/,
define-selected; the descriptor is this dir's flag-parametrized copy of the op's).

    CT_VARIANTS=base,ar,mr,ar+mr,...   '+'-joined flags (ar, mr, fraw); 'base' = no flags; 'op' = the op itself
    CT_CASES=a,b                       case names (see CASES) or a group (focus, sweep, guard, all)
    CT_RUNS=runs.jsonl                 output log (one line per run: variant, case, pcc, rms, bit-diff vs first variant)
The ops CSV rows follow the runs.jsonl order (parse.py joins them).
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

import importlib.util

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("ct_descriptor", HERE / "descriptor.py")
ctd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctd)

RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
G88, G48 = (8, 8), (4, 8)
GB, GO, NA = "gamma_beta", "gamma_only", "no_affine"
# name: (shape, G, grid or None (DRAM), layout, eps, in_place, affine)
CASES = {
    # focus
    "unet_rm_ip_1024x1280": ((1, 1, 1024, 1280), 32, G88, RM, 1e-5, True, GB),
    "unet_tile_1024x1280": ((1, 1, 1024, 1280), 32, G88, TL, 1e-6, False, GB),
    "refiner_rm_ip_256x1536": ((1, 1, 256, 1536), 32, G88, RM, 1e-5, True, GB),
    "unet_rm_ip_1024x640": ((1, 1, 1024, 640), 32, G88, RM, 1e-5, True, GB),
    "unet_tile_4096x640_4x8": ((1, 1, 4096, 640), 32, G48, TL, 1e-6, False, GB),
    # sweep: large sharded, RM out of place, DRAM TILE / RM
    "unet_rm_ip_16384x960": ((1, 1, 16384, 960), 32, G88, RM, 1e-5, True, GB),
    "vae_tile_16384x512": ((1, 1, 16384, 512), 32, G88, TL, 1e-6, False, GB),
    "unet_rm_oop_1024x1280": ((1, 1, 1024, 1280), 32, G88, RM, 1e-5, False, GB),
    "vae_dram_65536x512": ((1, 1, 65536, 512), 32, None, TL, 1e-6, False, GB),
    "flux1_dram_16384x256": ((1, 1, 16384, 256), 16, None, TL, 1e-6, False, GB),
    "refiner_dram_16384x1152": ((1, 1, 16384, 1152), 32, None, TL, 1e-5, False, GB),
    "mid_4096x640_rm": ((1, 1, 4096, 640), 32, None, RM, 1e-5, False, GB),
    # guards: gamma-only / no-affine per path, Cg = 3
    "g_rm_ip_1024x1280_gonly": ((1, 1, 1024, 1280), 32, G88, RM, 1e-5, True, GO),
    "g_rm_ip_1024x1280_none": ((1, 1, 1024, 1280), 32, G88, RM, 1e-5, True, NA),
    "g_tile_oop_1024x1280_gonly": ((1, 1, 1024, 1280), 32, G88, TL, 1e-5, False, GO),
    "g_tile_oop_1024x1280_none": ((1, 1, 1024, 1280), 32, G88, TL, 1e-5, False, NA),
    "g_dram_tile_4096x640_gonly": ((1, 1, 4096, 640), 32, None, TL, 1e-5, False, GO),
    "g_dram_tile_4096x640_none": ((1, 1, 4096, 640), 32, None, TL, 1e-5, False, NA),
    "g_dram_rm_4096x640_gonly": ((1, 1, 4096, 640), 32, None, RM, 1e-5, False, GO),
    "g_dram_rm_4096x640_none": ((1, 1, 4096, 640), 32, None, RM, 1e-5, False, NA),
    "g_dram_tile_16384x96_cg3": ((1, 1, 16384, 96), 32, None, TL, 1e-5, False, GB),
    # multi-block DRAM (U >= P -> cpi = 1, several units per core, no peers)
    "b16_dram_tile_1024x1280": ((16, 1, 1024, 1280), 32, None, TL, 1e-5, False, GB),
    # correctness-only (not in the measured sweep): U >= P -> cpi = 1, several blocks per core
    "c_b160_dram_tile_64x640": ((160, 1, 64, 640), 32, None, TL, 1e-5, False, GB),
    "c_b160_dram_rm_64x640_gonly": ((160, 1, 64, 640), 32, None, RM, 1e-5, False, GO),
}
_names = list(CASES)
GROUPS = {
    "focus": _names[:5],
    "sweep": _names[5:12],
    "guard": _names[12:22],
    "multiblock": _names[22:],
    "all": _names,
}


def _case_names():
    out = []
    for s in os.environ.get("CT_CASES", "focus").split(","):
        out += GROUPS.get(s, [s])
    return out


VARIANTS = os.environ.get("CT_VARIANTS", "base").split(",")
PARAMS = [(v, c) for c in _case_names() for v in VARIANTS]
RUNS = HERE / os.environ.get("CT_RUNS", "runs.jsonl")
_first_out = {}
_plan_info = {}


@pytest.fixture(scope="module", autouse=True)
def _fresh_log():
    RUNS.write_text("")
    yield


@pytest.mark.parametrize("variant, case", PARAMS, ids=[f"{v}-{c}" for v, c in PARAMS])
def test_ct(device, variant, case):
    shape, G, grid, layout, eps, in_place, affine = CASES[case]
    saved_create = pd.create_program_descriptor
    if variant != "op":
        flags = () if variant == "base" else tuple(variant.split("+"))

        def _create(*a, **kw):
            plan = a[0]
            _plan_info[case] = dict(cpi=plan.cpi, max_blocks=max(w.num_blocks for w in plan.work), R=plan.R)
            return ctd.create_program_descriptor(*a, **kw, flags=flags, kernel_dir=HERE / "src")

        pd.create_program_descriptor = _create
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports

    try:
        torch.manual_seed(0)
        C = shape[-1]
        xt = torch.randn(shape, dtype=torch.bfloat16)
        gt = torch.randn((1, 1, 1, C), dtype=torch.bfloat16) if affine in (GB, GO) else None
        bt = torch.randn((1, 1, 1, C), dtype=torch.bfloat16) if affine == GB else None
        mem = (
            ttnn.DRAM_MEMORY_CONFIG
            if grid is None
            else block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
        )
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)
        g = None if gt is None else ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=RM, device=device)
        b = None if bt is None else ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=RM, device=device)
        out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps, in_place=in_place)
        ttnn.synchronize_device(device)
        y = ttnn.to_torch(out).reshape(shape)
        ref = pytorch_groupnorm_sc_N_1_HW_C(xt, G, gamma=gt, beta=bt, eps=eps)
        m = compute_metrics_torch(y, ref, readback_dtype=torch.bfloat16)
        rec = dict(variant=variant, case=case, pcc=m.pcc, rms=m.rms, **_plan_info.get(case, {}))
        if case in _first_out and variant != VARIANTS[0]:
            y0 = _first_out[case]
            rec["ndiff"] = int((y.view(torch.int16) != y0.view(torch.int16)).sum())
            rec["maxdiff"] = float((y.float() - y0.float()).abs().max())
        else:
            _first_out[case] = y.clone()
        with RUNS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if in_place:
            ttnn.deallocate(out, force=True)
        else:
            out.deallocate()
            x.deallocate()
        assert rec["pcc"] >= 0.999 and rec["rms"] <= 0.01, rec
    finally:
        pd.create_program_descriptor = saved_create
