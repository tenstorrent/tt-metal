# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""moments_fpu_stream bench (Perf 2): the full op with the descriptor's KERNEL_DIR swapped per variant.

Adapted from perf_experiments/_round2/bench.py (same variant syntax), plus input distributions, affine modes and an
fp64 reference for the precision menu.

    R2_VARIANTS=a,b    kernel-dir[@flags] ('kernels' = this dir's variant source, 'op' = the op's own kernels);
                       flags '_'-joined -> GN_ABL_<FLAG>=1 defines on every kernel (one source dir, so the profiler's
                       16-bit zone-location hashes do not multiply per variant)
    R2_CASES=a,b       case names (see CASES) or a group (focus, sweep, all)
    R2_DIST=randn      input distribution(s), comma list: randn | mean50 (50 + N(0,1)) | mean10 | gmean
    R2_CHECK=0         skip the correctness gate
    R2_CHUNK_ZONES=1   request the op's per-chunk zones
One line per run -> runs.jsonl (variant, case, dist, pcc / rms vs the fp32 golden reference, pcc64 / rms64 vs fp64,
sha1 of the output bytes); the ops CSV rows follow the same order.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest
import ttnn

from eval.metrics import compute_metrics_torch
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config, pytorch_groupnorm_sc_N_1_HW_C
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

opmod = importlib.import_module("ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b")
HERE = Path(__file__).parent
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
G88, G48 = (8, 8), (4, 8)
GB, GO, NA = "gamma_beta", "gamma_only", "no_affine"
# name: (shape, G, grid or None (DRAM), layout, eps, in_place, affine, tuning overrides)
CASES = {
    # focus
    "unet_rm_ip_16384x960": ((1, 1, 16384, 960), 32, G88, RM, 1e-5, True, GB, {}),
    "vae_tile_16384x512": ((1, 1, 16384, 512), 32, G88, TL, 1e-6, False, GB, {}),
    "unet_rm_ip_16384x320": ((1, 1, 16384, 320), 32, G88, RM, 1e-5, True, GB, {}),
    "unet_rm_ip_1024x1280": ((1, 1, 1024, 1280), 32, G88, RM, 1e-5, True, GB, {}),
    # sharded sweep
    "unet_rm_oop_16384x320": ((1, 1, 16384, 320), 32, G88, RM, 1e-5, False, GB, {}),
    "unet_rm_oop_1024x1280": ((1, 1, 1024, 1280), 32, G88, RM, 1e-5, False, GB, {}),
    "unet_tile_4096x640_4x8": ((1, 1, 4096, 640), 32, G48, TL, 1e-6, False, GB, {}),
    "vae_tile_ip_16384x512": ((1, 1, 16384, 512), 32, G88, TL, 1e-6, True, GB, {}),
    "unet_tile_ip_1024x1280": ((1, 1, 1024, 1280), 32, G88, TL, 1e-6, True, GB, {}),
    "g_rm_ip_1024x1280_gonly": ((1, 1, 1024, 1280), 32, G88, RM, 1e-5, True, GO, {}),
    "g_tile_oop_1024x1280_none": ((1, 1, 1024, 1280), 32, G88, TL, 1e-5, False, NA, {}),
    # DRAM sweep
    "vae_dram_65536x512": ((1, 1, 65536, 512), 32, None, TL, 1e-6, False, GB, {}),
    "flux1_dram_16384x256": ((1, 1, 16384, 256), 16, None, TL, 1e-6, False, GB, {}),
    "mid_4096x640_rm": ((1, 1, 4096, 640), 32, None, RM, 1e-5, False, GB, {}),
    "g_dram_tile_16384x96_cg3": ((1, 1, 16384, 96), 32, None, TL, 1e-5, False, GB, {}),
    "g_dram_rm_4096x640_none": ((1, 1, 4096, 640), 32, None, RM, 1e-5, False, NA, {}),
    # precision-only: the per-chunk spill / reload paths (multi-chunk pass 1)
    "dram_stream_65536x128": (
        (1, 1, 65536, 128),
        32,
        None,
        TL,
        1e-5,
        False,
        GB,
        {"max_resident_chunks": 1, "chunk_tiles_target": 8},
    ),
    "rm_ip_partial_4096x640": (
        (1, 1, 4096, 640),
        32,
        G88,
        RM,
        1e-5,
        True,
        GB,
        {"max_resident_chunks": 1, "chunk_tiles_target": 5},
    ),
    "dram_w1_65536x32": ((1, 1, 65536, 32), 1, None, TL, 1e-6, False, GB, {}),
}
FOCUS = ["unet_rm_ip_16384x960", "vae_tile_16384x512", "unet_rm_ip_16384x320", "unet_rm_ip_1024x1280"]
SWEEP = [
    "unet_rm_oop_16384x320",
    "unet_rm_oop_1024x1280",
    "unet_tile_4096x640_4x8",
    "vae_tile_ip_16384x512",
    "unet_tile_ip_1024x1280",
    "g_rm_ip_1024x1280_gonly",
    "g_tile_oop_1024x1280_none",
    "vae_dram_65536x512",
    "flux1_dram_16384x256",
    "mid_4096x640_rm",
    "g_dram_tile_16384x96_cg3",
    "g_dram_rm_4096x640_none",
]
PREC = FOCUS + ["unet_tile_4096x640_4x8", "dram_stream_65536x128", "rm_ip_partial_4096x640", "dram_w1_65536x32"]
GROUPS = {"focus": FOCUS, "sweep": SWEEP, "prec": PREC, "all": list(CASES)}


def _case_names():
    sel = os.environ.get("R2_CASES", "focus")
    out = []
    for s in sel.split(","):
        out += GROUPS.get(s, [s])
    return out


VARIANTS = os.environ.get("R2_VARIANTS", "op").split(",")
# R2_ORDER=rev reverses the variant order inside every case (alternate A B / B A between runs).
if os.environ.get("R2_ORDER") == "rev":
    VARIANTS = VARIANTS[::-1]
DISTS = os.environ.get("R2_DIST", "randn").split(",")
PARAMS = [(v, c, d) for d in DISTS for c in _case_names() for v in VARIANTS]
RUNS = HERE / os.environ.get("R2_RUNS", "runs.jsonl")


@pytest.fixture(scope="module", autouse=True)
def _fresh_log():
    RUNS.write_text("")
    yield


def _make(dist, shape, G, seed=0):
    import torch

    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.float32)
    if dist == "randn":
        return x.to(torch.bfloat16)
    if dist.startswith("mean"):  # per-element offset: |mean| >> std, the shifted-moment regime
        return (x + float(dist[4:])).to(torch.bfloat16)
    if dist == "gmean":  # per-group means N(0, 8^2), std 1: mixed-sign offsets across groups
        C = shape[-1]
        return (x + 8.0 * torch.randn((1, 1, 1, G)).repeat_interleave(C // G, -1)).to(torch.bfloat16)
    raise ValueError(dist)


def _ref64(x_bf16, G, gamma, beta, eps):
    import torch

    x = x_bf16.to(torch.float64)
    N, _, HW, C = x.shape
    xg = x.reshape(N, HW, G, C // G)
    mean = xg.mean(dim=(1, 3), keepdim=True)
    var = ((xg - mean) ** 2).mean(dim=(1, 3), keepdim=True)
    y = ((xg - mean) / torch.sqrt(var + eps)).reshape(N, 1, HW, C)
    if gamma is not None:
        y = y * gamma.to(torch.float64).reshape(1, 1, 1, C)
    if beta is not None:
        y = y + beta.to(torch.float64).reshape(1, 1, 1, C)
    return y


@pytest.mark.parametrize("variant, case, dist", PARAMS, ids=[f"{v}-{c}-{d}" for v, c, d in PARAMS])
def test_r2(device, variant, case, dist):
    import torch

    shape, G, grid, layout, eps, in_place, affine, tov = CASES[case]
    saved = pd.KERNEL_DIR
    saved_tuning = dict(opmod.TUNING_OVERRIDES)
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
    opmod.TUNING_OVERRIDES.update(tov)
    if os.environ.get("R2_CHUNK_ZONES") == "1":
        opmod.TUNING_OVERRIDES["chunk_zones"] = True
    try:
        C = shape[-1]
        xt = _make(dist, shape, G)
        torch.manual_seed(7)
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
        check = os.environ.get("R2_CHECK", "1") != "0"
        rec = dict(variant=variant, case=case, dist=dist)
        if check:
            y = ttnn.to_torch(out).reshape(shape)
            rec["sha1"] = hashlib.sha1(y.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:12]
            ref = pytorch_groupnorm_sc_N_1_HW_C(xt, G, gamma=gt, beta=bt, eps=eps)
            m = compute_metrics_torch(y, ref, readback_dtype=torch.bfloat16)
            rec.update(pcc=m.pcc, rms=m.rms)
            r64 = _ref64(xt, G, gt, bt, eps).flatten()
            y64 = y.to(torch.float64).flatten()
            rec["rms64"] = ((y64 - r64).norm() / r64.norm()).item()
            rec["pcc64"] = torch.corrcoef(torch.stack([y64, r64]))[0, 1].item()
        with RUNS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if in_place:
            ttnn.deallocate(out, force=True)
        else:
            out.deallocate()
            x.deallocate()
        if check and dist == "randn":
            assert rec["pcc"] >= 0.999 and rec["rms"] <= 0.01, rec
    finally:
        pd.KERNEL_DIR = saved
        pd.create_program_descriptor = saved_create
        opmod.TUNING_OVERRIDES.clear()
        opmod.TUNING_OVERRIDES.update(saved_tuning)
