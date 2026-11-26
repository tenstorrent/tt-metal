# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A/B bench of kernel variants of groupnorm_sc_N_1_HW_C_b (per_tile_pass_chains experiment).

Each variant is a kernel-source dir under this experiment dir (k_<variant>/). The test swaps the program
descriptor's KERNEL_DIR per case (CT/RT args and CB layout are the op's own), runs the full op once, and
checks PCC / rel-RMS against the fp32 torch reference (PERF_TOLERANCE 0.999 / 0.01). Correctness is the only
pass/fail; device time is read from the profiler CSV (one GenericOp row per test, in test order).

env:
  PTC_VARIANTS = comma list of variants, run in that order per case (default "base")
  PTC_CASES    = comma list of case names (default: the three focus shapes)
  PTC_ZONES=1  -> compile the per-chunk device zones in (breakdowns only)

run:
  PTC_VARIANTS=base,cand scripts/run_safe_pytest.sh --profile --no-py-zones \
      ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/per_tile_pass_chains/bench_variants.py
"""

import os
from pathlib import Path

import pytest
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C_b import TUNING_OVERRIDES, groupnorm_sc_N_1_HW_C_b
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd

HERE = Path(__file__).parent
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

# name -> (shape, G, grid or None (DRAM), layout, in_place, eps)
ALL_CASES = {
    "vae_tile_16384x512": ((1, 1, 16384, 512), 32, (8, 8), TL, False, 1e-6),
    "unet_rm_ip_16384x320": ((1, 1, 16384, 320), 32, (8, 8), RM, True, 1e-5),
    "unet_rm_ip_16384x960": ((1, 1, 16384, 960), 32, (8, 8), RM, True, 1e-5),
    "unet_rm_ip_1024x1280": ((1, 1, 1024, 1280), 32, (8, 8), RM, True, 1e-5),
    "unet_rm_ip_4096x320": ((1, 1, 4096, 320), 32, (8, 8), RM, True, 1e-5),
    "unet_rm_ip_4096x640": ((1, 1, 4096, 640), 32, (8, 8), RM, True, 1e-5),
    "unet_rm_ip_1024x640": ((1, 1, 1024, 640), 32, (8, 8), RM, True, 1e-5),
    "refiner_rm_ip_256x1536": ((1, 1, 256, 1536), 32, (8, 8), RM, True, 1e-5),
    "refiner_rm_ip_16384x384": ((1, 1, 16384, 384), 32, (8, 8), RM, True, 1e-5),
    "unet_rm_ip_16384x640": ((1, 1, 16384, 640), 32, (8, 8), RM, True, 1e-5),
    "unet_tile_1024x1280": ((1, 1, 1024, 1280), 32, (8, 8), TL, False, 1e-6),
    "unet_tile_4096x640_4x8": ((1, 1, 4096, 640), 32, (4, 8), TL, False, 1e-6),
    "unet_rm_oop_16384x320": ((1, 1, 16384, 320), 32, (8, 8), RM, False, 1e-5),
    "vae_dram_262144x256": ((1, 1, 262144, 256), 32, None, TL, False, 1e-6),
    "mid_4096x640_rm": ((1, 1, 4096, 640), 32, None, RM, False, 1e-5),
}
FOCUS = ["vae_tile_16384x512", "unet_rm_ip_16384x320", "unet_rm_ip_16384x960"]

# Variants that need extra (non-aliased) CBs: variant-name prefix -> [(cb_id, dtype, page_bytes, pages)].
# The plan's footprint grows by the same bytes, so the L1 plan sees them.
EXTRA_CBS = {
    "n1": [(25, ttnn.bfloat16, 2048, 1)],  # constant -1 tile (normalize folded into two ELWMULs)
    "p1l": [(25, ttnn.bfloat16, 2048, 1), (26, ttnn.bfloat16, 2048, 8)],  # + the x - k ring (2 DEST windows)
}


def _extra_cbs(variant):
    for prefix, cbs in EXTRA_CBS.items():
        if variant.startswith(prefix):
            return cbs
    return []


VARIANTS = [v for v in os.environ.get("PTC_VARIANTS", "base").split(",") if v]
CASES = [c for c in os.environ.get("PTC_CASES", ",".join(FOCUS)).split(",") if c]
PARAMS = [(c, v) for c in CASES for v in VARIANTS]


@pytest.fixture(autouse=True)
def _zones():
    saved = dict(TUNING_OVERRIDES)
    if os.environ.get("PTC_ZONES"):
        TUNING_OVERRIDES["chunk_zones"] = True
    yield
    TUNING_OVERRIDES.clear()
    TUNING_OVERRIDES.update(saved)


def _metrics(out, ref):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports

    o = out.to(torch.float64).flatten()
    r = ref.to(torch.float64).flatten()
    pcc = torch.corrcoef(torch.stack([o, r]))[0, 1].item()
    rms = (torch.sqrt(torch.mean((o - r) ** 2)) / torch.sqrt(torch.mean(r**2))).item()
    return pcc, rms


@pytest.mark.parametrize("case, variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_variant(device, case, variant):
    kdir = HERE / f"k_{variant}"
    assert kdir.is_dir(), kdir
    pd.KERNEL_DIR = kdir
    extra = _extra_cbs(variant)
    orig_fp, orig_cpd = pd.footprint_bytes, pd.create_program_descriptor
    if extra:
        extra_bytes = sum(page * pages for _, _, page, pages in extra)

        def fp(*a, **k):
            return orig_fp(*a, **k) + extra_bytes

        def cpd(plan, *a, **k):
            desc = orig_cpd(plan, *a, **k)
            cores = plan.shard_grid if plan.sharded else pd._used_core_range_set(len(plan.work), plan.grid_x)
            cbs = list(desc.cbs)
            for cb_id, dt, page, pages in extra:
                cbs.append(pd._cb(page * pages, cores, [(cb_id, dt, page)]))
            desc.cbs = cbs
            return desc

        pd.footprint_bytes, pd.create_program_descriptor = fp, cpd
    try:
        _run_case(device, case, variant)
    finally:
        pd.footprint_bytes, pd.create_program_descriptor = orig_fp, orig_cpd


def _run_case(device, case, variant):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports
    from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import (
        block_shard_memory_config,
        pytorch_groupnorm_sc_N_1_HW_C,
    )

    shape, G, grid, layout, in_place, eps = ALL_CASES[case]
    torch.manual_seed(0)
    C = shape[-1]
    xt = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    gt = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    bt = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    ref = pytorch_groupnorm_sc_N_1_HW_C(xt, G, gamma=gt, beta=bt, eps=eps)

    if grid is None:
        mem = ttnn.DRAM_MEMORY_CONFIG
    else:
        mem = block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)
    g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=RM, device=device)
    b = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=RM, device=device)
    out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps, in_place=in_place)
    ttnn.synchronize_device(device)
    y = ttnn.to_torch(out)
    if in_place:
        ttnn.deallocate(out, force=True)
    else:
        out.deallocate()
        x.deallocate()
    pcc, rms = _metrics(y, ref)
    print(f"\nPTC_RESULT case={case} variant={variant} pcc={pcc:.6f} rms={rms:.5f}")
    if not variant.startswith("abl"):
        assert pcc >= 0.999 and rms <= 0.01, f"{case}/{variant}: pcc {pcc} rms {rms}"
