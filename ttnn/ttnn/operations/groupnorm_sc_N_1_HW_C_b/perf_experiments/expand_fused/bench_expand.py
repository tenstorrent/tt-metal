# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""expand_fused bake-off: the whole groupnorm_sc_N_1_HW_C_b op with its kernels taken from a variant dir.

Not collected by default (run it by path). One op dispatch per (case, variant); correctness of every dispatch
is checked against the fp32 torch reference (PCC / relative RMS, eval.metrics) and appended to results.jsonl
in this dir (one line per dispatch, in dispatch order = ops_perf_results row order).

    EXPF_VARIANTS=v0,v1 EXPF_CASES=unet_rm_ip_1024x1280,vae_tile_16384x512 \
        scripts/run_safe_pytest.sh --profile --no-py-zones \
        ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/expand_fused/bench_expand.py

Env:
  EXPF_VARIANTS  comma list of kernel dirs under this dir (default v0); dispatched in the given order per case
  EXPF_CASES     comma list of case names from CASES (default: the three focus shapes)
  EXPF_AFFINE    comma list of gamma_beta (default) | gamma_only | no_affine
  EXPF_OFFSET    comma list of input mean offsets (default 0.0): offset-heavy precision probe
  EXPF_TAG       free-form tag stored in results.jsonl
"""

import json
import os
import time
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as gn_pd
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.jsonl"

G8, G48 = (8, 8), (4, 8)
CASES = {
    # name: (shape, G, grid or None (DRAM), layout, in_place, eps)
    "unet_rm_ip_1024x1280": ((1, 1, 1024, 1280), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5),
    "unet_rm_ip_16384x320": ((1, 1, 16384, 320), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5),
    "unet_rm_ip_4096x320": ((1, 1, 4096, 320), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5),
    "unet_rm_ip_4096x640": ((1, 1, 4096, 640), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5),
    "unet_rm_ip_1024x640": ((1, 1, 1024, 640), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5),
    "refiner_rm_ip_256x1536": ((1, 1, 256, 1536), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5),
    "refiner_rm_ip_16384x384": ((1, 1, 16384, 384), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5),
    "unet_rm_ip_16384x640": ((1, 1, 16384, 640), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5),
    "unet_rm_ip_16384x960": ((1, 1, 16384, 960), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5),
    "unet_tile_1024x1280": ((1, 1, 1024, 1280), 32, G8, ttnn.TILE_LAYOUT, False, 1e-6),
    "unet_tile_4096x640_4x8": ((1, 1, 4096, 640), 32, G48, ttnn.TILE_LAYOUT, False, 1e-6),
    "vae_tile_16384x512": ((1, 1, 16384, 512), 32, G8, ttnn.TILE_LAYOUT, False, 1e-6),
    "flux1_dram_16384x256": ((1, 1, 16384, 256), 16, None, ttnn.TILE_LAYOUT, False, 1e-6),
    "vae_dram_65536x512": ((1, 1, 65536, 512), 32, None, ttnn.TILE_LAYOUT, False, 1e-6),
    "mochi_dram_8x407040x128": ((8, 1, 407040, 128), 32, None, ttnn.TILE_LAYOUT, False, 1e-6),
    "mid_4096x640_rm": ((1, 1, 4096, 640), 32, None, ttnn.ROW_MAJOR_LAYOUT, False, 1e-5),
    # small DRAM shapes for correctness of odd W / G_core = 32 / multi-block
    "dram_g32_256x128": ((2, 1, 256, 128), 32, None, ttnn.TILE_LAYOUT, False, 1e-5),
    "dram_odd_512x480": ((1, 1, 512, 480), 8, None, ttnn.TILE_LAYOUT, False, 1e-5),
}
FOCUS = ["unet_rm_ip_1024x1280", "vae_tile_16384x512", "unet_rm_ip_16384x960"]

_variants = [v for v in os.environ.get("EXPF_VARIANTS", "v0").split(",") if v]
_cases = [c for c in os.environ.get("EXPF_CASES", ",".join(FOCUS)).split(",") if c]
_affines = [a for a in os.environ.get("EXPF_AFFINE", "gamma_beta").split(",") if a]
_offsets = [float(o) for o in os.environ.get("EXPF_OFFSET", "0").split(",") if o]
_tag = os.environ.get("EXPF_TAG", "")

PARAMS = [(c, v, a, o) for a in _affines for o in _offsets for c in _cases for v in _variants]

_ORIG_CPD = gn_pd.create_program_descriptor


def _mark_kernels(desc, mark):
    """EXPF_MARK=1: a unique define per (variant, case, kernel) so its JIT cache dir is identifiable (text sizes)."""
    ks = list(desc.kernels)
    for i, k in enumerate(ks):
        name = Path(k.kernel_source).stem.split("_")[-1]
        if name == "compute":
            name += "_root" if list(k.compile_time_args)[9] else "_leaf"
        k.defines = list(k.defines) + [(f"EXPF_MARK_{mark}_{name}_{i}", "1")]
    desc.kernels = ks
    return desc


def _install_descriptor_patch(kdir, mark=None):
    """A variant dir may carry descriptor_patch.py: patch(desc, plan, input, output, **kw) -> desc (CB layout only)."""
    pf = kdir / "descriptor_patch.py"
    patch = lambda desc, *a, **kw: desc
    if pf.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location(f"expf_patch_{kdir.name}", pf)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        patch = mod.patch
    if not pf.exists() and mark is None:
        gn_pd.create_program_descriptor = _ORIG_CPD
        return

    def cpd(plan, input_tensor, output_tensor, **kw):
        desc = patch(_ORIG_CPD(plan, input_tensor, output_tensor, **kw), plan, input_tensor, output_tensor, **kw)
        return _mark_kernels(desc, mark) if mark is not None else desc

    gn_pd.create_program_descriptor = cpd


@pytest.mark.parametrize("case, variant, _affine, _offset", PARAMS, ids=[f"{c}-{v}-{a}-{o}" for c, v, a, o in PARAMS])
def test_expand_bench(device, case, variant, _affine, _offset):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports
    from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import (
        block_shard_memory_config,
        pytorch_groupnorm_sc_N_1_HW_C,
    )
    from eval.metrics import compute_metrics_torch

    shape, G, grid, layout, in_place, eps = CASES[case]
    kdir = HERE / variant
    assert (kdir / "groupnorm_sc_N_1_HW_C_b_compute.cpp").exists(), kdir
    gn_pd.KERNEL_DIR = kdir
    _install_descriptor_patch(kdir, f"{variant}_{case}_{_affine}" if os.environ.get("EXPF_MARK") else None)

    torch.manual_seed(0)
    C = shape[-1]
    x_t = (torch.randn(shape, dtype=torch.float32) + _offset).to(torch.bfloat16)
    g_t = torch.randn((1, 1, 1, C), dtype=torch.bfloat16) if _affine in ("gamma_beta", "gamma_only") else None
    b_t = torch.randn((1, 1, 1, C), dtype=torch.bfloat16) if _affine == "gamma_beta" else None
    expected = pytorch_groupnorm_sc_N_1_HW_C(x_t, G, gamma=g_t, beta=b_t, eps=eps)

    mem = (
        block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
        if grid is not None
        else ttnn.DRAM_MEMORY_CONFIG
    )
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)
    aff = lambda t: (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device) if t is not None else None
    )
    g, b = aff(g_t), aff(b_t)
    out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps, in_place=in_place)
    ttnn.synchronize_device(device)
    actual = ttnn.to_torch(out)
    m = compute_metrics_torch(actual, expected, readback_dtype=torch.bfloat16)
    rec = dict(
        ts=time.time(),
        tag=_tag,
        case=case,
        variant=variant,
        affine=_affine,
        offset=_offset,
        pcc=m.pcc,
        rms=m.rms,
        max_abs=m.max_abs_diff,
        nan=m.has_nan or m.has_inf,
    )
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(
        f"EXPF {case} {variant} {_affine} off={_offset}: pcc={m.pcc:.6f} rms={m.rms:.5f} max_abs={m.max_abs_diff:.4f}"
    )
    ttnn.deallocate(out, force=True)
    if not in_place:
        x.deallocate()
    # Correctness gate (perf-case bar by default; EXPF_TOL=pcc,rms): a faster wrong answer is disqualified.
    pcc_min, rms_max = (float(t) for t in os.environ.get("EXPF_TOL", "0.999,0.01").split(","))
    assert not rec["nan"], rec
    assert m.pcc >= pcc_min and m.rms <= rms_max, rec
