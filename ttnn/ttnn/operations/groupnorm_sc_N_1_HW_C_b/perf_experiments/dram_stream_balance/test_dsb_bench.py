# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""dram_stream_balance bake-off bench: the full op on the 9 DRAM perf cases, one dispatch per case, with an
optional correctness gate (PCC / rel-RMS vs the torch reference).

Variant selection (env):
  DSB_VARIANT=<name>  patch the op's planner / descriptor with variant_descriptor.py's <name> (see VARIANTS there);
                      unset or "baseline" = the op's own descriptor and kernels.
  DSB_CHECK=1         gate on PCC >= 0.999 and rel-RMS <= 0.01 (perf contract) after the dispatch.

Run (one case):
  scripts/run_safe_pytest.sh --profile --no-py-zones \
      ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C_b/perf_experiments/dram_stream_balance/test_dsb_bench.py -k vae_dram_262144x256
"""

import os
import sys

import pytest
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import variant_descriptor  # noqa: E402

CASES = [
    ("vae_dram_1048576x128", (1, 1, 1048576, 128), 32, 1e-6),
    ("vae_dram_65536x512", (1, 1, 65536, 512), 32, 1e-6),
    ("vae_dram_262144x256", (1, 1, 262144, 256), 32, 1e-6),
    ("vae_dram_262144x512", (1, 1, 262144, 512), 32, 1e-6),
    ("vae_dram_1048576x256", (1, 1, 1048576, 256), 32, 1e-6),
    ("refiner_dram_16384x1152", (1, 1, 16384, 1152), 32, 1e-5),
    ("flux1_dram_16384x256", (1, 1, 16384, 256), 16, 1e-6),
    ("flux1_dram_1048576x64", (1, 1, 1048576, 64), 16, 1e-6),
    ("mochi_dram_8x407040x128", (8, 1, 407040, 128), 32, 1e-6),
    # Domain regimes beyond the perf list (same DRAM-interleaved stream, other layout / size)
    ("rm_dram_262144x256", (1, 1, 262144, 256), 32, 1e-6),
    ("rm_dram_16384x1152", (1, 1, 16384, 1152), 32, 1e-5),
]
RM_CASES = {"rm_dram_262144x256", "rm_dram_16384x1152"}

_SEL = [
    c for c in os.environ.get("DSB_CASES", "").split(",") if c
]  # case-name substrings (run_safe_pytest splits -k on spaces)
if _SEL:
    CASES = [c for c in CASES if any(t in c[0] for t in _SEL)]


def _ref(x, G, g, b, eps):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports

    N, _, HW, C = x.shape
    y = torch.nn.functional.group_norm(
        x.float().squeeze(1).permute(0, 2, 1), G, weight=g.float().reshape(C), bias=b.float().reshape(C), eps=eps
    )
    return y.permute(0, 2, 1).unsqueeze(1)


def _metrics(got, ref, chunk=1 << 24):
    """PCC and relative RMS in float64, accumulated over chunks (keeps host memory bounded on the big cases)."""
    got = got.flatten()
    ref = ref.flatten()
    n = got.numel()
    sg = sr = sgg = srr = sgr = sdd = 0.0
    for i in range(0, n, chunk):
        g = got[i : i + chunk].double()
        r = ref[i : i + chunk].double()
        sg += float(g.sum())
        sr += float(r.sum())
        sgg += float((g * g).sum())
        srr += float((r * r).sum())
        sgr += float((g * r).sum())
        sdd += float((g - r).pow(2).sum())
    cov = sgr - sg * sr / n
    vg = sgg - sg * sg / n
    vr = srr - sr * sr / n
    pcc = cov / (vg * vr) ** 0.5
    rms = (sdd / n) ** 0.5 / (srr / n) ** 0.5
    return pcc, rms


VARIANT_LIST = [v for v in os.environ.get("DSB_VARIANT", "baseline").split(",") if v]


@pytest.mark.parametrize("name, shape, G, eps", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("variant", VARIANT_LIST)
def test_dsb(device, variant, name, shape, G, eps):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports
    from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b
    from ttnn.operations.groupnorm_sc_N_1_HW_C_b import TUNING_OVERRIDES

    saved_tuning = dict(TUNING_OVERRIDES)
    if os.environ.get("DSB_CHUNK_ZONES"):
        TUNING_OVERRIDES["chunk_zones"] = True
    restore = variant_descriptor.install(variant)
    try:
        torch.manual_seed(0)
        C = shape[-1]
        xt = torch.randn(shape, dtype=torch.bfloat16)
        gt = torch.randn((1, 1, 1, C), dtype=torch.bfloat16)
        bt = torch.randn((1, 1, 1, C), dtype=torch.bfloat16)
        layout = ttnn.ROW_MAJOR_LAYOUT if name in RM_CASES else ttnn.TILE_LAYOUT
        x = ttnn.from_torch(
            xt, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        b = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps)
        ttnn.synchronize_device(device)
        print(f"DSB_RUN {variant} {name}")  # after the dispatch: labels map 1:1 to profiled ops
        if os.environ.get("DSB_CHECK"):
            got = ttnn.to_torch(out)
            pcc, rms = _metrics(got, _ref(xt, G, gt, bt, eps))
            print(f"DSB_CHECK {name} variant={variant} pcc={pcc:.6f} rms={rms:.5f}")
            assert pcc >= 0.999 and rms <= 0.01, (pcc, rms)
        out.deallocate()
        x.deallocate()
    finally:
        restore()
        TUNING_OVERRIDES.update(saved_tuning)


# L1 BLOCK_SHARDED perf cases (from the op's _op_bench.py): the variant must leave them untouched. Selected with
# DSB_SHARDED=1 (otherwise none run).
SHARDED_CASES = [
    # name, shape, G, grid, layout, eps, in_place
    ("sh_unet_tile_1024x1280", (1, 1, 1024, 1280), 32, (8, 8), ttnn.TILE_LAYOUT, 1e-6, False),
    ("sh_vae_tile_16384x512", (1, 1, 16384, 512), 32, (8, 8), ttnn.TILE_LAYOUT, 1e-6, False),
    ("sh_unet_rm_ip_16384x320", (1, 1, 16384, 320), 32, (8, 8), ttnn.ROW_MAJOR_LAYOUT, 1e-5, True),
]
if not os.environ.get("DSB_SHARDED"):
    SHARDED_CASES = []


@pytest.mark.parametrize(
    "name, shape, G, grid, layout, eps, in_place", SHARDED_CASES, ids=[c[0] for c in SHARDED_CASES]
)
@pytest.mark.parametrize("variant", VARIANT_LIST)
def test_dsb_sharded(device, variant, name, shape, G, grid, layout, eps, in_place):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports
    from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config
    from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

    restore = variant_descriptor.install(variant)
    try:
        torch.manual_seed(0)
        C = shape[-1]
        mem = block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
        xt = torch.randn(shape, dtype=torch.bfloat16)
        gt = torch.randn((1, 1, 1, C), dtype=torch.bfloat16)
        bt = torch.randn((1, 1, 1, C), dtype=torch.bfloat16)
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)
        g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        b = ttnn.from_torch(bt, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        out = groupnorm_sc_N_1_HW_C_b(x, G, gamma=g, beta=b, eps=eps, in_place=in_place)
        ttnn.synchronize_device(device)
        print(f"DSB_RUN {variant} {name}")
        if os.environ.get("DSB_CHECK"):
            pcc, rms = _metrics(ttnn.to_torch(out), _ref(xt, G, gt, bt, eps))
            print(f"DSB_CHECK {name} variant={variant} pcc={pcc:.6f} rms={rms:.5f}")
            assert pcc >= 0.999 and rms <= 0.01, (pcc, rms)
        ttnn.deallocate(out, force=True)
        if not in_place:
            x.deallocate()
    finally:
        restore()
