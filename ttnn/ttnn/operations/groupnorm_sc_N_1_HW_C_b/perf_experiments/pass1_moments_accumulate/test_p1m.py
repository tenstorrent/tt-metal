# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pass1_moments_accumulate bake-off harness (perf_experiments only; not part of the op's test suite).

The variant under test is selected by GN_KERNEL_DIR (the program descriptor's kernel-source-dir override):
    GN_KERNEL_DIR=<abs>/kernels_baseline   or   <abs>/kernels_c1   ...

test_precision : full op on a case x input distribution; prints one P1M_PREC line with
                 PCC / rel-RMS vs an fp64 reference, and the worst per-group scale error (gamma=1, beta=0
                 runs only: the output's per-group slope vs the reference = the rstd error).
                 Correctness gate: PCC >= 0.999 and rel-RMS <= 0.01 vs fp64 for ordinary / mean-5 / mean-10 inputs.
test_bench     : one dispatch per case (run under --profile --no-py-zones; DEVICE KERNEL DURATION per op row).
"""

import os

import pytest
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C_b import TUNING_OVERRIDES, groupnorm_sc_N_1_HW_C_b

G8 = (8, 8)

# name, shape, G, placement (grid or None = DRAM), layout, in_place, eps, tuning overrides
CASES = {
    # focus
    "unet_rm_ip_16384x960": ((1, 1, 16384, 960), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5, {}),
    "vae_tile_16384x512": ((1, 1, 16384, 512), 32, G8, ttnn.TILE_LAYOUT, False, 1e-6, {}),
    "unet_rm_ip_16384x320": ((1, 1, 16384, 320), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5, {}),
    # sharded sweep
    "unet_rm_ip_1024x1280": ((1, 1, 1024, 1280), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5, {}),
    "unet_rm_ip_4096x320": ((1, 1, 4096, 320), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5, {}),
    "unet_rm_ip_4096x640": ((1, 1, 4096, 640), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5, {}),
    "unet_rm_ip_1024x640": ((1, 1, 1024, 640), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5, {}),
    "refiner_rm_ip_256x1536": ((1, 1, 256, 1536), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5, {}),
    "refiner_rm_ip_16384x384": ((1, 1, 16384, 384), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5, {}),
    "unet_rm_ip_16384x640": ((1, 1, 16384, 640), 32, G8, ttnn.ROW_MAJOR_LAYOUT, True, 1e-5, {}),
    "unet_tile_1024x1280": ((1, 1, 1024, 1280), 32, G8, ttnn.TILE_LAYOUT, False, 1e-6, {}),
    "unet_tile_4096x640_4x8": ((1, 1, 4096, 640), 32, (4, 8), ttnn.TILE_LAYOUT, False, 1e-6, {}),
    # DRAM sweep
    "vae_dram_262144x256": ((1, 1, 262144, 256), 32, None, ttnn.TILE_LAYOUT, False, 1e-6, {}),
    "flux1_dram_1048576x64": ((1, 1, 1048576, 64), 16, None, ttnn.TILE_LAYOUT, False, 1e-6, {}),
    # precision-only coverage of the per-chunk (spill / reload) paths
    "dram_stream_65536x128": (
        (1, 1, 65536, 128),
        32,
        None,
        ttnn.TILE_LAYOUT,
        False,
        1e-5,
        {"max_resident_chunks": 1, "chunk_tiles_target": 8},
    ),
    "rm_ip_partial_4096x640": (
        (1, 1, 4096, 640),
        32,
        G8,
        ttnn.ROW_MAJOR_LAYOUT,
        True,
        1e-5,
        {"max_resident_chunks": 1, "chunk_tiles_target": 5},
    ),
    "dram_w1_65536x32": ((1, 1, 65536, 32), 1, None, ttnn.TILE_LAYOUT, False, 1e-6, {}),
}


def _make(dist, shape, G, seed=42):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports

    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.float32)
    if dist == "randn":
        return x.to(torch.bfloat16)
    if dist.startswith("mean"):  # per-element offset: |mean| >> std, the shifted-moment regime
        return (x + float(dist[4:])).to(torch.bfloat16)
    if dist == "gmean":  # per-group means drawn N(0, 8^2), std 1: mixed-sign offsets across groups
        C = shape[-1]
        return (x + 8.0 * torch.randn((1, 1, 1, G)).repeat_interleave(C // G, -1)).to(torch.bfloat16)
    raise ValueError(dist)


def _ref64(x_bf16, G, gamma, beta, eps):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports

    x = x_bf16.to(torch.float64)
    N, _, HW, C = x.shape
    xg = x.reshape(N, HW, G, C // G)
    mean = xg.mean(dim=(1, 3), keepdim=True)
    var = ((xg - mean) ** 2).mean(dim=(1, 3), keepdim=True)
    y = ((xg - mean) / torch.sqrt(var + eps)).reshape(N, 1, HW, C)
    if gamma is not None:
        y = y * gamma.to(torch.float64).reshape(1, 1, 1, C) + beta.to(torch.float64).reshape(1, 1, 1, C)
    return y


def _to_device(x, shape, grid, layout, device):
    if grid is None:
        return ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config

    mem = block_shard_memory_config(shape, grid, layout=layout, dtype=ttnn.bfloat16, device=device)
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)


@pytest.fixture
def tuning():
    saved = dict(TUNING_OVERRIDES)
    yield TUNING_OVERRIDES
    TUNING_OVERRIDES.clear()
    TUNING_OVERRIDES.update(saved)


PREC = [
    (c, d, aff)
    for c in [
        "unet_rm_ip_16384x960",
        "vae_tile_16384x512",
        "unet_rm_ip_16384x320",
        "unet_rm_ip_1024x1280",
        "refiner_rm_ip_256x1536",
        "unet_tile_4096x640_4x8",
        "dram_stream_65536x128",
        "rm_ip_partial_4096x640",
        "dram_w1_65536x32",
    ]
    for d, aff in [("randn", True), ("mean10", False), ("mean5", True), ("mean50", False), ("gmean", False)]
]


SEEDS = [(c, s) for c in ["dram_w1_65536x32", "vae_tile_16384x512", "unet_rm_ip_16384x960"] for s in range(1, 6)]


@pytest.mark.parametrize("case, seed", SEEDS, ids=[f"{c}-s{s}" for c, s in SEEDS])
def test_seed_mean50(device, tuning, case, seed):
    """Offset-heavy (mean 50) seed sweep: is a variant's mean50 delta systematic or the luck of the TF32 collapse?"""
    test_precision(device, tuning, case, "mean50", False, seed=seed)


@pytest.mark.parametrize("case, dist, affine", PREC, ids=[f"{c}-{d}" for c, d, _ in PREC])
def test_precision(device, tuning, case, dist, affine, seed=42):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports

    shape, G, grid, layout, in_place, eps, tov = CASES[case]
    tuning.update(tov)
    x = _make(dist, shape, G, seed)
    C = shape[-1]
    torch.manual_seed(7)
    if affine:
        gamma = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
        beta = torch.randn((1, 1, 1, C), dtype=torch.float32).to(torch.bfloat16)
    else:
        gamma = torch.ones((1, 1, 1, C), dtype=torch.bfloat16)
        beta = torch.zeros((1, 1, 1, C), dtype=torch.bfloat16)
    tx = _to_device(x, shape, grid, layout, device)
    tg = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tb = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = groupnorm_sc_N_1_HW_C_b(tx, G, gamma=tg, beta=tb, eps=eps, in_place=in_place)
    y = ttnn.to_torch(out).to(torch.float64).reshape(shape)
    ttnn.deallocate(out, force=True)
    if not in_place:
        tx.deallocate()
    ref = _ref64(x, G, gamma, beta, eps)
    d = (y - ref).flatten()
    r = ref.flatten()
    rms = (d.norm() / r.norm()).item()
    pcc = torch.corrcoef(torch.stack([y.flatten(), r]))[0, 1].item()
    scale = float("nan")
    if not affine:
        N, _, HW, _ = shape
        yg = y.reshape(N, HW, G, C // G)
        rg = ref.reshape(N, HW, G, C // G)
        ratio = (yg * rg).sum(dim=(1, 3)) / (rg * rg).sum(dim=(1, 3))
        scale = (ratio - 1).abs().max().item()
    ok = pcc >= 0.999 and rms <= 0.01
    print(
        f"\nP1M_PREC {case} {dist}{'' if seed == 42 else f'-s{seed}'} pcc={pcc:.7f} rms={rms:.5f} max_group_scale_err={scale:.2e} ok={ok}"
    )
    if dist in ("randn", "mean5", "mean10"):
        assert ok, f"pcc={pcc} rms={rms}"


BENCH = [
    "unet_rm_ip_16384x960",
    "vae_tile_16384x512",
    "unet_rm_ip_16384x320",
    "unet_rm_ip_1024x1280",
    "unet_rm_ip_4096x320",
    "unet_rm_ip_4096x640",
    "unet_rm_ip_1024x640",
    "refiner_rm_ip_256x1536",
    "refiner_rm_ip_16384x384",
    "unet_rm_ip_16384x640",
    "unet_tile_1024x1280",
    "unet_tile_4096x640_4x8",
    "vae_dram_262144x256",
    "flux1_dram_1048576x64",
]


@pytest.mark.parametrize("case", BENCH)
def test_bench(device, tuning, case):
    import torch  # function-local: ttnn/ttnn forbids module-level torch imports

    shape, G, grid, layout, in_place, eps, tov = CASES[case]
    tuning.update(tov)
    if os.environ.get("GN_BENCH_CHUNK_ZONES"):
        tuning["chunk_zones"] = True
    torch.manual_seed(0)
    C = shape[-1]
    tx = _to_device(torch.randn(shape, dtype=torch.bfloat16), shape, grid, layout, device)
    tg = ttnn.from_torch(
        torch.randn((1, 1, 1, C), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tb = ttnn.from_torch(
        torch.randn((1, 1, 1, C), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    out = groupnorm_sc_N_1_HW_C_b(tx, G, gamma=tg, beta=tb, eps=eps, in_place=in_place)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out, force=True)
    if not in_place:
        tx.deallocate()
