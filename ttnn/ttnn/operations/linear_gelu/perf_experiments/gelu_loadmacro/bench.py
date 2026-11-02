# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated pack-thread tanh-GELU bench (single core, fp32 DEST, HiFi2, approx off).

Input: num_tiles fp32 tiles, height-sharded in L1 (UnpackToDestFp32, lossless into DEST).
Output: bf16 (the op's output format; packer converts) or fp32 (raw SFPU result, for bias analysis).
"""
from pathlib import Path

import ttnn

KERNEL = Path(__file__).parent / "kernels" / "bench_compute.cpp"
TILE = 32
CB_IN, CB_OUT = 0, 16


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _mem(num_tiles):
    return ttnn.create_sharded_memory_config(
        shape=(num_tiles * TILE, TILE),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def run(device, x, *, variant, reps=1, out_dtype=ttnn.bfloat16):
    """x: torch [num_tiles*32, 32] float32. Returns torch float64 output."""
    import torch

    num_tiles = x.shape[0] // TILE
    xd = ttnn.from_torch(
        x.to(torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mem(num_tiles)
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([num_tiles * TILE, TILE]), out_dtype, ttnn.TILE_LAYOUT, device, _mem(num_tiles)
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[CB_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    kern = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_core(),
        compile_time_args=[variant],
        runtime_args=[(ttnn.CoreCoord(0, 0), [num_tiles, reps])],
        config=cfg,
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_IN, xd), ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out)]
    desc = ttnn.ProgramDescriptor(kernels=[kern], semaphores=[], cbs=cbs)
    res = ttnn.generic_op([xd, out], desc)
    return ttnn.to_torch(res).to(torch.float64)


def gelu_ref(x):
    import torch

    x = x.to(torch.float64)
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def sweep_input(num_tiles=64, seed=0):
    """Dense [-12, 12] ramp (fp32) for half the tiles, N(0, 3) for the rest."""
    import torch

    n = num_tiles * TILE * TILE
    g = torch.Generator().manual_seed(seed)
    ramp = torch.linspace(-12.0, 12.0, n // 2, dtype=torch.float64)
    rnd = torch.randn(n - n // 2, generator=g, dtype=torch.float64) * 3.0
    return torch.cat([ramp, rnd]).to(torch.float32).reshape(num_tiles * TILE, TILE)


def bf16_ulp_err(got, ref):
    """|got - round_bf16(ref)| in bf16 ULPs of the reference (got is bf16-representable)."""
    import torch

    ref_bf = ref.to(torch.float32).to(torch.bfloat16).to(torch.float64)
    a = ref_bf.abs().clamp_min(1e-38)
    ulp = torch.pow(2.0, torch.floor(torch.log2(a)) - 7)
    return (got - ref_bf).abs() / ulp


def precision_report(x, got, out_is_bf16):
    import torch

    ref = gelu_ref(x)
    mask = ref.abs() >= 1e-3
    err = got - ref
    rep = {
        "max_abs": err.abs().max().item(),
        "mean_signed_rel": (err[mask] / ref[mask].abs()).mean().item(),
        "mean_signed_abs": err.mean().item(),
    }
    if out_is_bf16:
        u = bf16_ulp_err(got, ref)
        rep["max_ulp_masked"] = u[mask].max().item()
        rep["frac_ulp_ge1"] = (u[mask] >= 1).double().mean().item()
    else:
        rel = err[mask].abs() / ref[mask].abs()
        rep["max_rel_masked"] = rel.max().item()
        # the bf16 the packer would produce under RNE, as ULP vs torch
        got_bf = got.to(torch.float32).to(torch.bfloat16).to(torch.float64)
        u = bf16_ulp_err(got_bf, ref)
        rep["rne_max_ulp_masked"] = u[mask].max().item()
    return rep
