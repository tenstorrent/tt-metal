# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated bench of genify_matmul's fused swiglu epilogue (one core, L1-resident Float32 block).

Input: one accumulator block of block_m x block_n Float32 tiles, height-sharded on core (0,0) and
fronted in cb_partials (zero-copy). Output: block_m x block_n/2 tiles (bf16 or fp32), sharded on the
same core and drained by a NoC-free writer. Nothing in the timed program touches DRAM.

Variants (compute kernel `variant` ct arg) -- see kernels/swiglu_epilogue_compute.cpp:
  baseline (0), fused (1), fast3 (2), fast2 (3), fast3_nr0 (4), floor (5).
"""
from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
TILE = 32
CB_PARTIALS = 0
CB_OUT = 16

VARIANTS = {
    "baseline": 0,
    "fused": 1,
    "fast3": 2,
    "fast2": 3,
    "fast3_nr0": 4,
    "floor": 5,
    "fusedU": 6,
    "fast3U": 7,
    "fast2U": 8,
    "fast3_nr0U": 9,
    "sym3U": 10,
    "sym2U": 11,
    "symh1U": 12,
    "symh2U": 13,
    "spU": 14,
    "spiU": 15,
    "spi2U": 16,
    "spikU": 17,
    "spi2kU": 19,
    "spi4U": 20,
    "grad": 21,
}


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _shard_cfg(h, w):
    return ttnn.create_sharded_memory_config(
        shape=(h, w),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def make_block(block_m, block_n, natural_pairs, seed=0, scale=3.0, specials=True, shift=0.0):
    """Returns (partials fp32 [bm*32, bn*32] in the CB's tile-column order, gate, up) as torch fp32."""
    import torch  # local: the ttnn tree forbids a global torch import

    g = torch.Generator().manual_seed(seed)
    half = block_n // 2
    gate = torch.randn(block_m * TILE, half * TILE, generator=g) * scale + shift
    up = torch.randn(block_m * TILE, half * TILE, generator=g)
    if specials and gate.shape[1] >= 128 and gate.shape[0] >= 38:
        sp = torch.tensor([0.0, -0.0, 1e-4, -1e-4, 50.0, -50.0, 87.5, -87.5, 90.0, -90.0, 200.0, -200.0, 20.0, -20.0])
        gate[0, : sp.numel()] = sp
        gate[37, 100 : 100 + sp.numel()] = sp
    gt = gate.reshape(block_m * TILE, half, TILE)
    ut = up.reshape(block_m * TILE, half, TILE)
    if natural_pairs:
        cols = torch.stack([gt, ut], dim=2).reshape(block_m * TILE, block_n * TILE)
    else:
        cols = torch.cat([gt, ut], dim=1).reshape(block_m * TILE, block_n * TILE)
    return cols.contiguous(), gate, up


def run(
    device,
    partials_host,
    *,
    block_m,
    block_n,
    variant,
    natural_pairs=False,
    out_dtype=ttnn.bfloat16,
    fp32_dest=True,
    block_size=4,
    unpack_to_dest=False,
    silu_fp32=None,
    reps=1,
):
    half = block_n // 2
    in_dtype = ttnn.float32 if fp32_dest else ttnn.bfloat16  # the op's partials dtype follows DEST
    if silu_fp32 is None:
        silu_fp32 = out_dtype == ttnn.float32  # SWIGLU_FAST_SILU policy
    h, w_in, w_out = block_m * TILE, block_n * TILE, half * TILE
    partials = ttnn.from_torch(
        partials_host, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard_cfg(h, w_in)
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([h, w_out]), out_dtype, ttnn.TILE_LAYOUT, device, _shard_cfg(h, w_out)
    )

    if natural_pairs:
        rows, cols, stride, up_base = block_m * half, 1, 2, 1
    else:
        rows, cols, stride, up_base = block_m, half, block_n, half
    compute_ct = [
        CB_PARTIALS,
        CB_OUT,
        rows,
        cols,
        stride,
        up_base,
        block_size,
        VARIANTS[variant],
        int(silu_fp32),
        block_m * block_n,
    ]
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
    )
    if unpack_to_dest:
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[CB_PARTIALS] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes
    c0 = ttnn.CoreCoord(0, 0)
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "swiglu_epilogue_compute.cpp"),
        core_ranges=_core(),
        compile_time_args=compute_ct,
        runtime_args=[(c0, [reps])],
        config=cfg,
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "swiglu_epilogue_writer.cpp"),
        core_ranges=_core(),
        compile_time_args=[CB_OUT, cols, rows],
        runtime_args=[(c0, [reps])],
        config=ttnn.WriterConfigDescriptor(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_PARTIALS, partials),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
    ]
    desc = ttnn.ProgramDescriptor(kernels=[compute, writer], semaphores=[], cbs=cbs)
    res = ttnn.generic_op([partials, out], desc)
    return res


def metrics(got, ref):
    """PCC, rel-RMS and error in bf16 ULPs of the reference (ULP floor at the smallest normal)."""
    import torch  # local: the ttnn tree forbids a global torch import

    got = got.double().flatten()
    ref = ref.double().flatten()
    pcc = torch.corrcoef(torch.stack([got, ref]))[0, 1].item()
    rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    # ULP stats over |ref| >= 1e-30 (bf16 ULP at |ref|); the near-zero tail (silu of |g| > ~70) is
    # reported as an absolute error instead, where a ULP count is meaningless.
    mask = ref.abs() >= 1e-30
    r, gg = ref[mask], got[mask]
    ulp = torch.pow(2.0, torch.floor(torch.log2(r.abs())) - 7)
    e = (gg - r).abs() / ulp
    return {
        "pcc": pcc,
        "rel_rms": rms,
        "max_ulp": e.max().item(),
        "mean_ulp": e.mean().item(),
        "frac_gt_0p5ulp": (e > 0.5 + 1e-9).double().mean().item(),
        "frac_gt_1ulp": (e > 1.0 + 1e-9).double().mean().item(),
        "argmax_ref": r[e.argmax()].item(),
        "argmax_got": gg[e.argmax()].item(),
        "max_ulp32": (e * 2.0**16).max().item(),  # same error in fp32 ULPs (for fp32 outputs)
        "mean_ulp32": (e * 2.0**16).mean().item(),
        "tiny_max_abs": (got[~mask] - ref[~mask]).abs().max().item() if (~mask).any() else 0.0,
    }
