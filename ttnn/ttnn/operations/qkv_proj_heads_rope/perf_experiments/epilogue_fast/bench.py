# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-core isolated qkv_proj_heads_rope epilogue bench (resident Float32 partials + bf16 table ->
bf16 rows through two alternating 2-row output CBs, as in the op). See epi_compute.cpp for variants."""

from pathlib import Path

import ttnn

KDIR = Path(__file__).parent / "kernels"
TILE = 32
CB_PARTIALS, CB_TABLE, CB_OUT, CB_OUT_NOC0 = 2, 3, 4, 5
VARIANTS = {"baseline": 0, "win4": 1, "win2": 2, "v_baseline": 3, "tilesrc4": 4, "tilesrc_pm": 5}


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _sharded(h, w):
    return ttnn.create_sharded_memory_config(
        shape=(h, w),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def make_host(block_m, heads, seed=0):
    import torch

    torch.manual_seed(seed)
    n = 2 * heads
    partials = torch.randn(block_m * TILE, n * TILE, dtype=torch.float32) * 0.7
    pos = torch.arange(block_m * TILE, dtype=torch.float32)
    inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    f = torch.outer(pos, inv)
    cos_half, sin_half = torch.cos(f).bfloat16(), torch.sin(f).bfloat16()
    cos = torch.cat([cos_half, cos_half], 1)
    sin = torch.cat([sin_half, sin_half], 1)
    table = torch.cat([cos[:, :32], -sin[:, :32], sin[:, 32:], cos[:, 32:]], 1)
    return partials, table, cos, sin


def reference(partials, cos, sin, heads, rotary=True):
    import torch

    p = partials.double()
    if not rotary:
        return p
    c, s = cos.double(), sin.double()
    outs = []
    for h in range(heads):
        p0, p1 = p[:, 64 * h : 64 * h + 32], p[:, 64 * h + 32 : 64 * h + 64]
        outs += [p0 * c[:, :32] - p1 * s[:, :32], p1 * c[:, 32:] + p0 * s[:, 32:]]
    return torch.cat(outs, 1)


def device_tensors(device, partials, table, block_m, heads, partials_dtype=ttnn.float32):
    import torch

    n = 2 * heads
    p = ttnn.from_torch(
        partials,
        dtype=partials_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(block_m * TILE, n * TILE),
    )
    t = ttnn.from_torch(
        table,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_sharded(block_m * TILE, 4 * TILE),
    )
    out = ttnn.from_torch(
        torch.zeros(block_m * TILE, n * TILE),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return p, t, out


def descriptor(p, t, out, *, variant, block_m, heads, iters, seq_tiles=16, window_heads=2, compute_config=None, diag=0):
    n = 2 * heads
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_PARTIALS, p),
        ttnn.cb_descriptor_from_sharded_tensor(CB_TABLE, t),
    ]
    for cb in (CB_OUT, CB_OUT_NOC0):
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * n * bf16_tile,
                core_ranges=_core(),
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.bfloat16, page_size=bf16_tile)
                ],
            )
        )
    out_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args()
    rt_out = ttnn.RuntimeArgs()
    rt_out[0][0] = [out.buffer_address()]

    def drain(cb, first_row, expose, config):
        rt = ttnn.RuntimeArgs()
        rt[0][0] = [out.buffer_address()]
        return ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "epi_drain.cpp"),
            core_ranges=_core(),
            compile_time_args=[
                cb,
                n,
                block_m,
                iters,
                first_row,
                bf16_tile,
                expose,
                CB_PARTIALS,
                block_m * n,
                CB_TABLE,
                block_m * 4,
                *out_ct,
            ],
            runtime_args=rt,
            config=config,
        )

    compute_config = compute_config or ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi3, fp32_dest_acc_en=True, math_approx_mode=False
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "epi_compute.cpp"),
        core_ranges=_core(),
        compile_time_args=[
            CB_PARTIALS,
            CB_TABLE,
            CB_OUT,
            CB_OUT_NOC0,
            block_m,
            heads,
            seq_tiles,
            iters,
            VARIANTS[variant],
            window_heads,
            diag,
        ],
        runtime_args=ttnn.RuntimeArgs(),
        config=compute_config,
        defines=[("EPI_DIAG_B_NOP", str(diag - 2))] if diag in (3, 4) else [],
    )
    kernels = [
        drain(CB_OUT_NOC0, 1, 1, ttnn.ReaderConfigDescriptor()),
        drain(CB_OUT, 0, 0, ttnn.WriterConfigDescriptor()),
        compute,
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
