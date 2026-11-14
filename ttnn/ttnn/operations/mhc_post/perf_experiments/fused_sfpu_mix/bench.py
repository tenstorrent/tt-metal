# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-core isolated bench of mhc_post's mix_block (R2, packed coefficients, n = 4).

Variants (CT arg `variant` of kernels/mix_bench_compute.cpp):
  0 baseline  — the op's mix_units_j<B> chain (j-major, column-broadcast coefficient, operand re-unpacked per j)
  1 fused     — operand-outer fused SFPU sweep, one window per unit, 4-coefficient K pages

All inputs / outputs are L1-sharded on core (0,0) so the bench measures compute only.
"""

from pathlib import Path

import ttnn

TILE = 32
N = 4
KERNEL = Path(__file__).parent / "kernels" / "mix_bench_compute.cpp"

CB_F, CB_X, CB_COEF, CB_OUT, CB_BF, CB_SQ, CB_DSQ = 0, 1, 2, 3, 4, 5, 6
VARIANTS = {"baseline": 0, "fused": 1}
# name -> (variant, ablate bits)
CONFIGS = {
    "baseline": (0, 0),
    "fused": (1, 0),  # K page in one slot
    "fused_kdb": (2, 0),  # K pages double-buffered (D5 / D7)
    "fused_kdb_il": (3, 0),  # + two accumulators in flight
    "fused_il": (4, 0),  # two accumulators in flight, K page always in D5 (the ported form)
    # ablations of fused_kdb_il (bits: 1 = no SFPU, 2 = no operand copies, 4 = no K copies)
    "abl_nosfpu": (3, 1),
    "abl_noopcopy": (3, 2),
    "abl_nokcopy": (3, 4),
    "abl_nocopy": (3, 6),
    "abl_packonly": (3, 7),
}


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _mem(w_tiles):
    return ttnn.create_sharded_memory_config(
        shape=(TILE, TILE * w_tiles),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dev(t, dtype, device):
    import torch

    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mem(t.shape[1] // TILE)
    )


def jcol():
    import torch

    """K page column -> output stream j: j = 2*(C >= 16) + (C & 1)."""
    c = torch.arange(TILE)
    return 2 * (c >= 16).long() + (c & 1)


def make_inputs(units, seed=0, f_dtype=None):
    import torch

    f_dtype = torch.bfloat16 if f_dtype is None else f_dtype
    g = torch.Generator().manual_seed(seed)
    F = torch.randn(TILE, TILE * units, generator=g).to(f_dtype).float()
    X = torch.randn(TILE, TILE * units * N, generator=g).to(f_dtype).float()  # tile u*N+i = X_i of unit u
    post = (torch.rand(N, TILE, generator=g) * 2.0).float()  # post[j][token]
    comb = torch.rand(N, N, TILE, generator=g).float()  # comb[i][j][token]
    return F, X, post, comb


def coef_pages_baseline(post, comb):
    import torch

    """24 fp32 pages, coefficient k of the 32 tokens in column 0: [post_0..3 | comb[i][j] at 4+4i+j | P(0)]."""
    pages = torch.zeros(TILE, TILE * (N + N * N + N))
    for j in range(N):
        pages[:, TILE * j] = post[j]
    for i in range(N):
        for j in range(N):
            pages[:, TILE * (N + i * N + j)] = comb[i][j]
    return pages


def coef_pages_fused(post, comb):
    import torch

    """5 fp32 K pages: page s (F, X0..X3), element (R, C) = coef(s, j(C))[R]."""
    jc = jcol()
    pages = torch.zeros(TILE, TILE * (1 + N))
    pages[:, 0:TILE] = post[jc].T  # [R, C] = post[jc[C]][R]
    for i in range(N):
        pages[:, TILE * (1 + i) : TILE * (2 + i)] = comb[i][jc].T
    return pages


def golden(F, X, post, comb, units, block_cols):
    import torch

    """fp32 reference with the baseline's exact per-element op order (fma emulated in fp64 then rounded)."""
    A = torch.zeros(TILE, TILE * units * N, dtype=torch.float64)  # j-major per pass: page p*16 + j*B + u
    dsq = torch.zeros(TILE, TILE, dtype=torch.float32)
    passes = units // block_cols
    for p in range(passes):
        acc = {}
        for j in range(N):
            for u in range(block_cols):
                g_ = p * block_cols + u
                f = F[:, g_ * TILE : (g_ + 1) * TILE]
                a = (post[j][:, None] * f).float()
                for i in range(N):
                    x = X[:, (g_ * N + i) * TILE : (g_ * N + i + 1) * TILE]
                    a = (comb[i][j][:, None].double() * x.double() + a.double()).float()
                acc[(j, u)] = a
                k = p * N * block_cols + j * block_cols + u
                A[:, k * TILE : (k + 1) * TILE] = a.double()
        for j in range(N):
            for u in range(block_cols):
                a = acc[(j, u)]
                dsq = (a.double() * a.double() + dsq.double()).float()
    return A.float(), dsq


def run(device, variant, F, X, post, comb, units, block_cols=4, unpack_coef_fp32=True, has_w=True, fp32_io=False):
    import torch

    v, ablate = CONFIGS[variant]
    passes = units // block_cols
    coef = coef_pages_baseline(post, comb) if v == 0 else coef_pages_fused(post, comb)
    n_coef_pages = coef.shape[1] // TILE
    io_t, io_dt = (torch.float32, ttnn.float32) if fp32_io else (torch.bfloat16, ttnn.bfloat16)
    f_dev = _dev(F.to(io_t), io_dt, device)
    x_dev = _dev(X.to(io_t), io_dt, device)
    c_dev = _dev(coef, ttnn.float32, device)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE * units * N]), io_dt, ttnn.TILE_LAYOUT, device, _mem(units * N)
    )
    bf = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE * units * N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, _mem(units * N)
    )
    dsq = ttnn.allocate_tensor_on_device(ttnn.Shape([TILE, TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, _mem(1))

    f32_page = TILE * TILE * 4
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_F, f_dev),
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x_dev),
        ttnn.cb_descriptor_from_sharded_tensor(CB_COEF, c_dev),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
        ttnn.cb_descriptor_from_sharded_tensor(CB_BF, bf),
        ttnn.CBDescriptor(
            total_size=f32_page,
            core_ranges=_core(),
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SQ, data_format=ttnn.float32, page_size=f32_page)
            ],
        ),
        ttnn.cb_descriptor_from_sharded_tensor(CB_DSQ, dsq),
    ]
    unpack = [ttnn.UnpackToDestMode.Default] * 64
    if unpack_coef_fp32:
        unpack[CB_COEF] = ttnn.UnpackToDestMode.UnpackToDestFp32
    unpack[CB_SQ] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if fp32_io:
        unpack[CB_F] = ttnn.UnpackToDestMode.UnpackToDestFp32
        unpack[CB_X] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
        dst_full_sync_en=True,
    )
    cfg.unpack_to_dest_mode = unpack
    ct = [CB_F, CB_X, CB_COEF, CB_OUT, CB_BF, CB_SQ, CB_DSQ, passes, v, n_coef_pages, block_cols, ablate, int(has_w)]
    k = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=ct,
        runtime_args=[],
        config=cfg,
    )
    desc = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs)
    ttnn.generic_op([f_dev, x_dev, c_dev, out, bf, dsq], desc)
    return out, bf, dsq
