# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the RMSNorm blocks (kernels/norm/*): the distributed RMSNorm pieces (rms_norm_pre_all_gather /
post_all_gather) spread over the cores of one device. Each row is split into P parts; a core owns one part of a run
of rows. Stats writes each part's sum(x**2) tile into a small L1 stats tensor, a barrier makes them visible, then
Apply reads the row's P partial tiles, reduces them to E[x**2] and writes x * rsqrt(E[x**2] + eps)."""

from __future__ import annotations

import struct

import ttnn

from .compose import Block, LegoProgram
from .mm import KDIR, tile_bytes
from .sync import place_barrier


def float_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def plan_parts(Ht: int, Wt: int, ncores: int) -> int:
    """Parts per row minimizing the tiles one core handles (ties: fewer parts)."""
    best = None
    for P in [d for d in range(1, Wt + 1) if Wt % d == 0]:
        cpp = ncores // P
        if cpp == 0:
            break
        load = -(-Ht // cpp) * (Wt // P)
        if best is None or load < best[0]:
            best = (load, P)
    return best[1]


def stats_shape(src: ttnn.Tensor, cores: list):
    W = src.padded_shape[-1]
    H = 1
    for d in list(src.padded_shape)[:-1]:
        H *= d
    P = plan_parts(H // 32, W // 32, len(cores))
    return [1, 1, H, 32 * P]


def place_rmsnorm(
    prog: LegoProgram,
    device,
    cores: list,
    *,
    src: ttnn.Tensor,
    dst: ttnn.Tensor,
    stats: ttnn.Tensor,
    eps: float,
    fp32_dest_acc_en: bool = True,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    hw_startup: bool = True,
    label="norm",
):
    """Normalize every row of src [.., H, W] into dst. `stats` is an L1 float32 tensor of stats_shape(src, cores).
    Returns the cores that got work."""
    W = src.padded_shape[-1]
    Wt = W // 32
    Ht = stats.padded_shape[-2] // 32
    P = stats.padded_shape[-1] // 32
    Wp = Wt // P
    cpp = len(cores) // P
    rows_per = -(-Ht // cpp)
    work = {}
    for p in range(P):
        for k in range(cpp):
            r0 = k * rows_per
            nr = min(rows_per, Ht - r0)
            if nr > 0:
                work[cores[p * cpp + k]] = (p, r0, nr)
    used = list(work)

    in_df = src.dtype
    acc_df = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    blk = 4 if fp32_dest_acc_en else 8
    while Wp % blk:
        blk //= 2
    cb_s_inp = prog.cb(used, Wp, tile_bytes(in_df), in_df)
    cb_s_reduce = prog.cb(used, 1, tile_bytes(acc_df), acc_df)
    cb_x2 = prog.cb(used, Wp, tile_bytes(acc_df), acc_df)
    cb_s_out = prog.cb(used, 2, tile_bytes(acc_df), acc_df)
    cb_a_inp = prog.cb(used, 2 * blk, tile_bytes(in_df), in_df)
    cb_stats = prog.cb(used, P, tile_bytes(acc_df), acc_df)
    cb_a_reduce = prog.cb(used, 1, tile_bytes(acc_df), acc_df)
    cb_eps = prog.cb(used, 1, tile_bytes(ttnn.bfloat16), ttnn.bfloat16)
    cb_var = prog.cb(used, 1, tile_bytes(acc_df), acc_df)
    cb_recip = prog.cb(used, 1, tile_bytes(acc_df), acc_df)
    cb_out = prog.cb(used, 2 * blk, tile_bytes(dst.dtype), dst.dtype)

    compute = dict(math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_dest_acc_en, math_approx_mode=False)
    p_of = lambda c: work[c][0]  # noqa: E731
    r0_of = lambda c: work[c][1]  # noqa: E731
    nr_of = lambda c: work[c][2]  # noqa: E731

    # dm1: stream the input for Stats, later drain Apply's output.  dm0: write the partial stats, barrier, then
    # stream the stats and the input for Apply.
    prog.place(
        used,
        Block(
            cls="lego::norm::StatsReader",
            header=f"{KDIR}/norm/rms_dataflow.hpp",
            risc="dm1",
            ct=dict(cb_inp=cb_s_inp, cb_reduce=cb_s_reduce, Wt=Wp, row_stride=Wt),
            rt=lambda c: [src.buffer_address(), nr_of(c), r0_of(c) * Wt + p_of(c) * Wp],
            tensors={"src": src},
            label=f"{label}_sread",
        ),
    )
    prog.place(
        used,
        Block(
            cls="lego::norm::Stats",
            header=f"{KDIR}/norm/rms_compute.hpp",
            risc="compute",
            ct=dict(
                cb_inp=cb_s_inp,
                cb_reduce=cb_s_reduce,
                cb_x2=cb_x2,
                cb_out=cb_s_out,
                Wt=Wp,
                unpack_fp32_active=False,
                hw_startup=hw_startup,
            ),
            rt=lambda c: [nr_of(c)],
            compute=compute,
            label=f"{label}_stats",
        ),
    )
    prog.place(
        used,
        Block(
            cls="lego::common::TileWriter",
            header=f"{KDIR}/common/tile_writer.hpp",
            risc="dm0",
            ct=dict(cb_out=cb_s_out, blk=1, row_tiles=1, row_stride=P),
            rt=lambda c: [stats.buffer_address(), nr_of(c), r0_of(c) * P + p_of(c)],
            tensors={"dst": stats},
            label=f"{label}_swrite",
        ),
    )
    place_barrier(prog, device, used, risc="dm0", label=f"{label}_bar")
    prog.place(
        used,
        Block(
            cls="lego::norm::ApplyReader",
            header=f"{KDIR}/norm/rms_dataflow.hpp",
            risc="dm0",
            ct=dict(
                cb_inp=cb_a_inp,
                cb_stats=cb_stats,
                cb_reduce=cb_a_reduce,
                cb_eps=cb_eps,
                Wt=Wp,
                row_stride=Wt,
                stats_tiles_cols=P,
                stats_row_stride=P,
                reduce_factor=W,
                stats_from_cb=False,
            ),
            rt=lambda c: [
                src.buffer_address(),
                stats.buffer_address(),
                nr_of(c),
                r0_of(c) * Wt + p_of(c) * Wp,
                r0_of(c) * P,
                float_bits(eps),
            ],
            tensors={"src": src, "stats": stats},
            label=f"{label}_aread",
        ),
    )
    prog.place(
        used,
        Block(
            cls="lego::norm::Apply",
            header=f"{KDIR}/norm/rms_compute.hpp",
            risc="compute",
            ct=dict(
                cb_inp=cb_a_inp,
                cb_stats=cb_stats,
                cb_reduce=cb_a_reduce,
                cb_eps=cb_eps,
                cb_var=cb_var,
                cb_recip=cb_recip,
                cb_out=cb_out,
                Wt=Wp,
                blk=blk,
                stats_tiles_cols=P,
                hw_startup=False,
            ),
            rt=lambda c: [nr_of(c)],
            compute=compute,
            label=f"{label}_apply",
        ),
    )
    prog.place(
        used,
        Block(
            cls="lego::common::TileWriter",
            header=f"{KDIR}/common/tile_writer.hpp",
            risc="dm1",
            ct=dict(cb_out=cb_out, blk=blk, row_tiles=Wp, row_stride=Wt),
            rt=lambda c: [dst.buffer_address(), nr_of(c) * Wp, r0_of(c) * Wt + p_of(c) * Wp],
            tensors={"dst": dst},
            label=f"{label}_write",
        ),
    )
    return used


def _split_rows_parts(Ht: int, P: int, cores: list):
    """core -> (part, first row, rows): each core owns one part of a run of rows."""
    cpp = max(1, len(cores) // P)
    rows_per = -(-Ht // cpp)
    work = {}
    for p in range(P):
        for k in range(cpp):
            r0 = k * rows_per
            nr = min(rows_per, Ht - r0)
            if nr > 0 and p * cpp + k < len(cores):
                work[cores[p * cpp + k]] = (p, r0, nr)
    return work


def place_rms_stats(
    prog: LegoProgram,
    cores: list,
    *,
    src: ttnn.Tensor,
    stats: ttnn.Tensor,
    fp32_dest_acc_en=True,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    hw_startup=False,
    reader="dm0",
    writer="dm1",
    label="rstat",
    dst_full_sync_en=False,
):
    """rms_norm_pre_all_gather on this device's shard: stats[(row, p)] = sum over part p of row of x**2, for src
    [.., H, W] and stats [.., H, 32 * P] (P = parts per row). Returns the cores that got work."""
    W = src.padded_shape[-1]
    Wt = W // 32
    Ht = stats.padded_shape[-2] // 32
    P = stats.padded_shape[-1] // 32
    Wp = Wt // P
    work = _split_rows_parts(Ht, P, cores)
    used = list(work)
    acc_df = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    cb_inp = prog.cb(used, Wp, tile_bytes(src.dtype), src.dtype)
    cb_reduce = prog.cb(used, 1, tile_bytes(acc_df), acc_df)
    cb_x2 = prog.cb(used, Wp, tile_bytes(acc_df), acc_df)
    cb_out = prog.cb(used, 2, tile_bytes(stats.dtype), stats.dtype)
    compute = dict(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=False,
        dst_full_sync_en=dst_full_sync_en,
    )
    prog.place(
        used,
        Block(
            cls="lego::norm::StatsReader",
            header=f"{KDIR}/norm/rms_dataflow.hpp",
            risc=reader,
            ct=dict(cb_inp=cb_inp, cb_reduce=cb_reduce, Wt=Wp, row_stride=Wt),
            rt=lambda c: [src.buffer_address(), work[c][2], work[c][1] * Wt + work[c][0] * Wp],
            tensors={"src": src},
            label=f"{label}_read",
        ),
    )
    prog.place(
        used,
        Block(
            cls="lego::norm::Stats",
            header=f"{KDIR}/norm/rms_compute.hpp",
            risc="compute",
            ct=dict(
                cb_inp=cb_inp,
                cb_reduce=cb_reduce,
                cb_x2=cb_x2,
                cb_out=cb_out,
                Wt=Wp,
                unpack_fp32_active=False,
                hw_startup=hw_startup,
            ),
            rt=lambda c: [work[c][2]],
            compute=compute,
            label=f"{label}_stats",
        ),
    )
    prog.place(
        used,
        Block(
            cls="lego::common::TileWriter",
            header=f"{KDIR}/common/tile_writer.hpp",
            risc=writer,
            ct=dict(cb_out=cb_out, blk=1, row_tiles=1, row_stride=P),
            rt=lambda c: [stats.buffer_address(), work[c][2], work[c][1] * P + work[c][0]],
            tensors={"dst": stats},
            label=f"{label}_write",
        ),
    )
    return used


def place_rms_scale(
    prog: LegoProgram,
    cores: list,
    *,
    stats: ttnn.Tensor,
    ones: ttnn.Tensor,
    scale: ttnn.Tensor,
    width: int,
    eps: float,
    fp32_dest_acc_en=True,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    hw_startup=False,
    reader="dm0",
    writer="dm1",
    label="rscale",
    dst_full_sync_en=False,
):
    """rms_norm_post_all_gather's statistics half: per row tile, reduce the row's partial sums of squares (all
    stats.shape[-1] / 32 tiles: every device's parts) to E[x**2] over `width` elements, then 1/sqrt(E + eps), written
    as a full tile (every column = the row's scale) into scale [.., H, 32]. It is Apply on a tile of ones: 1 * rsqrt
    broadcast along columns. Returns the cores that got work."""
    Ht = stats.padded_shape[-2] // 32
    C = stats.padded_shape[-1] // 32
    work = _split_rows_parts(Ht, 1, cores)
    used = list(work)
    acc_df = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    cb_inp = prog.cb(used, 2, tile_bytes(ones.dtype), ones.dtype)
    cb_stats = prog.cb(used, C, tile_bytes(stats.dtype), stats.dtype)
    cb_reduce = prog.cb(used, 1, tile_bytes(acc_df), acc_df)
    cb_eps = prog.cb(used, 1, tile_bytes(ttnn.bfloat16), ttnn.bfloat16)
    cb_var = prog.cb(used, 1, tile_bytes(acc_df), acc_df)
    cb_recip = prog.cb(used, 1, tile_bytes(acc_df), acc_df)
    cb_out = prog.cb(used, 2, tile_bytes(scale.dtype), scale.dtype)
    compute = dict(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=False,
        dst_full_sync_en=dst_full_sync_en,
    )
    prog.place(
        used,
        Block(
            cls="lego::norm::ApplyReader",
            header=f"{KDIR}/norm/rms_dataflow.hpp",
            risc=reader,
            ct=dict(
                cb_inp=cb_inp,
                cb_stats=cb_stats,
                cb_reduce=cb_reduce,
                cb_eps=cb_eps,
                Wt=1,
                row_stride=0,
                stats_tiles_cols=C,
                stats_row_stride=C,
                reduce_factor=width,
                stats_from_cb=False,
            ),
            rt=lambda c: [
                ones.buffer_address(),
                stats.buffer_address(),
                work[c][2],
                0,
                work[c][1] * C,
                float_bits(eps),
            ],
            tensors={"src": ones, "stats": stats},
            label=f"{label}_read",
        ),
    )
    prog.place(
        used,
        Block(
            cls="lego::norm::Apply",
            header=f"{KDIR}/norm/rms_compute.hpp",
            risc="compute",
            ct=dict(
                cb_inp=cb_inp,
                cb_stats=cb_stats,
                cb_reduce=cb_reduce,
                cb_eps=cb_eps,
                cb_var=cb_var,
                cb_recip=cb_recip,
                cb_out=cb_out,
                Wt=1,
                blk=1,
                stats_tiles_cols=C,
                hw_startup=hw_startup,
            ),
            rt=lambda c: [work[c][2]],
            compute=compute,
            label=f"{label}_apply",
        ),
    )
    prog.place(
        used,
        Block(
            cls="lego::common::TileWriter",
            header=f"{KDIR}/common/tile_writer.hpp",
            risc=writer,
            ct=dict(cb_out=cb_out, blk=1, row_tiles=1, row_stride=1),
            rt=lambda c: [scale.buffer_address(), work[c][2], work[c][1]],
            tensors={"dst": scale},
            label=f"{label}_write",
        ),
    )
    return used
