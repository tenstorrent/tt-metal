# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the matmul blocks (kernels/mm/{in0,in1,compute}.hpp): the core roles, forward chains, block
ranges and CBs that minimal_matmul's program factory computes, for one matmul placed on a rectangle of cores."""

from __future__ import annotations

from dataclasses import dataclass

import ttnn

from .compose import Block, LegoProgram

KDIR = "ttnn/operations/experimental/lego/kernels"
INVALID, VALID = 0, 1

TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.float32: 4096,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
}


def tile_bytes(dtype) -> int:
    return TILE_BYTES[dtype]


def div_up(a, b):
    return -(-a // b)


def round_up(a, b):
    return div_up(a, b) * b


@dataclass(frozen=True)
class MatmulConfig:
    M_block: int
    K_block: int
    N_block: int
    subblock_h: int
    subblock_w: int


@dataclass
class MatmulPlan:
    cores: list
    transpose: bool
    M_tiles: int
    K_tiles: int
    N_tiles: int
    in0_cores: list  # injector cores of in0 (read in0 from its tensor)
    writer_risc: str


def _axis_order(idx_on_axis: int, length: int, increasing: bool):
    """minimal_matmul's build_core_order_for_axis, as indices along the axis: the injector (0) first, then the
    cores in NOC direction. Returns (order, position of idx_on_axis in it)."""
    order = [0] + (list(range(1, length)) if increasing else [length - w for w in range(1, length)])
    return order, order.index(idx_on_axis)


def place_matmul(
    prog: LegoProgram,
    device,
    *,
    origin: tuple[int, int],
    grid: tuple[int, int],
    in0: ttnn.Tensor,
    weight: ttnn.Tensor,
    out: ttnn.Tensor,
    cfg: MatmulConfig,
    fuse_swiglu: bool = False,
    math_fidelity=ttnn.MathFidelity.LoFi,
    fp32_dest_acc_en: bool = False,
    hw_startup: bool = True,
    transpose: bool | None = None,
    label: str = "mm",
    row_scale: tuple | None = None,
    k_order: dict | None = None,
    arena: dict | None = None,
    in1_cb_blocks: int = 2,
    m_batches: int = 1,
    signal_m_batches: tuple | None = None,
    n_range: tuple | None = None,
) -> MatmulPlan:
    """Place one matmul out[M,N] = in0[M,K] @ weight[K,N] (out N halved with SwiGLU) on the rectangle of
    `grid` cores at `origin`. Mirrors minimal_matmul_factory_helper_common (no bias / ternary / split / fusion).
    row_scale = (scale tensor [M, 32], semaphore id, count): multiply output row tile i by scale tile i once the
    semaphore (on every core of the grid) reaches count.
    k_order = dict(first, slices, counters, waits): visit K in slice-major order from slice `first` (own, +1, -1,
    +2, ...) and let the in0 injectors wait per slice on L1 counters (an all-gather's signal); waits(core) ->
    (own, near, far) counts for that injector.
    arena = {cb name: (Arena, offset)} for any of in0 / in1 / out / interm: place those CBs there (see Arena).
    in1_cb_blocks: depth of the weight CB in K blocks (2 = double buffered; deeper lets the weights stream ahead).
    m_batches > 1: the M rows are m_batches equal batches, each core computes its rows of batch 0, then of batch 1,
    ... (cfg.M_block must be a core's rows per batch); signal_m_batches = (counters, targets): the output writers
    increment counter word 16*b on each target once their batch-b output has landed.
    n_range = (n0, n_tiles): compute only weight columns [n0, n0 + n_tiles) (output columns halved with SwiGLU), so
    several placements on different cores can split one matmul by N (more cores reading weights in parallel). Every
    core gets a full range of ceil(n_tiles / cores) columns; columns past the tensor's width are padding the readers
    skip, but ranges past n0 + n_tiles inside the tensor overlap the next placement, so split at multiples of that."""
    ox, oy = origin
    gx, gy = grid
    cores = [(ox + i, oy + j) for j in range(gy) for i in range(gx)]
    K = in0.padded_shape[-1]
    M = 1
    for d in list(in0.padded_shape)[:-1]:
        M *= d
    N = weight.padded_shape[-1]
    M_tiles, K_tiles, N_tiles = M // 32, K // 32, N // 32

    if transpose is None:
        transpose = M > N
    # Small input + output on RISCV_1 / NOC_1, large input on RISCV_0 / NOC_0 (see minimal_matmul); transposing
    # swaps which input is which.
    in0_risc, in1_risc = ("dm0", "dm1") if transpose else ("dm1", "dm0")
    in0_noc0 = in0_risc == "dm0"
    in1_noc0 = in1_risc == "dm0"
    in0_axis = gx if transpose else gy  # cores M is split over
    in1_axis = gy if transpose else gx  # cores N is split over

    padded_M_tiles = round_up(M_tiles, in0_axis)
    padded_K_tiles = round_up(K_tiles, cfg.K_block)
    n0, n_span = n_range if n_range is not None else (0, N_tiles)
    if fuse_swiglu:
        assert n0 % 2 == 0 and n_span % 2 == 0
        out_N = n_span // 2
        padded_out_N = round_up(out_N, in1_axis)
        N_tiles_per_core = 2 * (padded_out_N // in1_axis)
        assert N_tiles % 2 == 0 and N_tiles_per_core % 2 == 0 and cfg.N_block % 2 == 0
    else:
        N_tiles_per_core = round_up(n_span, in1_axis) // in1_axis
    # The kernels address tiles by the full tensor width; padding only matters past the tensor's last column.
    padded_N_tiles = max(N_tiles, n0 + N_tiles_per_core * in1_axis)
    if fuse_swiglu:
        padded_N_tiles += padded_N_tiles % 2
    M_tiles_per_core = padded_M_tiles // in0_axis
    if m_batches > 1:
        assert M_tiles % (m_batches * in0_axis) == 0, (M_tiles, m_batches, in0_axis)
        m_per_batch = M_tiles // (m_batches * in0_axis)
        assert cfg.M_block == m_per_batch, f"M_block must be the rows per core per batch ({m_per_batch})"
    mb_ct = dict(m_batches=m_batches, m_batch_tiles=M_tiles // m_batches)
    K_blocks = padded_K_tiles // cfg.K_block
    M_blocks_per_core = div_up(M_tiles_per_core, cfg.M_block)
    if m_batches > 1:
        M_blocks_per_core = m_batches
    N_blocks_per_core = div_up(N_tiles_per_core, cfg.N_block)

    in0_tb, in1_tb, out_tb = tile_bytes(in0.dtype), tile_bytes(weight.dtype), tile_bytes(out.dtype)
    interm_fmt = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    in0_blk, in1_blk, out_blk = cfg.M_block * cfg.K_block, cfg.K_block * cfg.N_block, cfg.M_block * cfg.N_block
    out_written = out_blk // 2 if fuse_swiglu else out_blk

    arena = arena or {}
    cb_in0 = prog.cb(cores, 2 * in0_blk, in0_tb, in0.dtype, arena=arena.get("in0"))
    cb_in1 = prog.cb(cores, in1_cb_blocks * in1_blk, in1_tb, weight.dtype, arena=arena.get("in1"))
    cb_out = prog.cb(cores, 2 * out_written, out_tb, out.dtype, arena=arena.get("out"))
    cb_interm = prog.cb(cores, out_blk, tile_bytes(interm_fmt), interm_fmt, arena=arena.get("interm"))

    if row_scale is not None:
        scale_t, scale_sem, scale_count = row_scale
        cb_scale = prog.cb(cores, cfg.M_block, tile_bytes(scale_t.dtype), scale_t.dtype)
        rs_ct = dict(
            row_scale=True,
            cb_scale=cb_scale,
            scale_sem=scale_sem,
            scale_count=scale_count,
            scale_tile_size=tile_bytes(scale_t.dtype),
        )
    else:
        scale_t, cb_scale, rs_ct = None, 0, dict(row_scale=False)

    if k_order is not None:
        slice_blocks = (K_tiles // k_order["slices"]) // cfg.K_block
        assert slice_blocks * cfg.K_block * k_order["slices"] == K_tiles, "K blocks must not straddle slices"
        ko_ct = dict(k_order=True, k_slices=k_order["slices"], k_slice_blocks=slice_blocks, k_first=k_order["first"])
        ko0_ct = dict(ko_ct)
        ko_rt = lambda c: [k_order["counters"].buffer_address(), *k_order["waits"](c)]  # noqa: E731
    else:
        ko_ct = ko0_ct = dict(k_order=False)
        ko_rt = lambda c: []  # noqa: E731

    sems = {
        k: prog.sem(cores, v)
        for k, v in (
            ("in0_sender", INVALID),
            ("in0_receiver", INVALID),
            ("in0_valid", VALID),
            ("in1_sender", INVALID),
            ("in1_receiver", INVALID),
            ("in1_valid", VALID),
        )
    }

    k_blocks_per_core = div_up(K_blocks, in1_axis if transpose else in0_axis)
    max_defer = max(min(j * k_blocks_per_core, K_blocks - 1) for j in range(gy))

    common_ct = dict(
        M_tiles=M_tiles,
        padded_M_tiles=padded_M_tiles,
        K_tiles=K_tiles,
        padded_K_tiles=padded_K_tiles,
        N_tiles=N_tiles,
        padded_N_tiles=padded_N_tiles,
        M_block_tiles=cfg.M_block,
        K_block_tiles=cfg.K_block,
        N_block_tiles=cfg.N_block,
        M_blocks_per_core=M_blocks_per_core,
        N_blocks_per_core=N_blocks_per_core,
        out_tile_size=out_tb,
        fuse_swiglu=fuse_swiglu,
        cb_out=cb_out,
        **mb_ct,
        signal_m_batches=signal_m_batches is not None,
    )

    def phys(core):
        c = device.worker_core_from_logical_core(ttnn.CoreCoord(*core))
        return c.x, c.y

    def rel(core):
        return core[0] - ox, core[1] - oy

    def indices(core):
        i, j = rel(core)
        in0_idx = i if transpose else j
        in1_idx = j if transpose else i
        return in0_idx, in1_idx

    def chain(core, along_x: bool, increasing: bool):
        """(is_sink, next core, prev core) of `core` in the forward chain along x (or y)."""
        i, j = rel(core)
        pos, length = (i, gx) if along_x else (j, gy)
        order, at = _axis_order(pos, length, increasing)
        nxt = order[min(at + 1, len(order) - 1)]
        prv = order[max(at - 1, 0)]
        mk = (lambda p: (ox + p, oy + j)) if along_x else (lambda p: (ox + i, oy + p))
        return at == len(order) - 1, mk(nxt), mk(prv)

    def ranges(core):
        in0_idx, in1_idx = indices(core)
        mpc = M_tiles_per_core if m_batches == 1 else cfg.M_block  # with batches: the rows of batch 0
        return (
            mpc * in0_idx,
            mpc * (in0_idx + 1),
            n0 + N_tiles_per_core * in1_idx,
            n0 + N_tiles_per_core * (in1_idx + 1),
        )

    def dm_rt(core, along_x, increasing, tensor):
        is_sink, nxt, prv = chain(core, along_x, increasing)
        m0, m1, n0, n1 = ranges(core)
        j = rel(core)[1]
        defer = min(j * k_blocks_per_core, K_blocks - 1)
        return [
            tensor.buffer_address(),
            int(is_sink),
            *phys(nxt),
            *phys(prv),
            m0,
            m1,
            n0,
            n1,
            defer,
            max_defer,
            out.buffer_address(),
        ]

    def in0_rt(c):
        extra = [scale_t.buffer_address()] if scale_t is not None else []
        sig = []
        if signal_m_batches is not None:
            counters, targets = signal_m_batches
            sig = [counters.buffer_address(), len(targets)] + [v for t in targets for v in phys(t)]
        return dm_rt(c, in0_along_x, in0_noc0, in0) + extra + ko_rt(c) + sig

    # in0 travels along x when not transposed (rows of the grid share M), along y when transposed.
    in0_along_x = not transpose
    in0_injectors = [c for c in cores if (rel(c)[0] if in0_along_x else rel(c)[1]) == 0]
    in1_injectors = [c for c in cores if (rel(c)[1] if in0_along_x else rel(c)[0]) == 0]
    for injector in (True, False):
        sel = [c for c in cores if (c in in0_injectors) == injector]
        if sel:
            prog.place(
                sel,
                Block(
                    cls="lego::mm::In0",
                    header=f"{KDIR}/mm/in0.hpp",
                    risc=in0_risc,
                    ct=dict(
                        common_ct,
                        in0_tile_size=in0_tb,
                        sender_sem=sems["in0_sender"],
                        receiver_sem=sems["in0_receiver"],
                        valid_sem=sems["in0_valid"],
                        is_output_writer=not transpose,
                        is_injector_core=injector,
                        cb_in0=cb_in0,
                        **rs_ct,
                        **ko0_ct,
                    ),
                    rt=in0_rt,
                    tensors={"in0": in0, "out": out, **({"scale": scale_t} if scale_t is not None else {})},
                    label=f"{label}_in0",
                ),
            )
        sel = [c for c in cores if (c in in1_injectors) == injector]
        if sel:
            prog.place(
                sel,
                Block(
                    cls="lego::mm::In1",
                    header=f"{KDIR}/mm/in1.hpp",
                    risc=in1_risc,
                    ct=dict(
                        common_ct,
                        in1_tile_size=in1_tb,
                        sender_sem=sems["in1_sender"],
                        receiver_sem=sems["in1_receiver"],
                        valid_sem=sems["in1_valid"],
                        is_output_writer=transpose,
                        is_injector_core=injector,
                        cb_in1=cb_in1,
                        **ko_ct,
                    ),
                    rt=lambda c: dm_rt(c, not in0_along_x, in1_noc0, weight),
                    tensors={"in1": weight, "out": out},
                    label=f"{label}_in1",
                ),
            )
    prog.place(
        cores,
        Block(
            cls="lego::mm::Compute",
            header=f"{KDIR}/mm/compute.hpp",
            risc="compute",
            ct=dict(
                K_num_blocks=K_blocks,
                M_block_tiles=cfg.M_block,
                K_block_tiles=cfg.K_block,
                N_block_tiles=cfg.N_block,
                M_blocks_per_core=M_blocks_per_core,
                N_blocks_per_core=N_blocks_per_core,
                subblock_h=cfg.subblock_h,
                subblock_w=cfg.subblock_w,
                fuse_swiglu=fuse_swiglu,
                **mb_ct,
                # SwiGLU on the pack thread in the last K block; with a row scale its tiles need DST slots too
                swiglu_in_k_loop=fuse_swiglu
                and cfg.subblock_w % 2 == 0
                and (row_scale is None or cfg.subblock_h * cfg.subblock_w + cfg.subblock_h <= 8),
                swiglu_block_float_output=out.dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b),
                hw_startup=hw_startup,
                cb_in0=cb_in0,
                cb_in1=cb_in1,
                cb_out=cb_out,
                cb_interm=cb_interm,
                row_scale=row_scale is not None,
                cb_scale=cb_scale,
            ),
            rt=lambda c: list(ranges(c)),
            compute=dict(math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_dest_acc_en, math_approx_mode=False),
            label=f"{label}_compute",
        ),
    )
    return MatmulPlan(
        cores,
        transpose,
        M_tiles,
        K_tiles,
        N_tiles,
        in0_injectors,
        "dm0" if (transpose and in1_risc == "dm0") or (not transpose and in0_risc == "dm0") else "dm1",
    )


def matmul_cb_bytes(
    in0_dtype, w_dtype, out_dtype, cfg: MatmulConfig, fuse_swiglu=False, fp32_dest_acc_en=False, in1_cb_blocks=2
) -> dict:
    """Bytes of each CB place_matmul creates (for laying them out in an Arena)."""
    in0_blk, in1_blk, out_blk = cfg.M_block * cfg.K_block, cfg.K_block * cfg.N_block, cfg.M_block * cfg.N_block
    out_written = out_blk // 2 if fuse_swiglu else out_blk
    interm = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    return {
        "in0": 2 * in0_blk * tile_bytes(in0_dtype),
        "in1": in1_cb_blocks * in1_blk * tile_bytes(w_dtype),
        "out": 2 * out_written * tile_bytes(out_dtype),
        "interm": out_blk * tile_bytes(interm),
    }
