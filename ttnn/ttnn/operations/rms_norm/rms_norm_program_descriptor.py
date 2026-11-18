# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor.

`select_blocking()` is the single source of every blocking knob (op_design.md -> Work Distribution); every CB page
count, loop trip count, RT arg and Mcast group size below is derived from its result.

Regimes (both built):
  LOCAL   (w_split == 1): R (tile-rows) split over lines of one core; whole row resident per core.
  W_SPLIT (w_split  > 1): lines of `w_split` consecutive cores (row-major) share an R-range; each core holds its
                          W-slice resident; Σx² partials are gathered to the line root (cores[0] of the line), the
                          root combines + finalizes rinv and multicasts it back to every core of the line.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
FACE = 16

# ---- knob constants (one definition each; op_design.md Work Distribution) ----
L1_CB_BUDGET_BYTES = 1 << 20
MIN_CORE_W_TILES = 8
X_BLOCK_DEPTH = 2  # cb_x (TILE) depth in blocks
X_RM_DEPTH = 2  # cb_x_rm depth in tile-rows
Y_CHUNK_TILES = 8  # cb_y writer chunk
Y_DEPTH = 2  # cb_y depth in chunks
OUT_RM_DEPTH = 2  # cb_out_rm depth in tile-rows
READ_CHUNK_TILES = 8  # reader tiles per barrier+push (TILE)

# ---- CB indices (semantic names) ----
CB_X_RM = 0
CB_X = 1
CB_GAMMA_RM = 2
CB_GAMMA = 3
CB_SUMSQ_RAW = 4
CB_PARTIAL = 5
CB_GATHERED = 6
CB_RINV_COL = 7
CB_RINV_BCAST = 8
CB_RINV = 9
CB_Y = 10
CB_OUT_RM = 11

# ---- semaphore ids ----
SEM_PARTIALS = 0  # root: count of partials landed in cb_gathered (cumulative)
SEM_RINV_LOCAL = 1  # root: root writer -> root reader "rinv(b) landed in my own cb_rinv" (cumulative)
SEM_MCAST_FIRST = 2  # Mcast channel semaphores start here
MCAST_PREFIX = "rinv_mcast"


def _div_up(a, b):
    return (a + b - 1) // b


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@dataclass(frozen=True)
class OpGeometry:
    is_tile: bool
    has_gamma: bool
    gamma_is_rm: bool
    t_in: int  # input (= output) tile bytes
    t_g: int  # gamma tile bytes
    es_g: int  # gamma element bytes
    t_f: int  # Float32 tile bytes


@dataclass(frozen=True)
class Blocking:
    regime: str  # "LOCAL" | "W_SPLIT"
    w_split: int
    core_w_tiles: int
    num_lines: int
    block_rows: int


def _per_row_bytes(g: OpGeometry, cw: int, k: int) -> int:
    ws = k > 1
    b = 0
    b += (X_BLOCK_DEPTH * cw * g.t_in) if g.is_tile else (cw * g.t_in)  # cb_x
    b += 3 * g.t_f  # cb_sumsq_raw, cb_rinv_col, cb_rinv
    if ws:
        b += (2 + k) * g.t_f  # cb_partial, cb_rinv_bcast, cb_gathered
    return b


def _fixed_bytes(g: OpGeometry, cw: int) -> int:
    b = 0
    if g.has_gamma:
        b += cw * g.t_g  # cb_gamma
        if g.gamma_is_rm:
            b += cw * TILE * g.es_g  # cb_gamma_rm
    if g.is_tile:
        b += Y_DEPTH * Y_CHUNK_TILES * g.t_in  # cb_y
    else:
        b += X_RM_DEPTH * cw * g.t_in + OUT_RM_DEPTH * cw * g.t_in  # cb_x_rm, cb_out_rm
    return b


def _footprint(g: OpGeometry, br: int, cw: int, k: int) -> int:
    return br * _per_row_bytes(g, cw, k) + _fixed_bytes(g, cw)


def select_blocking(g: OpGeometry, R: int, Wt: int, C: int) -> Blocking:
    """The single source of every knob (op_design.md select_blocking)."""
    k_fit = None
    for k in range(1, C + 1):
        if _footprint(g, 1, _div_up(Wt, k), k) <= L1_CB_BUDGET_BYTES:
            k_fit = k
            break
    if k_fit is None:
        from ttnn.operations._op_contract import UnsupportedAxisValue

        raise UnsupportedAxisValue("rms_norm: row too wide to hold resident across the grid")
    k_occ = 1 if R >= C else max(1, min(C // R, Wt // MIN_CORE_W_TILES))
    k = max(k_fit, k_occ)
    cw = _div_up(Wt, k)
    w_split = _div_up(Wt, cw)  # every core of a line has >= 1 valid tile
    num_lines = min(R, C // w_split)
    line_rows_tiles_max = _div_up(R, num_lines)
    br_fit = (L1_CB_BUDGET_BYTES - _fixed_bytes(g, cw)) // _per_row_bytes(g, cw, w_split)
    block_rows = max(1, min(line_rows_tiles_max, br_fit))
    regime = "LOCAL" if w_split == 1 else "W_SPLIT"
    return Blocking(regime, w_split, cw, num_lines, block_rows)


def _core_range_set(cores):
    """Merge a row-major list of cores into per-row contiguous CoreRanges."""
    by_row = {}
    for c in cores:
        by_row.setdefault(c.y, []).append(c.x)
    ranges = []
    for y in sorted(by_row):
        xs = sorted(by_row[y])
        start = prev = xs[0]
        for x in xs[1:]:
            if x != prev + 1:
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(start, y), ttnn.CoreCoord(prev, y)))
                start = x
            prev = x
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(start, y), ttnn.CoreCoord(prev, y)))
    return ttnn.CoreRangeSet(ranges)


def _cb(index, num_pages, page_size, dtype, crs):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(input_tensor, gamma, output_tensor, epsilon, compute_config):
    device = input_tensor.device()
    shape = [int(d) for d in input_tensor.shape]
    W, H = shape[-1], shape[-2]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    is_tile = input_tensor.layout == ttnn.TILE_LAYOUT
    dtype = input_tensor.dtype
    es = input_tensor.element_size()
    has_gamma = gamma is not None
    gamma_is_rm = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT
    g_dtype = gamma.dtype if has_gamma else dtype

    geo = OpGeometry(
        is_tile=is_tile,
        has_gamma=has_gamma,
        gamma_is_rm=gamma_is_rm,
        t_in=ttnn.tile_size(dtype),
        t_g=ttnn.tile_size(g_dtype),
        es_g=gamma.element_size() if has_gamma else es,
        t_f=ttnn.tile_size(ttnn.float32),
    )

    # ---- shape-derived extents (alignment-aware) ----
    Wt = _div_up(W, TILE)
    w_valid_last = W - TILE * (Wt - 1)
    num_sticks = lead * H  # RM: total sticks (rows)
    R = lead * _div_up(H, TILE) if is_tile else _div_up(num_sticks, TILE)

    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    C = gx * gy

    blk = select_blocking(geo, R, Wt, C)
    k, cw, br = blk.w_split, blk.core_w_tiles, blk.block_rows
    is_ws = blk.regime == "W_SPLIT"

    cores = ttnn.grid_to_cores(blk.num_lines * k, gx, gy, row_wise=True)
    crs = _core_range_set(cores)

    # ---- CBs (identical descriptors on every participating core: uniform addresses) ----
    t_in, t_f = geo.t_in, geo.t_f
    cbs = []
    if not is_tile:
        cbs.append(_cb(CB_X_RM, X_RM_DEPTH * cw, t_in, dtype, crs))
    cbs.append(_cb(CB_X, (X_BLOCK_DEPTH * br * cw) if is_tile else (br * cw), t_in, dtype, crs))
    if has_gamma:
        if gamma_is_rm:
            cbs.append(_cb(CB_GAMMA_RM, 1, cw * TILE * geo.es_g, g_dtype, crs))
        cbs.append(_cb(CB_GAMMA, cw, geo.t_g, g_dtype, crs))
    cbs.append(_cb(CB_SUMSQ_RAW, br, t_f, ttnn.float32, crs))
    if is_ws:
        cbs.append(_cb(CB_PARTIAL, br, t_f, ttnn.float32, crs))
        cbs.append(_cb(CB_GATHERED, k * br, t_f, ttnn.float32, crs))
        cbs.append(_cb(CB_RINV_BCAST, br, t_f, ttnn.float32, crs))
    cbs.append(_cb(CB_RINV_COL, br, t_f, ttnn.float32, crs))
    cbs.append(_cb(CB_RINV, br, t_f, ttnn.float32, crs))
    if is_tile:
        cbs.append(_cb(CB_Y, Y_DEPTH * Y_CHUNK_TILES, t_in, dtype, crs))
    else:
        cbs.append(_cb(CB_OUT_RM, OUT_RM_DEPTH * cw, t_in, dtype, crs))

    semaphores = []
    if is_ws:
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_PARTIALS, core_ranges=crs, initial_value=0))
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_RINV_LOCAL, core_ranges=crs, initial_value=0))

    # ---- per-core work split ----
    x_page = input_tensor.buffer_page_size()
    y_page = output_tensor.buffer_page_size()
    g_page = gamma.buffer_page_size() if has_gamma else 0
    x_addr = input_tensor.buffer_address()
    y_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0

    reader_rt_by_core = {}
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    inv_w_bits = _f32_bits(1.0 / W)
    eps_bits = _f32_bits(epsilon)

    base_rows, extra_rows = divmod(R, blk.num_lines)
    row_start = 0
    for line in range(blk.num_lines):
        line_rows = base_rows + (1 if line < extra_rows else 0)
        line_cores = cores[line * k : (line + 1) * k]
        root = line_cores[0]
        root_v = device.worker_core_from_logical_core(root)
        for pos, core in enumerate(line_cores):
            w_start = pos * cw
            w_valid = min(cw, Wt - w_start)
            w_valid_elems = min(cw * TILE, W - w_start * TILE)
            is_root = 1 if pos == 0 else 0
            owns_partial_last = 1 if (w_start + w_valid == Wt and w_valid_last < TILE) else 0
            reader_rt_by_core[(core.x, core.y)] = [
                x_addr,
                row_start,
                line_rows,
                w_start,
                w_valid,
                w_valid_elems,
                owns_partial_last,
                is_root,
                num_sticks,
            ]
            writer_rt[core.x][core.y] = [
                y_addr,
                g_addr,
                row_start,
                line_rows,
                w_start,
                w_valid,
                w_valid_elems,
                is_root,
                pos,
                root_v.x,
                root_v.y,
                num_sticks,
            ]
            compute_rt[core.x][core.y] = [line_rows, is_root, inv_w_bits, eps_bits]
        row_start += line_rows

    # ---- kernels ----
    reader_ct = [
        int(not is_tile),  # 0 is_rm
        int(is_ws),  # 1 is_wsplit
        br,  # 2 block_rows
        cw,  # 3 core_w_tiles
        READ_CHUNK_TILES,  # 4
        X_BLOCK_DEPTH,  # 5
        X_RM_DEPTH,  # 6
        t_in,  # 7 tile bytes of cb_x / cb_x_rm pages
        es,  # 8 input element bytes
        w_valid_last,  # 9
        x_page,  # 10 accessor page size
        Wt,  # 11
        k,  # 12 w_split
        SEM_PARTIALS,  # 13
        SEM_RINV_LOCAL,  # 14
        t_f,  # 15
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    def _reader(core_list):
        rt = ttnn.RuntimeArgs()
        for c in core_list:
            rt[c.x][c.y] = reader_rt_by_core[(c.x, c.y)]
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
            core_ranges=_core_range_set(core_list),
            compile_time_args=list(reader_ct),
            runtime_args=rt,
            config=ttnn.ReaderConfigDescriptor(),
        )

    # W_SPLIT: the line roots' readers never touch the multicast channel (the root is its sender; its own copy
    # is signalled by SEM_RINV_LOCAL), and the Mcast adapter requires every kernel attached on a sender core to
    # run on the channel's NoC — so the roots get their own reader descriptor with an absent channel.
    root_cores = [c for i, c in enumerate(cores) if i % k == 0] if is_ws else []
    member_cores = [c for i, c in enumerate(cores) if i % k != 0] if is_ws else list(cores)
    readers_absent = [_reader(root_cores)] if root_cores else []
    readers_attached = [_reader(member_cores)] if member_cores else []

    writer_ct = [
        int(not is_tile),  # 0 is_rm
        int(is_ws),  # 1 is_wsplit
        int(has_gamma),  # 2
        int(gamma_is_rm),  # 3
        br,  # 4 block_rows
        cw,  # 5 core_w_tiles
        Y_CHUNK_TILES,  # 6
        Y_DEPTH,  # 7
        OUT_RM_DEPTH,  # 8
        t_in,  # 9 output tile bytes
        es,  # 10 output element bytes
        y_page,  # 11 output accessor page size
        g_page,  # 12 gamma accessor page size
        geo.es_g,  # 13 gamma element bytes
        geo.t_g,  # 14 gamma tile bytes
        Wt,  # 15
        k,  # 16 w_split
        SEM_PARTIALS,  # 17
        SEM_RINV_LOCAL,  # 18
        t_f,  # 19
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=crs,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_ct = [
        int(not is_tile),  # 0 is_rm
        int(is_ws),  # 1 is_wsplit
        int(has_gamma),  # 2
        int(gamma_is_rm),  # 3
        br,  # 4 block_rows
        cw,  # 5 core_w_tiles
        k,  # 6 w_split
        int(bool(compute_config.math_approx_mode)),  # 7 rsqrt approx
    ]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=crs,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )

    descriptor = ttnn.ProgramDescriptor(cbs=cbs, semaphores=semaphores)
    if is_ws:
        # rinv broadcast: one fixed in-group sender (the line root = cores[0] of each group of w_split
        # consecutive row-major cores), driven from the writer (NoC of WriterConfigDescriptor).
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1),
            crs,
            k,
            ttnn.McastFixedSenderConfig(sender_index=0),
            ttnn.McastCoreOrder.RowMajor,
        )
        mcast.attach(descriptor, MCAST_PREFIX, readers_attached + [writer], SEM_MCAST_FIRST)
    else:
        ttnn.attach_absent_mcast(writer, MCAST_PREFIX)
        for r in readers_attached:
            ttnn.attach_absent_mcast(r, MCAST_PREFIX)
    for r in readers_absent:
        ttnn.attach_absent_mcast(r, MCAST_PREFIX)
    descriptor.kernels = readers_absent + readers_attached + [writer, compute]
    return descriptor
