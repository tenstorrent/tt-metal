# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor: geometry, select_blocking(), CBs, kernels, per-core args.

Blocking (op_design.md -> Blocking Model):
  * row axis (tile-rows, leading dims folded): split over row-groups across the whole Tensix grid;
    each group's assignment is processed in blocks of `block_row_tiles` tile-rows.
  * W axis: split over `w_split_cores` (k) consecutive Tensix cores of one grid row; every member
    keeps its W share resident (cb_x) across pass 1 and pass 2, and the k members all-gather their
    per-row sum-of-squares partials over the NoC (k > 1 only).
Every knob below (k, block_row_tiles, x_block_depth, rm_chunk_w_tiles, out_unit_tiles) is computed
once here and handed to the kernels as a CT/RT arg; the kernels derive all trip counts from them.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32

# CB indices — must match kernels/rms_norm_common.hpp.
CB_X_RM = 0  # RM: raw stick pieces (tile-sized pages) -> tilize
CB_X = 1  # resident x block (ping)
CB_GAMMA = 2  # expanded gamma slice, program lifetime
CB_SUMSQ = 3  # per tile-row DEST-accumulated sum of x^2 (pass-1 output)
CB_PARTIAL = 4  # k > 1: this member's per-row partial (column 0)
CB_GATHERED = 5  # k > 1: k partials per row, all-gathered over the NoC
CB_INV_RMS = 6  # per tile-row rsqrt(mean + eps) (column 0)
CB_OUT = 7  # normalized tiles (TILE: -> writer; RM: -> untilize)
CB_OUT_RM = 8  # RM: untilized stick pieces -> writer
CB_X_ALT = 9  # resident x block (pong) — only when x_block_depth == 2

SEM_PARTIALS_ARRIVED = 0
WRITER_NOC_XY_ARG_BASE = 9  # writer RT index of the first peer (x, y) pair — passed to the kernel as a CT arg

# Knob constants (design Phase 0 values).
K_CAP_MAX = 8  # unicast all-gather mechanism cap
OUT_DEPTH = 2  # cb_out / cb_out_rm buffer depth
RM_IN_DEPTH = 2  # cb_x_rm buffer depth
OUT_UNIT_CAP = 16  # TILE writer batch (tiles per wait/flush/pop)
RM_CHUNK_CAP = 32  # balanced RM streaming chunk cap (tiles)
RM_PIECE_FLOOR_BYTES = 512  # occupancy split must keep RM stick pieces >= this
OCC_TILES_PER_MEMBER = 4  # occupancy split keeps >= 4 W tiles per member
OCC_ROWS_FRACTION = 8  # occupancy split only when 8 * tensor_row_tiles <= G


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class Geometry:
    W: int
    H: int
    num_images: int
    image_row_tiles: int
    tensor_row_tiles: int
    tensor_w_tiles: int
    w_valid_in_last_tile: int


def derive_geometry(shape) -> Geometry:
    shape = [int(d) for d in shape]
    W, H = shape[-1], shape[-2]
    num_images = 1
    for d in shape[:-2]:
        num_images *= d
    image_row_tiles = _div_up(H, TILE)
    tensor_w_tiles = _div_up(W, TILE)
    return Geometry(
        W=W,
        H=H,
        num_images=num_images,
        image_row_tiles=image_row_tiles,
        tensor_row_tiles=num_images * image_row_tiles,
        tensor_w_tiles=tensor_w_tiles,
        w_valid_in_last_tile=W - TILE * (tensor_w_tiles - 1),
    )


@dataclass
class Blocking:
    k: int  # w_split_cores
    slice_w_tiles: int  # W tiles of every non-last member
    rm_chunk_w_tiles: int  # C (RM streaming quantum along W; TILE: unused, = 1)
    x_row_pitch: int  # tile pitch of one tile-row inside cb_x (uniform across cores)
    num_groups: int  # row-groups actually used
    groups_per_grid_row: int
    group_rows: list  # per-group (row_start, core_row_tiles)
    block_row_tiles: int
    x_block_depth: int
    out_unit_tiles: int
    footprint: int


class _Sizes:
    def __init__(self, x_dtype, gamma_dtype, is_rm, has_gamma):
        self.T_in = ttnn.tile_size(x_dtype)
        self.T_out = self.T_in
        self.T_g = ttnn.tile_size(gamma_dtype) if has_gamma else 0
        self.T32 = ttnn.tile_size(ttnn.float32)
        self.is_rm = is_rm
        self.has_gamma = has_gamma


def _normalize_k(tw: int, k: int):
    s = _div_up(tw, k)
    return _div_up(tw, s), s


def _out_unit(slice_w_tiles: int) -> int:
    """TILE writer batch (tiles per wait/write/flush/pop) — also the cb_out depth unit."""
    return min(slice_w_tiles, OUT_UNIT_CAP)


def _rm_chunk(slice_w_tiles: int):
    chunks = _div_up(slice_w_tiles, RM_CHUNK_CAP)
    c = _div_up(slice_w_tiles, chunks)
    return c, chunks * c


def cb_plan(sz: _Sizes, k: int, slice_w: int, b: int, d: int) -> dict:
    """Single source of truth for every CB's capacity: {cb_index: (pages, page_bytes)}.

    Both footprint() (the L1-fit solve in select_blocking) and create_program_descriptor() (the CB
    declarations) read this dict, so a knob change lands in exactly one place (l1_ledger.md rows).
    """
    if sz.is_rm:
        c, pitch = _rm_chunk(slice_w)
    else:
        c, pitch = 1, slice_w
    plan = {
        CB_X: (b * pitch, sz.T_in),  # resident x block
        CB_SUMSQ: (b, sz.T32),
        CB_INV_RMS: (b, sz.T32),
    }
    if d == 2:
        plan[CB_X_ALT] = (b * pitch, sz.T_in)  # second half of the depth-2 x budget
    if sz.has_gamma:
        plan[CB_GAMMA] = (pitch, sz.T_g)
    if k > 1:
        plan[CB_PARTIAL] = (b, sz.T32)
        plan[CB_GATHERED] = (2 * k * b, sz.T32)  # 2 blocks: lockstep slot-reuse mechanism cap
    if sz.is_rm:
        plan[CB_X_RM] = (RM_IN_DEPTH * c, sz.T_in)
        plan[CB_OUT] = (b * c, sz.T_out)  # spans one chunk column of the block (untilize input)
        plan[CB_OUT_RM] = (OUT_DEPTH * c, sz.T_out)
    else:
        plan[CB_OUT] = (OUT_DEPTH * _out_unit(slice_w), sz.T_out)
    return plan


def footprint(sz: _Sizes, tw: int, k: int, b: int, d: int) -> int:
    """Per-Tensix-core CB bytes — the l1_ledger.md expression, summed from cb_plan()."""
    k, s = _normalize_k(tw, k)
    return sum(pages * page_bytes for pages, page_bytes in cb_plan(sz, k, s, b, d).values())


def select_blocking(geo: Geometry, sz: _Sizes, grid_x: int, grid_y: int, budget: int, elem_bytes: int) -> Blocking:
    G = grid_x * grid_y
    k_cap = min(K_CAP_MAX, grid_x)
    tw, trt = geo.tensor_w_tiles, geo.tensor_row_tiles

    k_fit = None
    for k in range(1, k_cap + 1):
        if footprint(sz, tw, k, 1, 1) <= budget:
            k_fit = k
            break
    if k_fit is None:
        raise RuntimeError(
            f"rms_norm: W={geo.W} too wide — even w_split_cores={k_cap} does not fit L1 ({budget} B per core)"
        )

    k_occ = 1
    if OCC_ROWS_FRACTION * trt <= G and tw >= 8:
        k_occ = min(k_cap, G // trt, tw // OCC_TILES_PER_MEMBER)
        if sz.is_rm:
            k_occ = min(k_occ, max(1, (geo.W * elem_bytes) // RM_PIECE_FLOOR_BYTES))
        k_occ = max(1, k_occ)
    k, slice_w = _normalize_k(tw, max(k_fit, k_occ))

    if sz.is_rm:
        c, pitch = _rm_chunk(slice_w)
    else:
        c, pitch = 1, slice_w

    groups_per_grid_row = grid_x // k
    num_groups_avail = groups_per_grid_row * grid_y
    num_groups = min(num_groups_avail, trt)
    base, rem = divmod(trt, num_groups)
    group_rows, start = [], 0
    for g in range(num_groups):
        n = base + (1 if g < rem else 0)
        group_rows.append((start, n))
        start += n
    crt_max = base + (1 if rem else 0)

    if footprint(sz, tw, k, crt_max, 1) <= budget:
        b, d = crt_max, 1
    else:
        d = 1 if sz.is_rm else 2
        b = 0
        for cand in range(crt_max, 0, -1):
            if footprint(sz, tw, k, cand, d) <= budget:
                b = cand
                break
        if b == 0:
            b, d = 1, 1
    if b >= crt_max:
        d = 1  # one block per core: nothing to double-buffer

    return Blocking(
        k=k,
        slice_w_tiles=slice_w,
        rm_chunk_w_tiles=c,
        x_row_pitch=pitch,
        num_groups=num_groups,
        groups_per_grid_row=groups_per_grid_row,
        group_rows=group_rows,
        block_row_tiles=b,
        x_block_depth=d,
        out_unit_tiles=_out_unit(slice_w),
        footprint=footprint(sz, tw, k, b, d),
    )


def cb_l1_budget(device) -> int:
    """Bytes of L1 per Tensix core the program's CBs may occupy.

    CBs are placed bottom-up from the allocator's L1 base; L1 buffers (interleaved-L1 / sharded
    tensors, incl. this op's own output) are allocated top-down. The budget is therefore the free
    region between the allocator base and the lowest allocated L1 buffer. NOT
    `ttnn.get_max_worker_l1_unreserved_size()`: that figure starts at the kernel-config ring buffer,
    below the allocator base, and overshoots by its size (~69 KB on BH) — shapes whose footprint fell
    in that gap threw "circular buffers grow to N B which is beyond max L1 size".
    """
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    allocated = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
    return min(allocated) if allocated else int(view.total_bytes_per_bank)


def _cb(index, pages, page_size, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config):
    device = input_tensor.device()
    geo = derive_geometry(input_tensor.shape)
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None
    x_dtype = input_tensor.dtype
    g_dtype = gamma.dtype if has_gamma else x_dtype
    sz = _Sizes(x_dtype, g_dtype, is_rm, has_gamma)
    elem_bytes = input_tensor.element_size()
    g_elem_bytes = gamma.element_size() if has_gamma else elem_bytes
    gamma_is_tile = has_gamma and gamma.layout == ttnn.TILE_LAYOUT

    grid = device.compute_with_storage_grid_size()
    budget = cb_l1_budget(device)
    blk = select_blocking(geo, sz, grid.x, grid.y, budget, elem_bytes)
    k, C, pitch = blk.k, blk.rm_chunk_w_tiles, blk.x_row_pitch
    b, d = blk.block_row_tiles, blk.x_block_depth
    combine_lookahead = 1 if (k > 1 and d == 2) else 0

    # ---- core placement: group (vx, vy) -> Tensix cores (vx*k + m, vy) ----
    group_cores = []  # per group: list of k logical CoreCoord
    for g in range(blk.num_groups):
        vx, vy = g % blk.groups_per_grid_row, g // blk.groups_per_grid_row
        group_cores.append([ttnn.CoreCoord(vx * k + m, vy) for m in range(k)])
    rows_used = _div_up(blk.num_groups, blk.groups_per_grid_row)
    ranges = []
    for vy in range(rows_used):
        n_groups_row = min(blk.groups_per_grid_row, blk.num_groups - vy * blk.groups_per_grid_row)
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, vy), ttnn.CoreCoord(n_groups_row * k - 1, vy)))
    core_ranges = ttnn.CoreRangeSet(ranges)

    # ---- circular buffers (identical list on every active core: peers address cb_gathered by our address) ----
    cb_dtype = {
        CB_X: x_dtype,
        CB_X_ALT: x_dtype,
        CB_GAMMA: g_dtype,
        CB_SUMSQ: ttnn.float32,
        CB_PARTIAL: ttnn.float32,
        CB_GATHERED: ttnn.float32,
        CB_INV_RMS: ttnn.float32,
        CB_X_RM: x_dtype,
        CB_OUT: x_dtype,
        CB_OUT_RM: x_dtype,
    }
    plan = cb_plan(sz, k, blk.slice_w_tiles, b, d)
    cbs = [_cb(idx, pages, page_bytes, cb_dtype[idx], core_ranges) for idx, (pages, page_bytes) in plan.items()]
    out_cb_pages = plan[CB_OUT][0]

    defines = []
    if is_rm:
        defines.append(("RMS_RM", "1"))
    if has_gamma:
        defines.append(("RMS_HAS_GAMMA", "1"))

    x_page = input_tensor.buffer_aligned_page_size()
    out_page = output_tensor.buffer_aligned_page_size()
    g_page = gamma.buffer_aligned_page_size() if has_gamma else 0

    reader_ct = [
        geo.tensor_w_tiles,
        geo.W,
        geo.H,
        geo.image_row_tiles,
        pitch,
        C,
        x_page,
        g_page,
        elem_bytes,
        g_elem_bytes,
        1 if gamma_is_tile else 0,
        d,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        (ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )

    writer_ct = [
        geo.tensor_w_tiles,
        geo.W,
        geo.H,
        geo.image_row_tiles,
        C,
        out_page,
        elem_bytes,
        k,
        blk.out_unit_tiles,
        SEM_PARTIALS_ARRIVED,
        out_cb_pages,
        WRITER_NOC_XY_ARG_BASE,
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct = [k, pitch, C, d, combine_lookahead]

    inv_w_bits = _f32_bits(1.0 / geo.W)
    eps_bits = _f32_bits(epsilon)
    x_addr = input_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    out_addr = output_tensor.buffer_address()

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    tw = geo.tensor_w_tiles
    for g, (row_start, core_rows) in enumerate(blk.group_rows):
        num_blocks = _div_up(core_rows, b)
        last_rows = core_rows - (num_blocks - 1) * b
        members = group_cores[g]
        noc_xy = []
        for c in members:
            v = device.worker_core_from_logical_core(c)
            noc_xy.extend([v.x, v.y])
        for m, c in enumerate(members):
            w_start = m * blk.slice_w_tiles
            core_w = min(blk.slice_w_tiles, tw - w_start)
            chunks_this = _div_up(core_w, C)
            reader_rt[c.x][c.y] = [x_addr, g_addr, row_start, num_blocks, b, last_rows, w_start, core_w, chunks_this]
            compute_rt[c.x][c.y] = [num_blocks, b, last_rows, core_w, chunks_this, inv_w_bits, eps_bits]
            writer_base = [out_addr, row_start, num_blocks, b, last_rows, w_start, core_w, chunks_this, m]
            assert len(writer_base) == WRITER_NOC_XY_ARG_BASE
            writer_rt[c.x][c.y] = writer_base + noc_xy

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        defines=defines,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        defines=defines,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        defines=defines,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )
    semaphores = []
    if k > 1:
        semaphores.append(ttnn.SemaphoreDescriptor(id=SEM_PARTIALS_ARRIVED, core_ranges=core_ranges, initial_value=0))
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)
