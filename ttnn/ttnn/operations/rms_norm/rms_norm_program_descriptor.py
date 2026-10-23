# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor: blocking plan, CBs, kernels, runtime args (op_design.md).

Single source of truth for every block knob: `_select_blocking(...) -> BlockingPlan`.  Every CB page count,
loop trip count, core range and kernel argument below is derived from the plan's fields.

Regimes (op_design.md §Regimes):
  R1 row_local            (w_group_size == 1): each core owns whole rows; the row's W slice is resident.
  R2 w_split_root_combine (w_group_size >= 2): W split across a consecutive row-major group of cores; per block
                          every member sends its per-row partial sum of squares to the group root, the root
                          finalizes rsqrt(S/W + eps) and multicasts it back to the group.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
SCALER_TILE_BYTES = 2048  # bf16 scaler tile (reduce LLK scaler format)

# Selector constants (op_design.md §Selection function; perf lamps L2 / L3).
CHUNK_FLOOR_TILES = 8
MIN_W_TILES_PER_CORE = 8
X_DEPTH_BLOCKS = 1  # cb_x_tiles depth in blocks (perf lamp L1)

# CB indices (semantic names; op_design.md §Circular Buffers).
CB_X_RM_STICKS = 0
CB_X_TILES = 1
CB_GAMMA_RM_STICK = 2
CB_GAMMA_TILES = 3
CB_REDUCE_SCALER = 4
CB_SUMSQ = 5
CB_ROWSUM_PARTIAL = 6
CB_ROWSUM_GATHER = 7
CB_INV_RMS_ROOT = 8
CB_INV_RMS = 9
CB_X_GAMMA = 10
CB_OUT_TILES = 11
CB_OUT_RM_STICKS = 12

# Data-movement placement: reader on RISCV_1 / NOC_0, writer on RISCV_0 / NOC_1 (the ReaderConfig / WriterConfig
# defaults, spelled out because the inverse-RMS multicast is issued by the writer and must use the writer's NoC).
READER_NOC = ttnn.NOC.NOC_0
WRITER_NOC = ttnn.NOC.NOC_1

SEM_GATHER_ARRIVALS = 0
SEM_INV_MCAST_FIRST = 1  # Mcast attach allocates its semaphores from this id on


# ---------------------------------------------------------------------------------------------------------------------
# Geometry + blocking plan
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    tensor_w_elems: int
    tensor_w_tiles: int
    last_tile_valid_cols: int
    tensor_rows_tiles: int
    tensor_rows_sticks: int  # RM only (0 for TILE)
    is_rm: bool
    in_tile_bytes: int
    out_tile_bytes: int
    in_elem_bytes: int
    out_elem_bytes: int
    has_gamma: bool
    gamma_is_rm: bool
    gamma_tile_bytes: int
    gamma_elem_bytes: int
    gamma_is_block_float: bool  # bfloat8_b TILE gamma: the page is shared exponents + mantissas, read whole
    # DEST-produced intermediates (cb_sumsq, cb_rowsum_*, cb_inv_rms*, cb_x_gamma) follow DEST width: Float32 pages
    # under fp32_dest_acc_en=True, Float16_b pages under False (a 16-bit DEST value has no extra bits to keep).
    interm_dtype: object
    interm_tile_bytes: int


@dataclass(frozen=True)
class BlockingPlan:
    w_group_size: int  # gw: cores per W-group (1 => R1)
    core_w_tiles: int  # n: W slice per core (tiles)
    num_row_groups: int  # gr
    rows_hi: int  # ceil(R / gr)
    block_rows: int  # b: tile-rows per block
    chunk_w_tiles: int  # c: streaming quantum along W (tiles)
    chunks_per_row: int  # k
    xg_lookahead_chunks: int  # L
    x_depth_blocks: int = X_DEPTH_BLOCKS

    @property
    def row_stride_tiles(self):  # s = k * c
        return self.chunks_per_row * self.chunk_w_tiles

    @property
    def is_w_split(self):
        return self.w_group_size >= 2

    @property
    def used_cores(self):
        return self.num_row_groups * self.w_group_size


def _round_up(v, m):
    return ((v + m - 1) // m) * m


# CB page-format roles, resolved to a concrete dtype in _build_cbs.
_FMT_IN, _FMT_OUT, _FMT_GAMMA, _FMT_SCALER, _FMT_INTERM = "in", "out", "gamma", "scaler", "interm"


def _cb_specs(geo: Geometry, b, n, c, k, L, gw, x_depth):
    """THE single definition of every CB: [(index, num_pages, page_bytes, format_role)].

    Both the L1 solve (_footprint) and the allocation (_build_cbs) read this list, so a block / chunk / depth knob
    changes the budget and the real CB sizes in exactly one place (l1_ledger.md §Ledger mirrors it).
    """
    s = k * c
    out_depth = 1 if geo.is_rm else 2  # RM: normalize -> untilize are sequential on compute (no overlap to buy)
    stream_depth = 2  # reader / writer run one chunk ahead of compute
    specs = []
    if geo.is_rm:
        specs.append((CB_X_RM_STICKS, stream_depth * c, geo.in_tile_bytes, _FMT_IN))
    specs.append((CB_X_TILES, x_depth * b * s, geo.in_tile_bytes, _FMT_IN))
    if geo.has_gamma and geo.gamma_is_rm:
        stick_bytes = _round_up(n * TILE * geo.gamma_elem_bytes, ttnn.get_dram_alignment())
        specs.append((CB_GAMMA_RM_STICK, 1, stick_bytes, _FMT_GAMMA))
    if geo.has_gamma:
        specs.append((CB_GAMMA_TILES, s, geo.gamma_tile_bytes, _FMT_GAMMA))
    specs.append((CB_REDUCE_SCALER, 2, SCALER_TILE_BYTES, _FMT_SCALER))
    specs.append((CB_SUMSQ, 2 * b, geo.interm_tile_bytes, _FMT_INTERM))
    if gw >= 2:
        specs.append((CB_ROWSUM_PARTIAL, b, geo.interm_tile_bytes, _FMT_INTERM))
        specs.append((CB_ROWSUM_GATHER, gw * b, geo.interm_tile_bytes, _FMT_INTERM))
        specs.append((CB_INV_RMS_ROOT, b, geo.interm_tile_bytes, _FMT_INTERM))
    specs.append((CB_INV_RMS, b, geo.interm_tile_bytes, _FMT_INTERM))
    if geo.has_gamma:
        specs.append((CB_X_GAMMA, L * b * c, geo.interm_tile_bytes, _FMT_INTERM))
    specs.append((CB_OUT_TILES, out_depth * b * c, geo.out_tile_bytes, _FMT_OUT))
    if geo.is_rm:
        specs.append((CB_OUT_RM_STICKS, stream_depth * b * c, geo.out_tile_bytes, _FMT_OUT))
    return specs


def _footprint(geo: Geometry, b, n, c, k, L, gw, x_depth=X_DEPTH_BLOCKS):
    """Per-core CB bytes — the closed form of l1_ledger.md §Footprint, summed from _cb_specs."""
    return sum(pages * page_bytes for _, pages, page_bytes, _ in _cb_specs(geo, b, n, c, k, L, gw, x_depth))


def _select_blocking(geo: Geometry, num_cores_available: int, l1_budget_bytes: int) -> BlockingPlan:
    C, R, Wt = num_cores_available, geo.tensor_rows_tiles, geo.tensor_w_tiles
    if R < C and C // R >= 2 and Wt >= 2 * MIN_W_TILES_PER_CORE:
        gw_fill = min(C // R, Wt // MIN_W_TILES_PER_CORE)
    else:
        gw_fill = 1
    seen = set()
    for gw_try in range(max(gw_fill, 1), C + 1):  # smallest W-group first: fewest combines
        n = -(-Wt // gw_try)
        gw = -(-Wt // n)  # M2: no empty slices
        if gw in seen:
            continue
        seen.add(gw)
        gr = min(R, C // gw)
        rows_hi = -(-R // gr)
        k = 1
        while True:  # largest balanced chunk first
            chunk = -(-n // k)
            if chunk < min(n, CHUNK_FLOOR_TILES):
                break
            for L in [k, 1] if gw >= 2 else [1]:  # R2: whole-block x*gamma lookahead first (stall fill S1)
                if _footprint(geo, 1, n, chunk, k, L, gw) <= l1_budget_bytes:
                    if k > 1:
                        block_rows = 1  # M5: a chunk of a multi-row block is not FIFO-contiguous
                    else:
                        block_rows = 1
                        for b in range(rows_hi, 0, -1):
                            if _footprint(geo, b, n, chunk, k, L, gw) <= l1_budget_bytes:
                                block_rows = b
                                break
                    return BlockingPlan(
                        w_group_size=gw,
                        core_w_tiles=n,
                        num_row_groups=gr,
                        rows_hi=rows_hi,
                        block_rows=block_rows,
                        chunk_w_tiles=chunk,
                        chunks_per_row=k,
                        xg_lookahead_chunks=L,
                    )
            k += 1
    raise RuntimeError("rms_norm: W too wide for the grid")  # M3


_BLOCK_FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat4_b)


def _elem_bytes(tensor):
    """Bytes per element, consumed only by the ROW_MAJOR stick paths and the TILE-gamma face-row reads.

    A block-float tensor (TILE-only, shared exponents) has no per-element size and never takes those paths, so 0.
    """
    return 0 if tensor.dtype in _BLOCK_FLOAT_DTYPES else tensor.element_size()


def _geometry(input_tensor, output_tensor, gamma, fp32_dest_acc_en) -> Geometry:
    shape = list(input_tensor.shape)
    W = shape[-1]
    H = shape[-2]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    w_tiles = -(-W // TILE)
    if is_rm:
        rows_sticks = lead * H
        rows_tiles = -(-rows_sticks // TILE)
    else:
        rows_sticks = 0
        rows_tiles = lead * (-(-H // TILE))
    has_gamma = gamma is not None
    interm_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    return Geometry(
        tensor_w_elems=W,
        tensor_w_tiles=w_tiles,
        last_tile_valid_cols=W - TILE * (w_tiles - 1),
        tensor_rows_tiles=rows_tiles,
        tensor_rows_sticks=rows_sticks,
        is_rm=is_rm,
        in_tile_bytes=ttnn.tile_size(input_tensor.dtype),
        out_tile_bytes=ttnn.tile_size(output_tensor.dtype),
        in_elem_bytes=_elem_bytes(input_tensor),
        out_elem_bytes=_elem_bytes(output_tensor),
        has_gamma=has_gamma,
        gamma_is_rm=has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT,
        gamma_tile_bytes=ttnn.tile_size(gamma.dtype) if has_gamma else 0,
        gamma_elem_bytes=_elem_bytes(gamma) if has_gamma else 0,
        gamma_is_block_float=has_gamma and gamma.dtype in _BLOCK_FLOAT_DTYPES,
        interm_dtype=interm_dtype,
        interm_tile_bytes=ttnn.tile_size(interm_dtype),
    )


def _l1_budget_bytes(device):
    """Worker-L1 bytes available to CBs: from the allocator base up to the lowest allocated L1 buffer.

    CBs grow up from the L1 allocator base (``GetAllocatorBaseAddress``) while L1 buffers (including any
    L1-interleaved input / output / gamma) are allocated top-down, so the budget is the free run at the bottom of
    the bank as reported by the allocator's memory view. No safety fraction.
    """
    view = ttnn._ttnn.device.GetMemoryView(device, ttnn.BufferType.L1)
    budget = view.total_bytes_per_bank
    for block in view.block_table:
        if block["allocated"] == "yes":
            budget = min(budget, int(block["address"]))
    return budget


# ---------------------------------------------------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------------------------------------------------


def _cb(index, core_ranges, num_pages, page_bytes, dtype):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def _build_cbs(geo: Geometry, plan: BlockingPlan, cores, in_dtype, out_dtype, gamma_dtype):
    """Every CB declared once over all used cores at one nominal size (M1), from the same _cb_specs as the solve."""
    fmt = {
        _FMT_IN: in_dtype,
        _FMT_OUT: out_dtype,
        _FMT_GAMMA: gamma_dtype,
        _FMT_SCALER: ttnn.bfloat16,
        _FMT_INTERM: geo.interm_dtype,
    }
    specs = _cb_specs(
        geo,
        plan.block_rows,
        plan.core_w_tiles,
        plan.chunk_w_tiles,
        plan.chunks_per_row,
        plan.xg_lookahead_chunks,
        plan.w_group_size,
        plan.x_depth_blocks,
    )
    return [_cb(index, cores, pages, page_bytes, fmt[role]) for index, pages, page_bytes, role in specs]


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def create_program_descriptor(input_tensor, output_tensor, *, gamma, epsilon, compute_kernel_config):
    device = input_tensor.device()
    geo = _geometry(input_tensor, output_tensor, gamma, bool(compute_kernel_config.fp32_dest_acc_en))
    grid = device.compute_with_storage_grid_size()
    num_cores_available = grid.x * grid.y
    plan = _select_blocking(geo, num_cores_available, _l1_budget_bytes(device))

    gw, gr = plan.w_group_size, plan.num_row_groups
    used = plan.used_cores
    cores = ttnn.num_cores_to_corerangeset(used, grid, True)
    core_list = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(used)]

    # Row split over row groups: split_work_to_cores semantics (two classes).
    q, rem = divmod(geo.tensor_rows_tiles, gr)

    def group_rows(gi):
        rows = q + 1 if gi < rem else q
        start = gi * q + min(gi, rem)
        return start, rows

    cbs = _build_cbs(
        geo,
        plan,
        cores,
        input_tensor.dtype,
        output_tensor.dtype,
        gamma.dtype if gamma is not None else ttnn.bfloat16,
    )

    # ---------------- reader ----------------
    reader_ct = [
        CB_X_RM_STICKS,
        CB_X_TILES,
        CB_REDUCE_SCALER,
        int(geo.is_rm),
        plan.block_rows,
        plan.chunk_w_tiles,
        plan.chunks_per_row,
        geo.in_elem_bytes,
        geo.tensor_w_tiles,
        geo.tensor_w_elems,
        geo.tensor_rows_sticks,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # ---------------- writer ----------------
    # TILE gamma: row 0 of faces 0 / 1 by byte offset (16 elements per face row) — or, for a bfloat8_b page whose
    # shared exponents precede the mantissas, the whole page (the Row broadcast still consumes only row 0).
    face_row_path = geo.has_gamma and not geo.gamma_is_rm and not geo.gamma_is_block_float
    gamma_face_row_read_bytes = _round_up(16 * geo.gamma_elem_bytes, ttnn.get_dram_alignment()) if face_row_path else 0
    writer_ct = [
        CB_GAMMA_RM_STICK,
        CB_GAMMA_TILES,
        CB_ROWSUM_PARTIAL,
        CB_ROWSUM_GATHER,
        CB_INV_RMS_ROOT,
        CB_INV_RMS,
        CB_OUT_TILES,
        CB_OUT_RM_STICKS,
        int(geo.is_rm),
        int(geo.has_gamma),
        int(geo.gamma_is_rm),
        int(plan.is_w_split),
        plan.block_rows,
        plan.chunk_w_tiles,
        plan.chunks_per_row,
        gw,
        geo.out_elem_bytes,
        geo.gamma_elem_bytes,
        geo.tensor_w_tiles,
        geo.tensor_w_elems,
        geo.tensor_rows_sticks,
        SEM_GATHER_ARRIVALS,
        gamma_face_row_read_bytes,
        int(geo.gamma_is_block_float),
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # ---------------- compute ----------------
    compute_ct = [
        CB_X_RM_STICKS,
        CB_X_TILES,
        CB_GAMMA_TILES,
        CB_REDUCE_SCALER,
        CB_SUMSQ,
        CB_ROWSUM_PARTIAL,
        CB_ROWSUM_GATHER,
        CB_INV_RMS_ROOT,
        CB_INV_RMS,
        CB_X_GAMMA,
        CB_OUT_TILES,
        CB_OUT_RM_STICKS,
        int(geo.is_rm),
        int(geo.has_gamma),
        int(plan.is_w_split),
        plan.block_rows,
        plan.chunk_w_tiles,
        plan.chunks_per_row,
        plan.xg_lookahead_chunks,
        gw,
    ]

    inv_w_bits = _f32_bits(1.0 / geo.tensor_w_elems)
    eps_bits = _f32_bits(epsilon)

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for i, core in enumerate(core_list):
        gi, g = divmod(i, gw)
        row_start, rows = group_rows(gi)
        slice_start = g * plan.core_w_tiles
        slice_w = min(plan.core_w_tiles, geo.tensor_w_tiles - slice_start)
        owns_last_col = slice_start + slice_w == geo.tensor_w_tiles
        last_valid = geo.last_tile_valid_cols if owns_last_col else TILE
        num_blocks = -(-rows // plan.block_rows)
        root = core_list[gi * gw]
        root_virtual = device.worker_core_from_logical_core(root)

        reader_rt[core.x][core.y] = [input_tensor.buffer_address(), row_start, rows, slice_start, slice_w, last_valid]
        writer_rt[core.x][core.y] = [
            output_tensor.buffer_address(),
            gamma.buffer_address() if gamma is not None else 0,
            row_start,
            rows,
            slice_start,
            slice_w,
            g,
            root_virtual.x,
            root_virtual.y,
        ]
        compute_rt[core.x][core.y] = [num_blocks, slice_w, int(g == 0), inv_w_bits, eps_bits]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=READER_NOC),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=WRITER_NOC),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_GATHER_ARRIVALS, core_ranges=cores, initial_value=0)]
    descriptor = ttnn.ProgramDescriptor(semaphores=semaphores, cbs=cbs)
    if plan.is_w_split:
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(noc=WRITER_NOC, handshake=False),
            cores,
            gw,
            ttnn.McastFixedSenderConfig(sender_index=0),
            ttnn.McastCoreOrder.RowMajor,
        )
        mcast.attach(descriptor, "inv_mcast", [writer], SEM_INV_MCAST_FIRST)
    else:
        ttnn.attach_absent_mcast(writer, "inv_mcast")
    descriptor.kernels = [reader, writer, compute]
    return descriptor
