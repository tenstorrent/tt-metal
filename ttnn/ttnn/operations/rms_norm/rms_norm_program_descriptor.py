# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — ProgramDescriptor.

Implements the Blocking Model of `op_design.md`:

  * two axes — `row` (independent, flattened leading dims in 32-row tile-rows)
    and `hidden` (dependent, the reduced last dim in 32-column tiles);
  * the grid is partitioned into `num_row_groups` axis-aligned rectangles of
    `w_group_cols x w_group_rows` cores.  Each rectangle is one reduction group:
    its members own disjoint hidden slices of the SAME rows and combine their
    partial sums-of-squares over the NoC (gather-to-root + multicast-back);
  * `block_row_tiles` (R) is the block extent along `row`, chosen by the closed-
    form L1 residency solve below.  `core_w_tiles` (C) is the block extent along
    `hidden` — the block always spans a core's whole hidden slice, so one block
    is exactly one cross-core combine round.

EVERY block factor / buffer depth / core assignment below is a named parameter,
defined once, with every CB page count and loop bound derived from it.  Nothing
is sized off a whole-op dimension.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"


# --------------------------------------------------------------------------
# Circular buffer indices (semantic names; the number is only a buffer slot)
# --------------------------------------------------------------------------
CB_INPUT_RM = 0
CB_INPUT_TILES = 1
CB_SCALER = 2
CB_WMASK = 3
CB_ZERO_TILE = 4
CB_STAT_SQ = 5
CB_STAT_PARTIAL = 7
CB_STAT_GATHER = 8
CB_STAT_SUM = 9
CB_RSTD_SEND = 10
CB_RSTD = 11
CB_GAMMA_RM = 12
CB_GAMMA_TILES = 13
CB_NORMED = 14
CB_OUTPUT_TILES = 16
CB_OUTPUT_RM = 17

# --------------------------------------------------------------------------
# Semaphores
# --------------------------------------------------------------------------
SEM_GATHER = 0  # members -> root: "my partial is in your gather buffer"
SEM_MCAST_READY = 1  # mcast data-ready flag (mcast_pipe)
SEM_MCAST_CONSUMED = 2  # mcast consumer-ready counter (mcast_pipe)

# --------------------------------------------------------------------------
# Blocking / buffer-depth knobs — single source of truth.
# Each is a tunable parameter; every derived quantity below reads it.
# --------------------------------------------------------------------------
INPUT_CB_DEPTH = 2  # reader prefetches block b+1 while compute runs block b
OUTPUT_CB_DEPTH = 2  # writer drains tile-row r while compute produces r+1
RM_CB_DEPTH = 2  # overlaps stick reads/writes with tilize / untilize
L1_RESERVE_BYTES = 131072  # kernel binaries, stack, semaphores, allocator slack
MAX_GATHER_TILES = 64  # cap on block_row_tiles * w_group_size (cb_stat_gather)

TILE_HW = 32
FP32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048

# L1 per Tensix core, by arch. Queried by name because the Python device object
# does not expose l1_size_per_core().
_L1_SIZE_BY_ARCH = {
    "grayskull": 1048576,
    "wormhole_b0": 1499136,
    "blackhole": 1572864,
}
_L1_SIZE_DEFAULT = 1499136


def _div_up(a, b):
    return (a + b - 1) // b


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _l1_cb_budget():
    try:
        arch = str(ttnn.get_arch_name()).lower()
    except Exception:  # pragma: no cover - defensive
        arch = ""
    return _L1_SIZE_BY_ARCH.get(arch, _L1_SIZE_DEFAULT) - L1_RESERVE_BYTES


# ==========================================================================
# Geometry + regime selection
# ==========================================================================


class _Geometry:
    """Alignment-aware tile geometry of the whole tensor. `floor` appears nowhere."""

    def __init__(self, input_tensor, gamma):
        shape = list(input_tensor.shape)
        self.shape = shape
        self.W = shape[-1]
        self.tensor_w_tiles = _div_up(self.W, TILE_HW)
        self.partial_w = self.W % TILE_HW

        self.is_rm_in = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
        if self.is_rm_in:
            # Sticks are contiguous across the leading dims: no per-image pad.
            self.num_sticks = 1
            for d in shape[:-1]:
                self.num_sticks *= d
            self.tensor_row_tiles = _div_up(self.num_sticks, TILE_HW)
        else:
            # A TILE tensor pads EACH image's H to 32 independently.
            images = 1
            for d in shape[:-2]:
                images *= d
            self.num_sticks = 0  # unused on the tiled path
            self.tensor_row_tiles = images * _div_up(shape[-2], TILE_HW)

        self.in_elem_bytes = input_tensor.element_size()
        self.in_tile_bytes = ttnn.tile_size(input_tensor.dtype)

        self.has_gamma = gamma is not None
        if self.has_gamma:
            self.gamma_elem_bytes = gamma.element_size()
            self.gamma_tile_bytes = ttnn.tile_size(gamma.dtype)
            self.is_rm_gamma = gamma.layout == ttnn.ROW_MAJOR_LAYOUT
        else:
            self.gamma_elem_bytes = 0
            self.gamma_tile_bytes = 0
            self.is_rm_gamma = False


def _cb_bytes(geo, C, G, R, is_rm_out, has_tail):
    """Per-core CB footprint at (C, G, R). Mirrors l1_ledger.md exactly.

    per_row_bytes scales with R; fixed_bytes does not. Both scale with C; only
    cb_stat_gather scales with G.
    """
    T_in = geo.in_tile_bytes
    T_g = geo.gamma_tile_bytes
    rm_in = 1 if geo.is_rm_in else 0
    rm_out = 1 if is_rm_out else 0
    rm_g = 1 if geo.is_rm_gamma else 0
    has_gamma = 1 if geo.has_gamma else 0
    nc_max = 1 + has_tail  # cb_stat_sq columns per tile-row

    # cb_input_tiles + cb_normed + (RM out) cb_output_tiles ... then the stat CBs:
    #   cb_stat_sq(nc_max) + cb_stat_partial + cb_stat_sum + cb_rstd_send + cb_rstd + cb_stat_gather(G)
    per_row_bytes = T_in * C * (INPUT_CB_DEPTH + has_gamma + rm_out) + FP32_TILE_BYTES * (4 + nc_max + G)

    fixed_bytes = (
        T_in * C * ((0 if rm_out else OUTPUT_CB_DEPTH) + rm_in * RM_CB_DEPTH + rm_out * RM_CB_DEPTH)
        + T_g * has_gamma * C * (1 + rm_g)
        + BF16_TILE_BYTES * (1 + has_tail)
        + FP32_TILE_BYTES
    )
    return fixed_bytes, per_row_bytes


def _max_block_row_tiles(geo, C, G, core_row_tiles, is_rm_out, has_tail, budget):
    """Closed-form L1 residency solve (a single expression, not a search).

    Returns 0 when even R == 1 does not fit.
    """
    fixed_bytes, per_row_bytes = _cb_bytes(geo, C, G, 1, is_rm_out, has_tail)
    if fixed_bytes + per_row_bytes > budget:
        return 0
    cap = min(core_row_tiles, max(1, MAX_GATHER_TILES // G))
    return max(1, min((budget - fixed_bytes) // per_row_bytes, cap))


def _select_regime(geo, grid_x, grid_y, is_rm_out, budget):
    """Exact, deterministic regime-selection function (op_design.md).

    Returns (w_group_cols, w_group_rows, core_w_tiles_ceil, block_row_tiles).
    """
    has_tail = 1 if geo.partial_w != 0 else 0
    best = None
    for gc in _divisors(grid_x):
        for gr in _divisors(grid_y):
            G = gc * gr
            num_groups = (grid_x // gc) * (grid_y // gr)
            if G > geo.tensor_w_tiles:
                continue  # mechanism cap: a core owning zero hidden tiles hangs the gather
            C = _div_up(geo.tensor_w_tiles, G)
            active_groups = min(geo.tensor_row_tiles, num_groups)
            core_row_tiles = _div_up(geo.tensor_row_tiles, active_groups)
            R = _max_block_row_tiles(geo, C, G, core_row_tiles, is_rm_out, has_tail, budget)
            if R == 0:
                continue
            score = (active_groups * G, -G, R)
            if best is None or score > best[0]:
                best = (score, (gc, gr, C, R))
    if best is None:
        raise RuntimeError(
            "rms_norm: no work split fits L1 for shape "
            f"{tuple(geo.shape)} (regime R3, streaming two-pass, is not implemented). "
            "Reduce the hidden dimension or use a larger grid."
        )
    return best[1]


# ==========================================================================
# Program descriptor
# ==========================================================================


def _cb(index, core_ranges, num_pages, page_size, data_format):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=index,
                data_format=data_format,
                page_size=page_size,
            )
        ],
    )


def _core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def create_program_descriptor(
    input_tensor,
    gamma,
    output_tensor,
    *,
    epsilon,
    compute_kernel_config,
):
    device = input_tensor.device()
    geo = _Geometry(input_tensor, gamma)

    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)

    is_rm_out = output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    budget = _l1_cb_budget()

    # ---- block factors ---------------------------------------------------
    w_group_cols, w_group_rows, core_w_tiles_ceil, block_row_tiles = _select_regime(
        geo, grid_x, grid_y, is_rm_out, budget
    )
    w_group_size = w_group_cols * w_group_rows
    num_row_groups = (grid_x // w_group_cols) * (grid_y // w_group_rows)
    active_row_groups = min(geo.tensor_row_tiles, num_row_groups)

    # Hidden split inside a group: `rem` cores take ceil, the rest floor.
    w_floor = geo.tensor_w_tiles // w_group_size
    w_rem = geo.tensor_w_tiles % w_group_size
    core_w_tiles_floor = w_floor if w_rem else core_w_tiles_ceil

    # Row split across the ACTIVE row-groups (every core of a group gets the
    # same row range — the combine is a group-wide barrier).
    rows_per_group = geo.tensor_row_tiles // active_row_groups
    rows_extra = geo.tensor_row_tiles % active_row_groups

    has_tail_global = 1 if geo.partial_w != 0 else 0
    nc_max = 1 + has_tail_global

    # ---- per-core layout -------------------------------------------------
    groups = []
    all_cores = []
    for gi in range(active_row_groups):
        groups_across = grid_x // w_group_cols
        gx0 = (gi % groups_across) * w_group_cols
        gy0 = (gi // groups_across) * w_group_rows
        cores = [(gx0 + x, gy0 + y) for y in range(w_group_rows) for x in range(w_group_cols)]
        rect = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(gx0, gy0),
                    ttnn.CoreCoord(gx0 + w_group_cols - 1, gy0 + w_group_rows - 1),
                )
            ]
        )
        row_start = gi * rows_per_group + min(gi, rows_extra)
        row_count = rows_per_group + (1 if gi < rows_extra else 0)
        groups.append(
            {
                "cores": cores,
                "rect": rect,
                "root": ttnn.CoreCoord(*cores[-1]),
                "row_start": row_start,
                "row_count": row_count,
            }
        )
        all_cores.extend(cores)

    all_core_ranges = _core_range_set(all_cores)

    # mcast wiring — one Mcast2D per reduction group, all adopting the same
    # semaphore ids so the CT block is uniform across groups.
    mcast_cfg = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_1,  # the writer kernel runs on NoC1
        sem_ids=[SEM_MCAST_READY, SEM_MCAST_CONSUMED],
    )
    mcasts = [ttnn.Mcast2D(device, g["rect"], g["root"], mcast_cfg) for g in groups]
    mcast_ct = list(mcasts[0].compile_time_args())

    # ---- circular buffers ------------------------------------------------
    # Every CB is allocated on the FULL active core set at the `ceil` width, so
    # the L1 map is identical on every group member. That is required by both
    # the multicast destination (cb_rstd) and the gather destination
    # (cb_stat_gather), which are addressed by a peer's local pointer.
    R = block_row_tiles
    C = core_w_tiles_ceil
    G = w_group_size

    in_dtype = input_tensor.dtype
    in_tile = geo.in_tile_bytes
    out_dtype = output_tensor.dtype
    out_tile = ttnn.tile_size(out_dtype)

    cbs = [
        _cb(CB_INPUT_TILES, all_core_ranges, INPUT_CB_DEPTH * R * C, in_tile, in_dtype),
        _cb(CB_SCALER, all_core_ranges, 1, BF16_TILE_BYTES, ttnn.bfloat16),
        _cb(CB_ZERO_TILE, all_core_ranges, 1, FP32_TILE_BYTES, ttnn.float32),
        _cb(CB_STAT_SQ, all_core_ranges, R * nc_max, FP32_TILE_BYTES, ttnn.float32),
        _cb(CB_STAT_PARTIAL, all_core_ranges, R, FP32_TILE_BYTES, ttnn.float32),
        _cb(CB_STAT_GATHER, all_core_ranges, R * G, FP32_TILE_BYTES, ttnn.float32),
        _cb(CB_STAT_SUM, all_core_ranges, R, FP32_TILE_BYTES, ttnn.float32),
        _cb(CB_RSTD_SEND, all_core_ranges, R, FP32_TILE_BYTES, ttnn.float32),
        _cb(CB_RSTD, all_core_ranges, R, FP32_TILE_BYTES, ttnn.float32),
        # RM output needs the whole block resident before untilize runs (both are
        # compute-side, so they cannot pipeline); the tiled path streams.
        _cb(
            CB_OUTPUT_TILES,
            all_core_ranges,
            (R * C) if is_rm_out else (OUTPUT_CB_DEPTH * C),
            out_tile,
            out_dtype,
        ),
    ]
    if has_tail_global:
        cbs.append(_cb(CB_WMASK, all_core_ranges, 1, BF16_TILE_BYTES, ttnn.bfloat16))
    if geo.is_rm_in:
        cbs.append(_cb(CB_INPUT_RM, all_core_ranges, RM_CB_DEPTH * C, in_tile, in_dtype))
    if is_rm_out:
        cbs.append(_cb(CB_OUTPUT_RM, all_core_ranges, RM_CB_DEPTH * C, out_tile, out_dtype))
    if geo.has_gamma:
        cbs.append(_cb(CB_GAMMA_TILES, all_core_ranges, C, geo.gamma_tile_bytes, gamma.dtype))
        cbs.append(_cb(CB_NORMED, all_core_ranges, R * C, in_tile, in_dtype))
        if geo.is_rm_gamma:
            cbs.append(_cb(CB_GAMMA_RM, all_core_ranges, C, geo.gamma_tile_bytes, gamma.dtype))

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_READY, core_ranges=all_core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_MCAST_CONSUMED, core_ranges=all_core_ranges, initial_value=0),
    ]

    # ---- per-core runtime args ------------------------------------------
    # Collected per core first; the RuntimeArgs objects are built per kernel
    # core-range below so a descriptor only ever carries args for its own cores.
    reader_args = {}
    writer_args = {}
    compute_args = {}

    cores_by_width = {core_w_tiles_ceil: [], core_w_tiles_floor: []}

    for gi, g in enumerate(groups):
        mc = mcasts[gi]
        root_virtual = device.worker_core_from_logical_core(g["root"])
        row_start = g["row_start"]
        row_count = g["row_count"]
        num_blocks = _div_up(row_count, R)
        last_block_row_tiles = row_count - (num_blocks - 1) * R

        for slot, (cx, cy) in enumerate(g["cores"]):
            if slot < w_rem:
                core_w = w_floor + 1
                w_start = slot * (w_floor + 1)
            else:
                core_w = w_floor
                w_start = w_rem * (w_floor + 1) + (slot - w_rem) * w_floor
            cores_by_width[core_w].append((cx, cy))

            owns_last_w_tile = 1 if (w_start + core_w) == geo.tensor_w_tiles else 0
            has_tail = 1 if (owns_last_w_tile and geo.partial_w) else 0
            is_root = 1 if (cx, cy) == (int(g["root"].x), int(g["root"].y)) else 0

            # This core's hidden slice, in bytes, on the input/output row.
            w_elems = min(core_w * TILE_HW, geo.W - w_start * TILE_HW)
            in_slice_bytes = w_elems * geo.in_elem_bytes
            in_byte_offset = w_start * TILE_HW * geo.in_elem_bytes
            gamma_slice_bytes = w_elems * geo.gamma_elem_bytes
            gamma_byte_offset = w_start * TILE_HW * geo.gamma_elem_bytes
            out_slice_bytes = w_elems * output_tensor.element_size()
            out_byte_offset = w_start * TILE_HW * output_tensor.element_size()

            if geo.is_rm_in:
                stick_start = row_start * TILE_HW
                stick_end = min(geo.num_sticks, (row_start + row_count) * TILE_HW)
                num_sticks = max(0, stick_end - stick_start)
            else:
                stick_start = 0
                num_sticks = 0

            reader_args[(cx, cy)] = [
                input_tensor.buffer_address(),
                gamma.buffer_address() if geo.has_gamma else 0,
                row_start,
                num_blocks,
                R,
                last_block_row_tiles,
                w_start,
                owns_last_w_tile,
                num_sticks,
                stick_start,
                in_slice_bytes,
                in_byte_offset,
                gamma_slice_bytes,
                gamma_byte_offset,
            ]

            writer_args[(cx, cy)] = [
                output_tensor.buffer_address(),
                row_start,
                num_blocks,
                R,
                last_block_row_tiles,
                w_start,
                slot,
                is_root,
                int(root_virtual.x),
                int(root_virtual.y),
                num_sticks,
                stick_start,
                out_slice_bytes,
                out_byte_offset,
            ] + list(mc.runtime_args(ttnn.CoreCoord(cx, cy)))

            compute_args[(cx, cy)] = [
                num_blocks,
                R,
                last_block_row_tiles,
                has_tail,
                is_root,
            ]

    # ---- kernels ---------------------------------------------------------
    # `core_w_tiles` is a COMPILE-TIME template parameter of tilize/untilize and
    # of the CB page-count expressions, so the ragged hidden remainder is
    # expressed as two kernel core-ranges with separate CT blocks.
    inv_w_bits = _f32_bits(1.0 / float(geo.W))
    eps_bits = _f32_bits(epsilon)

    def _rt(per_core, cores):
        rt = ttnn.RuntimeArgs()
        for cx, cy in cores:
            rt[cx][cy] = per_core[(cx, cy)]
        return rt

    kernels = []
    for core_w, cores in cores_by_width.items():
        if not cores:
            continue
        crs = _core_range_set(cores)
        reader_rt = _rt(reader_args, cores)
        writer_rt = _rt(writer_args, cores)
        compute_rt = _rt(compute_args, cores)

        reader_ct = [
            core_w,
            geo.tensor_w_tiles,
            1 if geo.is_rm_in else 0,
            1 if geo.has_gamma else 0,
            1 if geo.is_rm_gamma else 0,
            geo.partial_w,
            geo.in_elem_bytes,
            geo.gamma_elem_bytes,
        ]
        reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
        reader_ct.extend(
            ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
            if geo.has_gamma
            else ttnn.TensorAccessorArgs().get_compile_time_args()
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
                core_ranges=crs,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        )

        writer_ct = [
            core_w,
            geo.tensor_w_tiles,
            1 if is_rm_out else 0,
            G,
            SEM_GATHER,
        ]
        writer_ct.extend(mcast_ct)
        writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
                core_ranges=crs,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            )
        )

        compute_ct = [
            core_w,
            G,
            1 if geo.has_gamma else 0,
            1 if geo.is_rm_in else 0,
            1 if is_rm_out else 0,
            inv_w_bits,
            eps_bits,
            1 if geo.is_rm_gamma else 0,
        ]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
                core_ranges=crs,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_kernel_config,
            )
        )

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
