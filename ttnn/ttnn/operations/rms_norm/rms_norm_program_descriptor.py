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
# Kernel arg-layout contracts. Each mirrors a hardcoded offset in a kernel
# (McastArgs<CT, RT> / TensorAccessorArgs<CT>), which cannot import this module;
# the asserts below pin them so an added arg fails loudly at program build
# instead of silently shifting a peer's args.
# --------------------------------------------------------------------------
READER_ACCESSOR_CT_BASE = 6  # rms_norm_reader.cpp: TensorAccessorArgs<6>
WRITER_MCAST_CT_BASE = 5  # rms_norm_writer.cpp: MCAST_CT_BASE
WRITER_MCAST_RT_BASE = 15  # rms_norm_writer.cpp: MCAST_RT_BASE

# --------------------------------------------------------------------------
# Blocking / buffer-depth knobs — single source of truth.
# Each is a tunable parameter; every derived quantity below reads it.
# --------------------------------------------------------------------------
INPUT_CB_DEPTH = 2  # reader prefetches block b+1 while compute runs block b
OUTPUT_CB_DEPTH = 2  # writer drains tile-row r while compute produces r+1
RM_CB_DEPTH = 2  # overlaps stick reads/writes with tilize / untilize
L1_RESERVE_BYTES = 131072  # kernel binaries, stack, semaphores, allocator slack
MAX_GATHER_TILES = 64  # cap on block_row_tiles * w_group_size (cb_stat_gather)
# Perf lamp P2 — an upper bound on the cores per reduction group. Maximum
# occupancy is the default first step, but at tensor_row_tiles == 1 the selection
# pushes w_group_size to the whole grid, leaving 3-4 hidden tiles of real work per
# core against a full gather + multicast round. 0 = uncapped.
#
# MEASURED on blackhole_p150b (110-core grid), interleaved bf16 decode shapes,
# device kernel ns, uncapped -> capped at 32 (which admits G <= 22 on this grid):
#   (1,1,32,5120): 17676 -> 11088 ns (1.59x)   (1,1,32,7168): 18624 -> 12165 ns (1.53x)
#   (1,1,32,2304): 12056 ->  9677 ns (1.25x)   (1,1,32,1024):  8927 ->  8672 ns (1.03x)
# Prefill (tensor_row_tiles >> grid) is unaffected: it already selects G = 1..5.
MAX_W_GROUP_SIZE = 32
# Perf lamp P1 — the smallest number of blocks a core's row assignment is cut
# into. `input_cb_depth = 2` only buys read/compute overlap when there is a
# block b+1 for the reader to prefetch while compute runs block b; at
# num_blocks == 1 the DRAM read fully serializes against compute. Raising this
# trades combine rounds (one per block) for that overlap. 1 = coarsest block,
# no forced pipelining.
#
# MEASURED FLAT on blackhole_p150b (device kernel ns, 1 / 2 / 3 / 4 blocks):
#   (1,1,8192,1024) 104674 / 103894 / 106145 / 103801
#   (1,1,8192,2304) 219050 / 218364 / 222682 / 224441
#   (1,1,8192,5120) 425343 / 428822 / 425210 / 422477
#   (1,1,8192,7168) 605501 / 597240 / 595999 / 598579
#   (1,1,  32,7168)  12378 /  12423 /  12256 /  12208
# i.e. these shapes are not read-vs-compute serialization bound — the wall is
# aggregate DRAM bandwidth (the widest prefill case moves 33.5 MB in ~104 us =
# ~330 GB/s) plus the 3-vs-2 tile-row imbalance of 256 tile-rows over 110 cores.
# The knob is KEPT at its byte-identical default: it costs nothing there, and it
# is the lever to turn once a co-binding stage (compute, or the DRAM run length)
# is shortened. Follow-up: re-measure after any change that lowers the DRAM
# floor, and pair it with input_cb_depth 3-4 as op_design.md's lamp P1 suggests.
MIN_PIPELINE_BLOCKS = 1

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

        self.in_dtype = input_tensor.dtype
        self.in_elem_bytes = input_tensor.element_size()
        self.in_tile_bytes = ttnn.tile_size(input_tensor.dtype)

        self.has_gamma = gamma is not None
        if self.has_gamma:
            self.gamma_dtype = gamma.dtype
            self.gamma_elem_bytes = gamma.element_size()
            self.gamma_tile_bytes = ttnn.tile_size(gamma.dtype)
            self.is_rm_gamma = gamma.layout == ttnn.ROW_MAJOR_LAYOUT
        else:
            self.gamma_dtype = None
            self.gamma_elem_bytes = 0
            self.gamma_tile_bytes = 0
            self.is_rm_gamma = False


def _cb_specs(geo, C, G, R, is_rm_out, has_tail, input_cb_depth=INPUT_CB_DEPTH):
    """THE statement of this op's CB inventory — `(index, num_pages, page_size, format)`.

    Both the L1 residency solve (`_cb_bytes`) and the descriptor's CB list are
    built from this list, so a page-count change lands in exactly ONE place and
    the solve can never drift from what is actually allocated.

    Every `num_pages` here is affine in `R` (either constant or `k·R`), which is
    what lets `_cb_bytes` recover `fixed_bytes` / `per_row_bytes` exactly by
    differencing. Mirrors the CB table in `l1_ledger.md`.

    `input_cb_depth` is a parameter rather than a direct read of the module knob
    only so the descriptor can drop the prefetch buffer when the core's whole row
    assignment is a single block (there is then no block b+1 to prefetch). The
    solve always passes the knob itself, i.e. the conservative value.
    """
    T_in = geo.in_tile_bytes
    # The output CBs carry T_in / the input dtype: create_program_descriptor
    # asserts the output dtype matches the input's, so they are the same format.
    in_dtype = geo.in_dtype
    nc_max = 1 + has_tail  # cb_stat_sq columns per tile-row

    specs = [
        (CB_INPUT_TILES, input_cb_depth * R * C, T_in, in_dtype),
        (CB_SCALER, 1, BF16_TILE_BYTES, ttnn.bfloat16),
        (CB_ZERO_TILE, 1, FP32_TILE_BYTES, ttnn.float32),
        (CB_STAT_SQ, R * nc_max, FP32_TILE_BYTES, ttnn.float32),
        (CB_STAT_PARTIAL, R, FP32_TILE_BYTES, ttnn.float32),
        (CB_STAT_GATHER, R * G, FP32_TILE_BYTES, ttnn.float32),
        (CB_STAT_SUM, R, FP32_TILE_BYTES, ttnn.float32),
        (CB_RSTD_SEND, R, FP32_TILE_BYTES, ttnn.float32),
        (CB_RSTD, R, FP32_TILE_BYTES, ttnn.float32),
        # RM output needs the whole block resident before untilize runs (both are
        # compute-side, so they cannot pipeline); the tiled path streams.
        (CB_OUTPUT_TILES, (R * C) if is_rm_out else (OUTPUT_CB_DEPTH * C), T_in, in_dtype),
    ]
    if has_tail:
        specs.append((CB_WMASK, 1, BF16_TILE_BYTES, ttnn.bfloat16))
    if geo.is_rm_in:
        specs.append((CB_INPUT_RM, RM_CB_DEPTH * C, T_in, in_dtype))
    if is_rm_out:
        specs.append((CB_OUTPUT_RM, RM_CB_DEPTH * C, T_in, in_dtype))
    if geo.has_gamma:
        specs.append((CB_GAMMA_TILES, C, geo.gamma_tile_bytes, geo.gamma_dtype))
        specs.append((CB_NORMED, R * C, T_in, in_dtype))
        if geo.is_rm_gamma:
            specs.append((CB_GAMMA_RM, C, geo.gamma_tile_bytes, geo.gamma_dtype))
    return specs


def _cb_bytes(geo, C, G, is_rm_out, has_tail):
    """`(fixed_bytes, per_row_bytes)` of the per-core CB footprint at `(C, G)`.

    DERIVED from `_cb_specs` — never restated — by differencing the inventory at
    R = 1 and R = 2. Every page count there is affine in R, so
    `footprint(R) = fixed_bytes + R · per_row_bytes` holds exactly.
    """

    def total(R):
        return sum(pages * page_size for _, pages, page_size, _ in _cb_specs(geo, C, G, R, is_rm_out, has_tail))

    at_1, at_2 = total(1), total(2)
    per_row_bytes = at_2 - at_1
    return at_1 - per_row_bytes, per_row_bytes


def _max_block_row_tiles(geo, C, G, core_row_tiles, is_rm_out, has_tail, budget):
    """Closed-form L1 residency solve (a single expression, not a search).

    Returns 0 when even R == 1 does not fit.
    """
    fixed_bytes, per_row_bytes = _cb_bytes(geo, C, G, is_rm_out, has_tail)
    if fixed_bytes + per_row_bytes > budget:
        return 0
    cap = min(core_row_tiles, max(1, MAX_GATHER_TILES // G))
    if MIN_PIPELINE_BLOCKS > 1:
        # Perf lamp P1: cut the assignment into >= MIN_PIPELINE_BLOCKS blocks so
        # the depth-`input_cb_depth` input CB has a block to prefetch.
        cap = min(cap, max(1, _div_up(core_row_tiles, MIN_PIPELINE_BLOCKS)))
    return max(1, min((budget - fixed_bytes) // per_row_bytes, cap))


def _select_candidates(geo, grid_x, grid_y, is_rm_out, budget, w_group_cap):
    """Every (gc, gr) split that clears the mechanism caps and fits L1.

    Score, in priority order:
      1. active cores        — fill the grid;
      2. -G                  — fewest combine partners (each round is a gather
         barrier + a root reduce + a multicast);
      3. R                   — coarsest block, i.e. fewest combine rounds.

    A `-max_tiles_per_core` term was measured as a second key (prefer the split
    whose BUSIEST core carries the least work, e.g. G=2 x 80 tiles over G=1 x 96
    tiles at (8192, 1024) — both fill all 110 cores). It is NOT used: measured on
    blackhole_p150b it won (1,1,8192,2304) 220420 -> 198307 ns but lost
    (1,1,8192,1024) 102445 -> 123259 ns, i.e. the extra combine round and the
    halved DRAM run length outweigh the balance gain more often than not.
    """
    has_tail = 1 if geo.partial_w != 0 else 0
    out = []
    for gc in _divisors(grid_x):
        for gr in _divisors(grid_y):
            G = gc * gr
            num_groups = (grid_x // gc) * (grid_y // gr)
            if G > geo.tensor_w_tiles:
                continue  # mechanism cap: a core owning zero hidden tiles hangs the gather
            if w_group_cap and G > w_group_cap:
                continue  # perf lamp P2
            C = _div_up(geo.tensor_w_tiles, G)
            active_groups = min(geo.tensor_row_tiles, num_groups)
            core_row_tiles = _div_up(geo.tensor_row_tiles, active_groups)
            R = _max_block_row_tiles(geo, C, G, core_row_tiles, is_rm_out, has_tail, budget)
            if R == 0:
                continue
            score = (active_groups * G, -G, R)
            out.append((score, (gc, gr, C, R)))
    return out


def _select_regime(geo, grid_x, grid_y, is_rm_out, budget):
    """Exact, deterministic regime-selection function (op_design.md).

    Returns (w_group_cols, w_group_rows, core_w_tiles_ceil, block_row_tiles).

    MAX_W_GROUP_SIZE is a PREFERENCE, not a mechanism cap: if no capped candidate
    fits L1 (a hidden dim so wide that residency needs a large group), the search
    is retried uncapped rather than failing.
    """
    candidates = _select_candidates(geo, grid_x, grid_y, is_rm_out, budget, MAX_W_GROUP_SIZE)
    if not candidates and MAX_W_GROUP_SIZE:
        candidates = _select_candidates(geo, grid_x, grid_y, is_rm_out, budget, 0)
    if not candidates:
        raise RuntimeError(
            "rms_norm: no work split fits L1 for shape "
            f"{tuple(geo.shape)} (regime R3, streaming two-pass, is not implemented). "
            "Reduce the hidden dimension or use a larger grid."
        )
    return max(candidates, key=lambda c: c[0])[1]


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

    # The whole work split — tile-row numbering, per-core row ranges and the
    # row-major stick range every core drains — is derived ONCE, from the INPUT
    # tensor's layout (a TILE tensor pads each image's H to 32 independently; a
    # ROW_MAJOR one does not, so the two give different tile-row counts for the
    # same shape). The writer replays that mapping, so it is only valid while the
    # output shares the input's layout. The public entry point always allocates
    # the output that way; assert it here so a future caller that passes a
    # differently-laid-out output gets a loud failure instead of a silently
    # unwritten buffer.
    if output_tensor.layout != input_tensor.layout:
        raise ValueError(
            f"rms_norm: output layout ({output_tensor.layout}) must match the input layout "
            f"({input_tensor.layout}) — the per-core row mapping is derived from the input."
        )
    if output_tensor.dtype != input_tensor.dtype:
        raise ValueError(
            f"rms_norm: output dtype ({output_tensor.dtype}) must match the input dtype " f"({input_tensor.dtype})."
        )

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

    # Row split across the ACTIVE row-groups (every core of a group gets the
    # same row range — the combine is a group-wide barrier).
    rows_per_group = geo.tensor_row_tiles // active_row_groups
    rows_extra = geo.tensor_row_tiles % active_row_groups

    has_tail_global = 1 if geo.partial_w != 0 else 0

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
    # `C` (= cb_w_tiles) is the UNIFORM per-core block width along `hidden`: the
    # ceil of the ragged split. Every CB is allocated on the FULL active core set
    # at that width, for two reasons:
    #   1. the L1 map must be identical on every group member — cb_rstd is a
    #      multicast destination and cb_stat_gather a gather destination, both
    #      addressed by a peer's LOCAL pointer;
    #   2. a CB's page capacity must be an exact multiple of its push/pop
    #      quantum (dataflow_api.h:216-221 — "no other wrap is legal"), and a
    #      ragged per-core quantum does not divide a uniform capacity.
    # A core whose VALID slice is narrower (core_w < C) still moves C pages per
    # tile-row; the trailing pad tiles are never read by the statistics phases
    # (which walk `core_w` columns at row stride C) and their apply-phase output
    # is never written to DRAM.
    R = block_row_tiles
    C = core_w_tiles_ceil
    G = w_group_size

    # `input_cb_depth` buys ONE thing: the reader prefetching block b+1 while
    # compute runs block b. When the busiest core's whole row assignment is a
    # single block there is no block b+1, so the second buffer is dead L1 —
    # allocate depth 1 there instead of reserving a prefetch slot that provably
    # cannot be used. The residency SOLVE above deliberately still used the full
    # knob (the conservative value), so this only ever lowers the footprint and
    # never changes the chosen (G, C, R) geometry.
    max_row_count = rows_per_group + (1 if rows_extra else 0)
    input_cb_depth = INPUT_CB_DEPTH if max_row_count > R else 1

    cbs = [
        _cb(index, all_core_ranges, num_pages, page_size, data_format)
        for index, num_pages, page_size, data_format in _cb_specs(
            geo, C, G, R, is_rm_out, has_tail_global, input_cb_depth=input_cb_depth
        )
    ]

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
                core_w,
                owns_last_w_tile,
                num_sticks,
                stick_start,
                in_slice_bytes,
                in_byte_offset,
                gamma_slice_bytes,
                gamma_byte_offset,
            ]

            writer_own_args = [
                output_tensor.buffer_address(),
                row_start,
                num_blocks,
                R,
                last_block_row_tiles,
                w_start,
                core_w,
                slot,
                is_root,
                int(root_virtual.x),
                int(root_virtual.y),
                num_sticks,
                stick_start,
                out_slice_bytes,
                out_byte_offset,
            ]
            # The writer kernel reads the mcast runtime args from a hardcoded
            # offset (`MCAST_RT_BASE`), so the count of the writer's own args is
            # a contract with the kernel. Pin it here: appending an arg above
            # without bumping the kernel constant would otherwise silently feed
            # McastArgs garbage.
            assert len(writer_own_args) == WRITER_MCAST_RT_BASE, (
                f"rms_norm: writer runtime-arg layout drifted ({len(writer_own_args)} own args); "
                f"update MCAST_RT_BASE in kernels/rms_norm_writer.cpp and WRITER_MCAST_RT_BASE here"
            )
            writer_args[(cx, cy)] = writer_own_args + list(mc.runtime_args(ttnn.CoreCoord(cx, cy)))

            compute_args[(cx, cy)] = [
                num_blocks,
                R,
                last_block_row_tiles,
                core_w,
                has_tail,
                is_root,
            ]

    # ---- kernels ---------------------------------------------------------
    # `cb_w_tiles` (= C) is a COMPILE-TIME template parameter of tilize/untilize
    # and the stride of every block CB, so it is uniform across the whole active
    # grid. The ragged hidden remainder rides as the per-core RUNTIME arg
    # `core_w` (the valid slice width, core_w <= C).
    inv_w_bits = _f32_bits(1.0 / float(geo.W))
    eps_bits = _f32_bits(epsilon)

    def _rt(per_core):
        rt = ttnn.RuntimeArgs()
        for (cx, cy), vals in per_core.items():
            rt[cx][cy] = vals
        return rt

    reader_ct = [
        C,
        geo.tensor_w_tiles,
        1 if geo.is_rm_in else 0,
        1 if geo.has_gamma else 0,
        1 if geo.is_rm_gamma else 0,
        geo.partial_w,
    ]
    # Same contract as the writer's runtime args: the reader reads its
    # TensorAccessorArgs from a hardcoded CT offset.
    assert len(reader_ct) == READER_ACCESSOR_CT_BASE, (
        f"rms_norm: reader compile-time-arg layout drifted ({len(reader_ct)} own args); "
        f"update TensorAccessorArgs<...> in kernels/rms_norm_reader.cpp and READER_ACCESSOR_CT_BASE here"
    )
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if geo.has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    writer_ct = [
        C,
        geo.tensor_w_tiles,
        1 if is_rm_out else 0,
        G,
        SEM_GATHER,
    ]
    assert len(writer_ct) == WRITER_MCAST_CT_BASE, (
        f"rms_norm: writer compile-time-arg layout drifted ({len(writer_ct)} own args); "
        f"update MCAST_CT_BASE in kernels/rms_norm_writer.cpp and WRITER_MCAST_CT_BASE here"
    )
    writer_ct.extend(mcast_ct)
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct = [
        C,
        G,
        1 if geo.has_gamma else 0,
        1 if geo.is_rm_in else 0,
        1 if is_rm_out else 0,
        inv_w_bits,
        eps_bits,
        1 if geo.is_rm_gamma else 0,
    ]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
            core_ranges=all_core_ranges,
            compile_time_args=reader_ct,
            runtime_args=_rt(reader_args),
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
            core_ranges=all_core_ranges,
            compile_time_args=writer_ct,
            runtime_args=_rt(writer_args),
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
            core_ranges=all_core_ranges,
            compile_time_args=compute_ct,
            runtime_args=_rt(compute_args),
            config=compute_kernel_config,
        ),
    ]

    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
