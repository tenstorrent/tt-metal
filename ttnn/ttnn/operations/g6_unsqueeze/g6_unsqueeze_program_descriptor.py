# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_unsqueeze program descriptor: plan selection (the single source of every knob) and the
ProgramDescriptor built from it.

Topology (op_design.md, Blocking Model): `w_tile` split across cores (contiguous tile range per
core, `split_work_to_cores(row_wise=True)` over the
most-square single-rectangle core pool from `_core_pool`), `tile_half` split across the two data-movement RISCs of
each core (HALVES_PER_RISC = 1). No compute kernel, no cross-core traffic. Each RISC owns a private
staging CB (`cb_face_stage_half{h}`) sized `buffer_depth * block_w_tiles * halves_per_risc` entries
(entry size set by WRITE_SHAPE) + 64 B alignment slack.
"""

from __future__ import annotations

from collections import OrderedDict, namedtuple
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- Module-level knobs (each defined once; read at call time so tests can pin them) ----
L1_BUDGET_BYTES = 1347776  # L1 left to the op per core by the model (BB-05)
MAX_CORES = None  # None = whole compute grid
MAX_BLOCK_W_TILES = None  # None = only the L1 cap bounds the block
HALVES_PER_RISC = 1  # 1: half 0 on the reader-config RISC, half 1 on the writer-config RISC; 2: one RISC
# Output write transfer shape (Refinement 2): "segment" 16 x 16 B column-0 writes per half (stateful
# one-packet issue), "face" one 1024 B face per half, "tile" one 4096 B tile per tile (forces
# HALVES_PER_RISC = 2). Kernel-side enum: WRITE_SHAPE_{SEGMENT,FACE,TILE} in g6_unsqueeze_face_move.hpp.
WRITE_SHAPE = "face"
WRITE_SHAPES = {"segment": 0, "face": 1, "tile": 2}
WRITE_ALIGN_BYTES = 16  # BH NoC write alignment: the "segment" staging row stride

TILE_W = 32
FACE_W = 16
HALVES_PER_TILE = TILE_W // FACE_W
READ_ALIGN_SLACK_BYTES = 64  # BH DRAM read alignment: landing is align_up(cb_base, 64)

# Runtime-arg slots of the two buffer addresses (the only per-call values; see get_program_descriptor).
RT_IN_ADDR = 0
RT_OUT_ADDR = 1

# Host-side descriptor cache (BB-07): a repeat call with the same program-defining key reuses the
# built ProgramDescriptor and only patches the buffer addresses in its runtime args.
DESCRIPTOR_CACHE_SIZE = 32
_DESCRIPTOR_CACHE = OrderedDict()

# Semantic CB ids, indexed by the first tile half the instance owns.
CB_FACE_STAGE_HALF0 = 0
CB_FACE_STAGE_HALF1 = 1
_CB_FACE_STAGE = {0: CB_FACE_STAGE_HALF0, 1: CB_FACE_STAGE_HALF1}

G6UnsqueezePlan = namedtuple(
    "G6UnsqueezePlan",
    [
        "regime",
        "num_cores",
        "core_w_tiles_max",
        "block_w_tiles",
        "buffer_depth",
        "core_ranges",
        "halves_per_risc",
        "core_assignments",  # [(CoreCoord, start_tile, core_w_tiles)]
        "tensor_w_tiles",
        "entry_bytes",  # staging bytes per (tile, half) entry, set by the write shape
        "write_shape",
        "placement",  # "rect WxH" (one core range) or "row-wise" (num_cores_to_corerangeset fallback)
    ],
)


def _entry_bytes(tensor, write_shape):
    """Staging bytes per (tile, half): mirrors FaceMover::entry_bytes in the kernel."""
    face_row_bytes = FACE_W * tensor.element_size()
    if write_shape == "tile":
        return 2 * FACE_W * face_row_bytes  # face 2h at 2048 h of the staged tile image
    row_bytes = WRITE_ALIGN_BYTES if write_shape == "segment" else face_row_bytes
    return FACE_W * row_bytes


def _block_cap(buffer_depth, halves_per_risc, entry_bytes):
    """Largest block (tiles) whose staging fits the per-instance share of the L1 budget."""
    num_instances = HALVES_PER_TILE // halves_per_risc
    per_cb_bytes = L1_BUDGET_BYTES // num_instances - READ_ALIGN_SLACK_BYTES
    cap = per_cb_bytes // (buffer_depth * halves_per_risc * entry_bytes)
    if MAX_BLOCK_W_TILES is not None:
        cap = min(cap, MAX_BLOCK_W_TILES)
    if cap < 1:
        raise ValueError(f"g6_unsqueeze: L1 budget {L1_BUDGET_BYTES} B cannot stage one block")
    return cap


def _core_rectangle(num_cores, grid):
    """Most-square w x h rectangle at the grid origin with exactly `num_cores` cores, or None.

    One rectangle = one core range: the binaries and launch messages are written once instead of once
    per range (BB-17), and the cores sit in a compact block (short NoC hops). Most-square minimizes
    the bounding box; ties prefer the wider (row-major) shape.
    """
    best = None
    for h in range(1, grid.y + 1):
        if num_cores % h:
            continue
        w = num_cores // h
        if w > grid.x:
            continue
        if best is None or abs(w - h) < abs(best[0] - best[1]):
            best = (w, h)
    return best


def _core_pool(num_cores, grid):
    """The single source of the core set: one rectangle when `num_cores` factors into the grid,
    else the row-wise line of `num_cores_to_corerangeset` (e.g. a prime count wider than the grid)."""
    rect = _core_rectangle(num_cores, grid)
    if rect is None:
        return ttnn.num_cores_to_corerangeset(num_cores, grid, True), "row-wise"
    w, h = rect
    pool = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, h - 1))])
    return pool, f"rect {w}x{h}"


def select_plan(input_tensor) -> G6UnsqueezePlan:
    write_shape = WRITE_SHAPE
    if write_shape not in WRITE_SHAPES:
        raise ValueError(f"g6_unsqueeze: WRITE_SHAPE must be one of {list(WRITE_SHAPES)}, got {write_shape!r}")
    halves_per_risc = 2 if write_shape == "tile" else HALVES_PER_RISC
    if halves_per_risc not in (1, 2):
        raise ValueError(f"g6_unsqueeze: HALVES_PER_RISC must be 1 or 2, got {halves_per_risc}")
    entry_bytes = _entry_bytes(input_tensor, write_shape)
    tensor_w_tiles = (int(input_tensor.shape[-1]) + TILE_W - 1) // TILE_W

    grid = input_tensor.device().compute_with_storage_grid_size()
    num_cores = min(grid.x * grid.y, tensor_w_tiles)
    if MAX_CORES is not None:
        num_cores = min(num_cores, MAX_CORES)
    core_pool, placement = _core_pool(num_cores, grid)
    (n_used, all_cores, group_1, group_2, tiles_1, tiles_2) = ttnn.split_work_to_cores(core_pool, tensor_w_tiles, True)
    core_w_tiles_max = max(tiles_1, tiles_2)

    cap1 = _block_cap(1, halves_per_risc, entry_bytes)
    if core_w_tiles_max <= cap1:
        regime, buffer_depth, block_w_tiles = "resident", 1, core_w_tiles_max
    else:
        regime, buffer_depth = "streaming", 2
        block_w_tiles = _block_cap(2, halves_per_risc, entry_bytes)

    assignments = []
    start = 0
    for group, tiles in ((group_1, tiles_1), (group_2, tiles_2)):
        if tiles == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignments.append((core, start, tiles))
            start += tiles
    assert start == tensor_w_tiles, f"work split covers {start} of {tensor_w_tiles} tiles"

    return G6UnsqueezePlan(
        regime=regime,
        num_cores=n_used,
        core_w_tiles_max=core_w_tiles_max,
        block_w_tiles=block_w_tiles,
        buffer_depth=buffer_depth,
        core_ranges=all_cores,
        halves_per_risc=halves_per_risc,
        core_assignments=assignments,
        tensor_w_tiles=tensor_w_tiles,
        entry_bytes=entry_bytes,
        write_shape=write_shape,
        placement=placement,
    )


def describe_plan(plan) -> str:
    """Plan report (BB-08/BB-21): knobs, core -> tile range, RISC -> tile half."""
    lines = [
        f"g6_unsqueeze plan: regime={plan.regime} cores={plan.num_cores} tiles={plan.tensor_w_tiles} "
        f"core_w_tiles_max={plan.core_w_tiles_max} block_w_tiles={plan.block_w_tiles} "
        f"buffer_depth={plan.buffer_depth} halves_per_risc={plan.halves_per_risc} write_shape={plan.write_shape}",
        f"  placement={plan.placement} core_ranges={len(plan.core_ranges.ranges())}",
    ]
    for core, start, n in plan.core_assignments:
        lines.append(f"  core ({core.x},{core.y}): tiles [{start}, {start + n})")
    for half_begin in range(0, HALVES_PER_TILE, plan.halves_per_risc):
        risc = "reader-config (NCRISC)" if half_begin == 0 else "writer-config (BRISC)"
        lines.append(f"  {risc}: tile halves [{half_begin}, {half_begin + plan.halves_per_risc})")
    return "\n".join(lines)


def create_program_descriptor(input_tensor, output_tensor, plan=None) -> ttnn.ProgramDescriptor:
    if plan is None:
        plan = select_plan(input_tensor)
    logical_w = int(input_tensor.shape[-1])
    elem_bytes = input_tensor.element_size()
    in_tile_bytes = input_tensor.buffer_page_size()
    out_tile_bytes = output_tensor.buffer_page_size()
    assert in_tile_bytes == out_tile_bytes, "input and output tiles must match"

    stage_entries = plan.buffer_depth * plan.block_w_tiles * plan.halves_per_risc
    stage_total_bytes = stage_entries * plan.entry_bytes + READ_ALIGN_SLACK_BYTES

    accessor_ct = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    accessor_ct = list(accessor_ct) + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    cbs = []
    kernels = []
    for half_begin in range(0, HALVES_PER_TILE, plan.halves_per_risc):
        cb_id = _CB_FACE_STAGE[half_begin]
        cbs.append(
            ttnn.CBDescriptor(
                total_size=stage_total_bytes,
                core_ranges=plan.core_ranges,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cb_id,
                        data_format=input_tensor.dtype,
                        # Private scratch (no push/pop): the page is the 64 B alignment unit so the
                        # +64 B landing slack keeps total_size a whole number of pages.
                        page_size=READ_ALIGN_SLACK_BYTES,
                    )
                ],
            )
        )
        ct_args = [
            half_begin,
            plan.halves_per_risc,
            plan.block_w_tiles,
            plan.buffer_depth,
            logical_w,
            cb_id,
            elem_bytes,
            in_tile_bytes,
            READ_ALIGN_SLACK_BYTES,
            WRITE_SHAPES[plan.write_shape],
        ] + accessor_ct
        rt_args = ttnn.RuntimeArgs()
        for core, start, n in plan.core_assignments:
            rt_args[core.x][core.y] = [0, 0, start, n]  # addresses: RT_IN_ADDR / RT_OUT_ADDR, patched per call
        if half_begin == 0:
            source, config = "g6_unsqueeze_reader.cpp", ttnn.ReaderConfigDescriptor()
        else:
            source, config = "g6_unsqueeze_writer.cpp", ttnn.WriterConfigDescriptor()
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / source),
                core_ranges=plan.core_ranges,
                compile_time_args=ct_args,
                runtime_args=rt_args,
                config=config,
            )
        )

    program_descriptor = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    _patch_buffer_addresses(program_descriptor, plan, input_tensor, output_tensor)
    return program_descriptor


def _patch_buffer_addresses(program_descriptor, plan, input_tensor, output_tensor):
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    for kernel in program_descriptor.kernels:
        rt_args = kernel.runtime_args
        for core, _, _ in plan.core_assignments:
            core_args = rt_args[core.x][core.y]
            core_args[RT_IN_ADDR] = in_addr
            core_args[RT_OUT_ADDR] = out_addr


def _cache_key(input_tensor):
    """Everything that changes the built program (BB-07): shape, dtype, layout, placement, grid, knobs."""
    mc = input_tensor.memory_config()
    grid = input_tensor.device().compute_with_storage_grid_size()
    return (
        tuple(int(s) for s in input_tensor.shape),
        input_tensor.dtype,
        input_tensor.layout,
        mc.memory_layout,
        mc.buffer_type,
        (grid.x, grid.y),
        L1_BUDGET_BYTES,
        MAX_CORES,
        MAX_BLOCK_W_TILES,
        HALVES_PER_RISC,
        WRITE_SHAPE,
    )


def get_program_descriptor(input_tensor, output_tensor):
    """Cached (plan, descriptor) for this call; a hit patches only the two buffer addresses."""
    key = _cache_key(input_tensor)
    entry = _DESCRIPTOR_CACHE.get(key)
    if entry is None:
        plan = select_plan(input_tensor)
        entry = (plan, create_program_descriptor(input_tensor, output_tensor, plan))
        _DESCRIPTOR_CACHE[key] = entry
        if len(_DESCRIPTOR_CACHE) > DESCRIPTOR_CACHE_SIZE:
            _DESCRIPTOR_CACHE.popitem(last=False)
    else:
        _DESCRIPTOR_CACHE.move_to_end(key)
        _patch_buffer_addresses(entry[1], entry[0], input_tensor, output_tensor)
    return entry
