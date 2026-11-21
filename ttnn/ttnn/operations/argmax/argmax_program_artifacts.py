# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for argmax (op_design.md: Blocking Model, Work Distribution, Buffers).

One data-movement kernel, `argmax_scan`, with SCAN_THREADS threads on every engaged cluster; no compute
kernel (there is no index-carrying max on Quasar's NEOs). Thread plan (R, C, W) = (SCAN_THREADS, 0, 0):
each thread is the reader, the scanner and the writer of its own contiguous range of units.

Two regimes, chosen by `select_regime()` from the grid and the shape:

    ROWS       a unit is one row; a scanner scans whole rows and writes their indices itself.
    SEGMENTED  a unit is one (row, segment) of `spr` segments per row; each scanner sends one 16-B
               record {key, index, tag, 0} per unit to the finalizer (the cluster holding unit 0,
               thread 0), which waits for all U records on a semaphore, merges each row's segments in
               column order and writes the indices.

Every knob is defined once below (host constants) or derived once in `derive_blocks()`; the kernel takes
them as compile-time or runtime arguments and sizes its loops from them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- The thread plan and the blocking knobs (op_design.md, "Single sources") ------------------------------
SCAN_THREADS = 6  # DM threads per cluster: the thread plan (6, 0, 0). User DM harts on Quasar: 6.
IN_DEPTH = 2  # input block slots per thread (effective depth min(IN_DEPTH, blocks per scanner))
OUT_DEPTH = 2  # output index staging slots per thread
MIN_SEGMENT_BYTES = 2048  # smallest row segment SEGMENTED deals to one scanner
L1_RESERVE_BYTES = 65536  # kept free of this op's scratchpads
RECORD_BYTES = 16  # one partial record {key, index, tag, 0}
# A DM hart reads and writes L1 through a write-back cache of 64-B lines (internal/tt-2xx/quasar/cache.h,
# overlay_addresses.h L2_CACHE_LINE_SIZE). Every slot one thread writes with the CPU is spaced and aligned
# to a line, so a flush never writes back bytes another thread owns; each scratchpad is padded by one line
# and its base aligned up in the kernel. Python has no query for it.
DM_CACHE_LINE_BYTES = 64
USER_TRIDS = 8  # NoC user transaction ids 0..7 on Quasar (noc.h: DFB_TXN_ID_BASE = 8)

ROWS, SEGMENTED = "ROWS", "SEGMENTED"

# Transaction ids: reads 0..IN_DEPTH-1, output writes next OUT_DEPTH, records the last.
READ_TRID_BASE = 0
WRITE_TRID_BASE = READ_TRID_BASE + IN_DEPTH
RECORD_TRID = WRITE_TRID_BASE + OUT_DEPTH
assert RECORD_TRID < USER_TRIDS, "IN_DEPTH + OUT_DEPTH + 1 user transaction ids must fit 0..7"
assert 1 <= SCAN_THREADS <= 6, "a DM KernelSpec takes at most 6 threads on Quasar"


def line_bytes():
    """The spacing of every per-thread CPU-written slot and the alignment pad of each scratchpad: one DM cache
    line, never below the L1 alignment. The single source for the host sizes and the kernel's CT arg."""
    return max(ttnn.get_l1_alignment(), DM_CACHE_LINE_BYTES)


def _align_up(x, a):
    return (x + a - 1) // a * a


def _align_down(x, a):
    return x // a * a


@dataclass(frozen=True)
class Regime:
    name: str  # ROWS or SEGMENTED
    spr: int  # segments per row (1 for ROWS)
    seg_cols: int  # columns per segment (tensor_w for ROWS); the last segment may be shorter
    num_units: int  # U


def select_regime(tensor_rows, tensor_w, elem_bytes, num_clusters, scan_threads=SCAN_THREADS):
    """ROWS, or SEGMENTED when ROWS would leave more than half of the scanners idle (op_design.md, Regimes)."""
    s_total = num_clusters * scan_threads
    spr = min(s_total // tensor_rows, (tensor_w * elem_bytes) // MIN_SEGMENT_BYTES)
    if spr >= 2:
        seg_cols = math.ceil(tensor_w / spr)
        spr = math.ceil(tensor_w / seg_cols)  # every segment non-empty
        assert spr >= 2 and (spr - 1) * seg_cols < tensor_w
        return Regime(SEGMENTED, spr, seg_cols, tensor_rows * spr)
    return Regime(ROWS, 1, tensor_w, tensor_rows)


def _grid(device, sub_core_grids):
    """The clusters the program may run on: `sub_core_grids` if given (R5), else the device grid (query)."""
    if sub_core_grids is not None:
        return sub_core_grids
    g = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])


def split_units(grid, num_units, scan_threads=SCAN_THREADS):
    """Contiguous unit ranges to clusters (split_work_to_cores, row-wise), then contiguously to threads.

    Returns (engaged core set, [core in dealing order], {core: (start, count)}, {core: [start per thread]},
    {core: [count per thread]}).
    """
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, num_units, row_wise=True)
    cores, ranges, start = [], {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            cores.append(core)
            ranges[core] = (start, per_core)
            start += per_core
    assert start == num_units
    t_start, t_count = {}, {}
    for core in cores:
        c_start, c_count = ranges[core]
        base, extra = divmod(c_count, scan_threads)
        counts = [base + (1 if t < extra else 0) for t in range(scan_threads)]
        starts = [c_start + t * base + min(t, extra) for t in range(scan_threads)]
        t_start[core], t_count[core] = starts, counts
    return all_cores, cores, ranges, t_start, t_count


@dataclass(frozen=True)
class Blocks:
    block_bytes: int  # one input block slot
    block_cols: int  # elements per row piece in a block
    block_rows: int  # row pieces per block (1 when a piece spans several blocks)
    row_pitch: int  # bytes between row pieces in a slot
    in_depth: int  # effective input depth
    max_blocks: int  # most blocks any scanner walks
    out_stage_rows: int  # indices one staging slot holds
    stage_bytes: int  # one staging slot
    partials_bytes: int  # landing zone + send slots (without the alignment pad)
    send_base: int  # offset of the send slots in sp_partials
    send_stride: int  # bytes between two threads' send slots
    line_bytes: int  # slot spacing and per-scratchpad alignment pad (line_bytes())


def derive_blocks(
    regime, tensor_rows, elem_bytes, num_clusters, max_units_per_scanner, scan_threads=SCAN_THREADS, block_bytes_cap=None
):
    """The coarsest block that fits L1: block_bytes = min(need, cap) (op_design.md, Blocking Model).

    `block_bytes_cap` (tests and measurements only) lowers the cap, so a scanner walks several blocks.
    """
    l1_align = ttnn.get_l1_alignment()
    line = line_bytes()
    assert RECORD_BYTES >= l1_align and RECORD_BYTES % l1_align == 0
    s_total = num_clusters * scan_threads
    send_base = _align_up(s_total * RECORD_BYTES, line)  # landing zone: one record per unit, U <= S_total
    send_stride = _align_up(RECORD_BYTES, line)
    partials_bytes = send_base + scan_threads * send_stride
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
    # per thread: IN_DEPTH block slots + OUT_DEPTH staging slots of at most block_bytes/4 each
    pads = 3 * line  # one alignment pad per scratchpad
    cap = _align_down((budget - partials_bytes - pads) * 4 // scan_threads // (4 * IN_DEPTH + OUT_DEPTH), line)
    if block_bytes_cap is not None:
        cap = min(cap, max(line, _align_down(block_bytes_cap, line)))
    piece_cols = regime.seg_cols
    need = _align_up(max_units_per_scanner * _align_up(piece_cols * elem_bytes, l1_align), line)
    block_bytes = min(need, cap)  # a slot; slots are line-aligned
    block_cols = min(piece_cols, block_bytes // elem_bytes)
    row_pitch = _align_up(block_cols * elem_bytes, l1_align)
    assert block_cols >= 1 and row_pitch <= block_bytes
    if block_cols == piece_cols:
        block_rows = max(1, min(max_units_per_scanner, block_bytes // row_pitch))
        blocks_per_piece = 1
    else:
        block_rows = 1
        blocks_per_piece = math.ceil(piece_cols / block_cols)
    max_blocks = math.ceil(max_units_per_scanner / block_rows) * blocks_per_piece
    in_depth = max(1, min(IN_DEPTH, max_blocks))
    out_stage_rows = block_rows if regime.name == ROWS else tensor_rows
    stage_bytes = _align_up(out_stage_rows * 4, line)
    total = scan_threads * (in_depth * block_bytes + OUT_DEPTH * stage_bytes) + partials_bytes + pads
    assert total <= budget, f"argmax: L1 footprint {total} B exceeds the budget {budget} B"
    return Blocks(
        block_bytes,
        block_cols,
        block_rows,
        row_pitch,
        in_depth,
        max_blocks,
        out_stage_rows,
        stage_bytes,
        partials_bytes,
        send_base,
        send_stride,
        line,
    )


def create_program_artifacts(
    input_tensor, output, *, sub_core_grids=None, probe_scanner_ids=False, block_bytes_cap=None
):
    """(ProgramSpec, ProgramRunArgs, tensor indices) of one argmax call along the last dimension.

    probe_scanner_ids  (tests only) ROWS: each index is replaced by the global scanner id
                       g = cluster_ordinal * SCAN_THREADS + thread of the thread that scanned the row.
                       SEGMENTED: each index is replaced by the sum over the row's merged records of (g + 1).
    block_bytes_cap    (tests and measurements only) an upper bound on block_bytes: streams several blocks
                       per scanner (depth IN_DEPTH) and splits a row piece over column blocks.
    """
    device = input_tensor.device()
    shape = tuple(input_tensor.shape)
    tensor_w = shape[-1]
    tensor_rows = math.prod(shape[:-1])
    elem_bytes = input_tensor.element_size()
    grid = _grid(device, sub_core_grids)
    num_clusters = grid.num_cores()
    T = SCAN_THREADS

    regime = select_regime(tensor_rows, tensor_w, elem_bytes, num_clusters, T)
    core_set, cores, _, t_start, t_count = split_units(grid, regime.num_units, T)
    max_units = max(max(c) for c in t_count.values())
    blocks = derive_blocks(regime, tensor_rows, elem_bytes, num_clusters, max_units, T, block_bytes_cap)
    assert regime.num_units <= num_clusters * T or regime.name == ROWS

    out_shape = tuple(output.shape)
    out_rows_per_page = out_shape[-1] if len(out_shape) >= 1 else 1
    finalizer = cores[0]  # holds unit 0; its thread 0 owns unit 0
    fin_noc = device.worker_core_from_logical_core(finalizer)

    kernel = E.KernelSpec(
        unique_id="argmax_scan",
        source=str(KERNEL_DIR / "argmax_scan.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=T,
        scratchpad_bindings=[
            E.KernelSpec.ScratchpadBinding(scratchpad_spec_name=n, accessor_name=n)
            for n in ("sp_blocks", "sp_out_stage", "sp_partials")
        ],
        semaphore_bindings=[
            E.KernelSpec.SemaphoreBinding(semaphore_spec_name="records_arrived", accessor_name="records_arrived")
        ],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name="input", accessor_name="input"),
            E.TensorBinding(tensor_parameter_name="output", accessor_name="output"),
        ],
        compile_time_args={
            "scan_threads": T,
            "in_depth": blocks.in_depth,
            "out_depth": OUT_DEPTH,
            "read_trid_base": READ_TRID_BASE,
            "write_trid_base": WRITE_TRID_BASE,
            "record_trid": RECORD_TRID,
            "elem_bytes": elem_bytes,
            "segmented": int(regime.name == SEGMENTED),
            "probe_ids": int(bool(probe_scanner_ids)),
            "line_bytes": blocks.line_bytes,
            "record_bytes": RECORD_BYTES,
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["is_finalizer", "cluster_ordinal"],
            common_runtime_arg_names=[
                "tensor_w",
                "tensor_rows",
                "spr",
                "seg_cols",
                "block_bytes",
                "block_cols",
                "block_rows",
                "row_pitch",
                "stage_bytes",
                "out_rows_per_page",
                "num_records",
                "finalizer_noc_x",
                "finalizer_noc_y",
                "send_base",
                "send_stride",
            ],
            runtime_array_args=[
                E.KernelSpec.RuntimeArgSchema.ArrayArgDecl(name="unit_start", size=T),
                E.KernelSpec.RuntimeArgSchema.ArrayArgDecl(name="unit_count", size=T),
            ],
        ),
    )
    pad = blocks.line_bytes  # the kernel aligns each base up to a line; derive_blocks counted these pads
    spec = E.ProgramSpec(
        name="argmax",
        kernels=[kernel],
        scratchpads=[
            E.ScratchpadSpec(unique_id="sp_blocks", size_per_node=T * blocks.in_depth * blocks.block_bytes + pad),
            E.ScratchpadSpec(unique_id="sp_out_stage", size_per_node=T * OUT_DEPTH * blocks.stage_bytes + pad),
            E.ScratchpadSpec(unique_id="sp_partials", size_per_node=blocks.partials_bytes + pad),
        ],
        semaphores=[E.SemaphoreSpec(unique_id="records_arrived", target_nodes=core_set)],
        tensor_parameters=[
            E.TensorParameter(unique_id="input", spec=input_tensor.spec),
            E.TensorParameter(unique_id="output", spec=output.spec),
        ],
        work_units=[E.WorkUnitSpec(name="main", kernels=["argmax_scan"], target_nodes=core_set)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel="argmax_scan",
                runtime_arg_values={
                    "is_finalizer": {c: int(c == finalizer) for c in cores},
                    "cluster_ordinal": {c: i for i, c in enumerate(cores)},
                },
                common_runtime_arg_values={
                    "tensor_w": tensor_w,
                    "tensor_rows": tensor_rows,
                    "spr": regime.spr,
                    "seg_cols": regime.seg_cols,
                    "block_bytes": blocks.block_bytes,
                    "block_cols": blocks.block_cols,
                    "block_rows": blocks.block_rows,
                    "row_pitch": blocks.row_pitch,
                    "stage_bytes": blocks.stage_bytes,
                    "out_rows_per_page": out_rows_per_page,
                    "num_records": regime.num_units if regime.name == SEGMENTED else 0,
                    "finalizer_noc_x": fin_noc.x,
                    "finalizer_noc_y": fin_noc.y,
                    "send_base": blocks.send_base,
                    "send_stride": blocks.send_stride,
                },
                runtime_array_arg_values={"unit_start": t_start, "unit_count": t_count},
            )
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}
