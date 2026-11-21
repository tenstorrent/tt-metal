# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for slice_write (regime `shard_owner_writes`, op_design.md).

The op is a page map. A row-major page is one row of the last dimension, and the block always spans the
destination's whole last dimension, so source row r goes, unchanged, to destination page

    (n, h, w)   = (r / (h_s*w_s), (r / w_s) mod h_s, r mod w_s)
    dst_page(r) = ((n + start[0])*H + (h + start[1]))*W + (w + start[2])

The source is height-sharded in L1. Shard i lives on core i of the shard grid (in the grid's orientation),
and that core writes it: the block is the whole resident shard, published as ring entries by the reader with
no data moved (the ring is borrowed onto the shard: zero-copy), and written row by row to the destination's
DRAM pages by the writer, `rows_per_write_batch` writes per barrier. No compute kernel: the op computes
nothing (spec DEFINITION).

Knobs (op_design.md, Blocking Model), each defined once here:
    THREAD_PLAN        (R, C, W) = reader harts, compute threads, writer harts per core
    WRITE_BATCH_ROWS   NoC writes in flight per barrier (cap of `rows_per_write_batch`)
    src_shard_entries  ring entries over the shard = W (one per writer hart); with W = 1 the whole shard
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# The thread plan of a data-movement-bound op with no compute (op_design.md, Work Distribution).
THREAD_PLAN = (1, 0, 1)
# Rows written per barrier (op_design.md, Mechanism caps: the per-trid outstanding-write counter).
WRITE_BATCH_ROWS = 32

FREE_DM_HARTS = 6  # of 8 per cluster (target page, cluster model): DM0 serves ring interrupts, DM1 remaps

DFB_SRC_SHARD = "dfb_src_shard"


def _validate_thread_plan(threads):
    r, c, w = threads
    if not (r >= 1 and c == 0 and w >= 1 and r + w <= FREE_DM_HARTS):
        raise ValueError(f"slice_write thread plan (R, C, W) = {threads}: need C = 0, R, W >= 1, R + W <= {FREE_DM_HARTS}")


def shard_assignment(input_tensor, total_rows):
    """[(core, first_row, num_rows)] for every core whose shard holds valid rows, in shard order."""
    shard_spec = input_tensor.memory_config().shard_spec
    shard_rows = int(shard_spec.shape[0])
    row_wise = shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    cores = ttnn.corerange_to_cores(shard_spec.grid, None, row_wise)
    assignment = []
    for i, core in enumerate(cores):
        first_row = i * shard_rows
        num_rows = min(shard_rows, total_rows - first_row)
        if num_rows <= 0:
            break
        assignment.append((core, first_row, num_rows))
    return shard_rows, assignment


def create_program_artifacts(input_tensor, output_tensor, start, end, *, threads=THREAD_PLAN,
                             write_batch_rows=WRITE_BATCH_ROWS):
    _validate_thread_plan(threads)
    R, _, W = threads
    in_shape = [int(d) for d in input_tensor.shape]
    out_shape = [int(d) for d in output_tensor.shape]
    _, block_h, block_w, _ = in_shape
    _, dst_h, dst_w, _ = out_shape
    total_rows = math.prod(in_shape[:-1])
    row_bytes = in_shape[-1] * input_tensor.element_size()

    # The ring over the resident shard. dfb_spec_from_sharded_tensor gives the shard's row stride in L1
    # (entry_size) and its allocation height (num_entries); the op then cuts the shard into
    # src_shard_entries entries, one per writer hart.
    derived = E.dfb_spec_from_sharded_tensor(DFB_SRC_SHARD, input_tensor, borrowed_from="src")
    src_row_stride_bytes = int(derived.entry_size)
    shard_rows, assignment = shard_assignment(input_tensor, total_rows)
    if int(derived.num_entries) != shard_rows:
        raise RuntimeError(f"slice_write: ring rows {derived.num_entries} != shard rows {shard_rows}")
    # One ring entry per writer hart. The entries must tile the borrowed shard exactly, so a plan whose W does
    # not divide the shard's rows runs at W = 1 (op_design.md, Perf lamps: "requires W | shard_rows, else
    # W = 1"): the knob never refuses a valid call.
    W = W if shard_rows % W == 0 else 1
    src_shard_entries = W
    entry_rows = shard_rows // src_shard_entries
    rows_per_write_batch = max(1, min(write_batch_rows, entry_rows))

    dfb = E.DataflowBufferSpec(
        unique_id=DFB_SRC_SHARD,
        entry_size=entry_rows * src_row_stride_bytes,
        num_entries=src_shard_entries,
        data_format_metadata=input_tensor.dtype,
        borrowed_from="src",
    )

    explicit = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=[DFB_SRC_SHARD])
    reader_hw = E.create_reader_datamovement_config()
    reader_hw.config_2xx = explicit
    writer_hw = E.create_writer_datamovement_config()
    writer_hw.config_2xx = explicit

    plan = {"reader_threads": R, "writer_threads": W, "src_shard_entries": src_shard_entries}
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "slice_write_reader.cpp"),
        hw_config=reader_hw,
        num_threads=R,
        dfb_bindings=[E.producer_of(DFB_SRC_SHARD, "src_shard")],
        compile_time_args=plan,
    )
    # Shape-dependent values are run-time arguments, so one compiled program serves every shape. The schema's
    # names are this dict's keys (one source).
    common = {"row_bytes": row_bytes, "src_row_stride_bytes": src_row_stride_bytes, "entry_rows": entry_rows,
              "block_h": block_h, "block_w": block_w, "dst_h": dst_h, "dst_w": dst_w,
              "start_n": int(start[0]), "start_h": int(start[1]), "start_w": int(start[2])}
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "slice_write_writer.cpp"),
        hw_config=writer_hw,
        num_threads=W,
        dfb_bindings=[E.consumer_of(DFB_SRC_SHARD, "src_shard")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="dst", accessor_name="dst")],
        compile_time_args={**plan, "rows_per_write_batch": rows_per_write_batch},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["first_row", "num_rows"], common_runtime_arg_names=list(common)),
    )

    cores = [core for core, _, _ in assignment]
    target_nodes = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
    spec = E.ProgramSpec(
        name="slice_write",
        kernels=[reader, writer],
        dataflow_buffers=[dfb],
        tensor_parameters=[E.TensorParameter(unique_id="src", spec=input_tensor.spec),
                           E.TensorParameter(unique_id="dst", spec=output_tensor.spec)],
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=target_nodes)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader"),
            E.KernelRunArgs(
                kernel="writer",
                runtime_arg_values={"first_row": {c: f for c, f, _ in assignment},
                                    "num_rows": {c: n for c, _, n in assignment}},
                common_runtime_arg_values=common,
            ),
        ]
    )
    return spec, run_args, {"src": 0, "dst": 1}
