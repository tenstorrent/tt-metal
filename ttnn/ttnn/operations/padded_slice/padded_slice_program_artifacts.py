# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for padded_slice (op_design.md, regime A: RM row gather into own shard).

One data-movement kernel, `gather` (kernels/padded_slice_reader.cpp), runs on every core of the output's
shard grid with GATHER_THREADS harts. Core k of the grid (in the shard orientation's order) owns result rows
[k*shard_h, (k+1)*shard_h) (rule R5) and reads each of them, one NoC read per row, from the input's DRAM page
straight into its place in its own shard: no ring, no compute kernel, no writer (design thread plan
(R, C, W) = (GATHER_THREADS, 0, 0); every byte crosses DRAM once and lands in its final location).

Knobs (each defined once here):
    GATHER_THREADS  thread plan R: reader harts per core; hart t takes a contiguous sub-range of each block
    MAX_BLOCK_ROWS  block extent along out_row; 0 = the whole shard (one block, one barrier per hart)
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# Thread plan (R, C, W) = (GATHER_THREADS, 0, 0): the design's Work Distribution.
GATHER_THREADS = 2
# The cluster model of the target page (.claude/references/targets/quasar.md): user DM harts per cluster.
FREE_DM_HARTS = 6
# Block extent along out_row, in rows; 0 = the whole shard.
MAX_BLOCK_ROWS = 0

K_GATHER = "gather"
TP_INPUT = "input"
TP_OUTPUT = "output"

COMMON_ARGS = [
    "block_rows",  # rows per block (the block knob)
    "block_h",  # extent of the block along H (address walk)
    "block_w",  # extent of the block along W (address walk)
    "in_h",  # input H
    "in_w",  # input W
    "start_n",
    "start_h",
    "start_w",
    "src_offset_bytes",  # start[3] * element bytes: where a row's data starts inside its input page
    "row_bytes",  # c * element bytes: what is read per row
    "out_row_stride",  # the output's aligned page size: row stride in the shard
]


def shard_cores(memory_config):
    """The shard grid's cores in shard order: core k holds shard k (R5)."""
    shard_spec = memory_config.shard_spec
    row_wise = shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    return list(ttnn.corerange_to_cores(shard_spec.grid, None, row_wise))


def create_program_artifacts(input_tensor, output, start, end, *, gather_threads=GATHER_THREADS,
                             max_block_rows=None):
    R = int(gather_threads)
    if not 1 <= R <= FREE_DM_HARTS:
        raise ValueError(f"padded_slice: gather_threads {R} must be 1..{FREE_DM_HARTS} (R + W <= 6, W = 0)")
    max_block_rows = MAX_BLOCK_ROWS if max_block_rows is None else int(max_block_rows)

    _, in_h, in_w, _ = (int(d) for d in input_tensor.shape)
    n, h, w, c = (e - s for s, e in zip(start, end))
    rows = n * h * w
    if in_h * in_w * int(input_tensor.shape[0]) >= 2**32:
        raise NotImplementedError("padded_slice: the input has too many pages for 32-bit page ids")

    # From the dtype, not the page: a page is one row only in row-major layout (a TILE page is a tile).
    elem_bytes = int(input_tensor.element_size())
    memory_config = output.memory_config()
    shard_h = int(memory_config.shard_spec.shape[0])
    block_rows = shard_h if max_block_rows == 0 else min(max_block_rows, shard_h)

    cores = shard_cores(memory_config)
    first_row, num_rows = {}, {}
    for k, core in enumerate(cores):
        first = k * shard_h
        first_row[core] = first
        num_rows[core] = max(0, min(rows - first, shard_h))

    common = {
        "block_rows": block_rows,
        "block_h": h,
        "block_w": w,
        "in_h": in_h,
        "in_w": in_w,
        "start_n": start[0],
        "start_h": start[1],
        "start_w": start[2],
        "src_offset_bytes": start[3] * elem_bytes,
        "row_bytes": c * elem_bytes,
        "out_row_stride": output.buffer_aligned_page_size(),
    }

    gather = E.KernelSpec(
        unique_id=K_GATHER,
        source=str(KERNEL_DIR / "padded_slice_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name=TP_INPUT, accessor_name=TP_INPUT),
            E.TensorBinding(tensor_parameter_name=TP_OUTPUT, accessor_name=TP_OUTPUT),
        ],
        compile_time_args={"reader_threads": R},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["first_row", "num_rows"], common_runtime_arg_names=COMMON_ARGS
        ),
    )
    spec = E.ProgramSpec(
        name="padded_slice",
        kernels=[gather],
        dataflow_buffers=[],
        tensor_parameters=[
            E.TensorParameter(unique_id=TP_INPUT, spec=input_tensor.spec),
            E.TensorParameter(unique_id=TP_OUTPUT, spec=output.spec),
        ],
        work_units=[E.WorkUnitSpec(name="main", kernels=[K_GATHER], target_nodes=memory_config.shard_spec.grid)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel=K_GATHER,
                runtime_arg_values={"first_row": first_row, "num_rows": num_rows},
                common_runtime_arg_values=common,
            )
        ]
    )
    return spec, run_args, {TP_INPUT: 0, TP_OUTPUT: 1}
