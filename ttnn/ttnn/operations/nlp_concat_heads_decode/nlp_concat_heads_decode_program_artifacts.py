# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for nlp_concat_heads_decode (regime `pull_rows`, op_design.md).

One data-movement kernel (the gather, `kernels/nlp_concat_heads_decode_reader.cpp`) runs on every output core:
core i (the i-th core of the output's shard grid, row major) owns head i. Its reader threads split the head's
`head_tiles` tiles into contiguous ranges of `block_tiles`; each thread zero-fills its range of the local output
shard (one iDMA call, one barrier) and then pulls, for every user u and every tile t of its range, the two
16-element runs of row `head_idx` of input page `u * head_tiles + t` into row u of local tile t: all reads issued
back to back on one transaction id, one barrier.

Thread plan (R, C, W) = (READER_THREADS, 0, 0): no compute kernel (nothing is computed) and no writer (the reads
land in the output shard). The op allocates no dataflow buffer, scratchpad or semaphore.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# The thread plan: reader harts per output core. The one place the count is stated; the kernel sizes its tile
# range from it (CT arg `reader_threads`) and its thread id. Target page: R + W <= 6 user DM harts.
READER_THREADS = 1
FREE_DM_HARTS = 6

# Zero the output shard's padding rows (rows users..31; rule R3 allows leaving them as allocated). CT knob.
ZERO_PADDING_ROWS = 1


def tile_geometry(tensor):
    """Byte geometry of one tile of `tensor`, derived once from its tile and page size."""
    tile_h, tile_w = (int(d) for d in tensor.tile.tile_shape)
    face_h, face_w = (int(d) for d in tensor.tile.face_shape)
    tile_bytes = int(tensor.buffer_page_size())
    elem_bytes = tile_bytes // (tile_h * tile_w)
    return {
        "tile_bytes": tile_bytes,
        "face_bytes": elem_bytes * face_h * face_w,
        "run_bytes": elem_bytes * face_w,  # one row of one face: the unit a NoC read moves
        "face_h": face_h,
        "runs_per_row": tile_w // face_w,  # faces side by side in a tile row
        "tile_w": tile_w,
    }


def create_program_artifacts(input_tensor, output, *, num_heads, reader_threads=None, zero_padding_rows=None):
    """(ProgramSpec, ProgramRunArgs, tensor indices) of one call. `reader_threads` / `zero_padding_rows` override
    the module knobs (tests)."""
    if reader_threads is None:
        reader_threads = READER_THREADS
    if zero_padding_rows is None:
        zero_padding_rows = ZERO_PADDING_ROWS
    if not 1 <= reader_threads <= FREE_DM_HARTS:
        raise ValueError(f"reader_threads must be in [1, {FREE_DM_HARTS}], got {reader_threads}")

    geom = tile_geometry(input_tensor)
    out_geom = tile_geometry(output)
    if geom != out_geom:
        raise ValueError(f"input and output tile geometry differ: {geom} vs {out_geom}")
    num_users = int(input_tensor.shape[1])
    head_dim = int(input_tensor.shape[3])
    head_tiles = -(-head_dim // geom["tile_w"])

    # Thread split of the head's tiles: contiguous ranges, the last one possibly shorter. A thread with an empty
    # range would be a wasted hart, so the plan is clamped to the tiles there are.
    block_tiles = -(-head_tiles // min(reader_threads, head_tiles))
    threads = -(-head_tiles // block_tiles)  # every thread gets at least one tile (5 tiles, R=4 -> 3 threads)

    cores = output.memory_config().shard_spec.grid
    head_cores = ttnn.corerange_to_cores(cores, None, True)  # row major: the order of the ROW_MAJOR shards
    if len(head_cores) != num_heads:
        raise ValueError(f"output shard grid has {len(head_cores)} cores for {num_heads} heads")

    gather = E.KernelSpec(
        unique_id="gather",
        source=str(KERNEL_DIR / "nlp_concat_heads_decode_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=threads,
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name="input", accessor_name="input"),
            E.TensorBinding(tensor_parameter_name="output", accessor_name="output"),
        ],
        compile_time_args={
            "tile_bytes": geom["tile_bytes"],
            "face_bytes": geom["face_bytes"],
            "run_bytes": geom["run_bytes"],
            "face_h": geom["face_h"],
            "runs_per_row": geom["runs_per_row"],
            "head_tiles": head_tiles,
            "num_users": num_users,
            "reader_threads": threads,
            "block_tiles": block_tiles,
            "zero_padding_rows": int(bool(zero_padding_rows)),
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["head_idx"]),
    )
    spec = E.ProgramSpec(
        name="nlp_concat_heads_decode",
        kernels=[gather],
        tensor_parameters=[
            E.TensorParameter(unique_id="input", spec=input_tensor.spec),
            E.TensorParameter(unique_id="output", spec=output.spec),
        ],
        work_units=[E.WorkUnitSpec(name="main", kernels=["gather"], target_nodes=cores)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="gather", runtime_arg_values={"head_idx": {c: i for i, c in enumerate(head_cores)}})
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}
