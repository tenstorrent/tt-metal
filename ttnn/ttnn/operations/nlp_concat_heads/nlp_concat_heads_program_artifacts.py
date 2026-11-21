# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for nlp_concat_heads, regime `tile_relay` (head_dim % 32 == 0).

Every output tile is exactly one input tile: output page p = ((b * St + r) * H + h) * Dt + c is input page
((b * H + h) * St + r) * Dt + c. The output pages are split into contiguous ranges over the whole grid; on each
cluster a reader walks its range in output order, reading each tile from the input into one ring, and a writer
stores the ring's tiles to the range. No compute kernel: no element passes through DEST, so every dtype moves
bit for bit (rule R1).

Knobs (each defined once here; every dependent quantity derives from them):

    THREAD_PLAN         (R, C, W): reader harts, compute threads (none: nothing to compute), writer harts
    RING_DEPTH_TILES    ring entries per DM thread endpoint: the reads in flight
    BLOCK_TILES         pages per load_block / store_block; None = the core's whole range (one block)
    MIN_TILES_PER_CORE  the least a cluster is given: engaged clusters = min(grid, ceil(T / this))
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

PROGRAM_NAME = "nlp_concat_heads_tile_relay"
TILE = 32

# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for it.
FREE_DM_HARTS = 6
DFB_RING_MAX_BYTES = 1_048_560

THREAD_PLAN = (1, 0, 1)
RING_DEPTH_TILES = 8
BLOCK_TILES = None
MIN_TILES_PER_CORE = 1

RING = "relay"
K_READER = "reader"
K_WRITER = "writer"
TP_INPUT = "input"
TP_OUTPUT = "output"


def validate_thread_plan(threads):
    r, c, w = threads
    if c != 0:
        raise ValueError(f"nlp_concat_heads runs no compute kernel: thread plan {threads} must have C = 0")
    if not (r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and (r % w == 0 or w % r == 0)):
        raise ValueError(f"thread plan (R, C, W) = {threads}: need R, W >= 1, R + W <= {FREE_DM_HARTS}, R | W or W | R")


def head_geometry(shape):
    """(B, H, St, Dt) of an input [B, H, S, D] with D % 32 == 0: the tile grid the walk runs over.

    D % 32 != 0 is the deferred `subtile_assemble` regime (a head ends inside a tile, so an output tile is not
    one input tile): refused here, so a validate() widened ahead of that regime cannot dispatch a wrong walk.
    """
    batch, heads, tokens, head_dim = (int(d) for d in shape)
    if head_dim % TILE != 0:
        raise NotImplementedError(
            f"nlp_concat_heads: head_dim={head_dim} is not a multiple of {TILE}: regime `subtile_assemble` is not built"
        )
    return batch, heads, -(-tokens // TILE), head_dim // TILE


def split_pages(num_pages, grid_size, min_tiles_per_core=MIN_TILES_PER_CORE):
    """(core set, {core: pages}, {core: first page}): contiguous ranges of output pages, row-major over the grid.

    The work is split in chunks of `min_tiles_per_core` pages (the last chunk may be short), so at most
    ceil(num_pages / min_tiles_per_core) clusters are engaged. With 1 this is split_work_to_cores over pages.
    """
    chunks = -(-num_pages // min_tiles_per_core)
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid_size, chunks, row_wise=True)
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            n = min(per_core * min_tiles_per_core, num_pages - start)
            counts[core], starts[core] = n, start
            start += n
    assert start == num_pages, (start, num_pages)
    return all_cores, counts, starts


def create_program_artifacts(
    input_tensor,
    output_tensor,
    *,
    threads=THREAD_PLAN,
    ring_depth_tiles=RING_DEPTH_TILES,
    block_tiles=BLOCK_TILES,
    min_tiles_per_core=MIN_TILES_PER_CORE,
):
    validate_thread_plan(threads)
    R, _, W = threads
    if ring_depth_tiles < 1 or ring_depth_tiles % 4 != 0:
        # entry_threshold = num_entries / num_txn_ids must split evenly over 1, 2 or 4 txn ids.
        raise ValueError(f"ring_depth_tiles={ring_depth_tiles} must be a positive multiple of 4")
    if block_tiles is not None and block_tiles < 1:
        raise ValueError(f"block_tiles={block_tiles} must be >= 1 or None")
    if min_tiles_per_core < 1:
        raise ValueError(f"min_tiles_per_core={min_tiles_per_core} must be >= 1")

    page_bytes = input_tensor.buffer_page_size()
    assert page_bytes == output_tensor.buffer_page_size(), "one ring entry is one tile of both tensors"
    num_entries = ring_depth_tiles * max(R, W)
    ring_bytes = num_entries * page_bytes
    assert ring_bytes <= min(DFB_RING_MAX_BYTES, ttnn.get_max_worker_l1_unreserved_size()), ring_bytes

    batch, heads, token_rows, head_dim_tiles = head_geometry(input_tensor.shape)
    num_pages = batch * token_rows * heads * head_dim_tiles
    assert num_pages == output_tensor.buffer_num_pages() == input_tensor.buffer_num_pages(), num_pages
    assert num_pages < 2**32

    core_set, counts, starts = split_pages(
        num_pages, input_tensor.device().compute_with_storage_grid_size(), min_tiles_per_core
    )
    blocks = {core: (n if block_tiles is None else min(block_tiles, n)) for core, n in counts.items()}

    ring = E.DataflowBufferSpec(
        unique_id=RING, entry_size=page_bytes, num_entries=num_entries, data_format_metadata=input_tensor.dtype
    )
    tensor_parameters = [
        E.TensorParameter(unique_id=TP_INPUT, spec=input_tensor.spec),
        E.TensorParameter(unique_id=TP_OUTPUT, spec=output_tensor.spec),
    ]
    reader_rt = ["core_tiles", "start_page", "block_tiles", "head_dim_tiles", "heads", "token_rows"]
    writer_rt = ["core_tiles", "start_page", "block_tiles"]

    reader = E.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "nlp_concat_heads_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        dfb_bindings=[E.producer_of(RING, RING)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_INPUT, accessor_name="input")],
        compile_time_args={"reader_threads": R},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=reader_rt),
    )
    writer = E.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "nlp_concat_heads_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of(RING, RING)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_OUTPUT, accessor_name="output")],
        compile_time_args={"writer_threads": W},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=writer_rt),
    )
    spec = E.ProgramSpec(
        name=PROGRAM_NAME,
        kernels=[reader, writer],
        dataflow_buffers=[ring],
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=[K_READER, K_WRITER], target_nodes=core_set)],
    )

    per_core = lambda value: {core: value for core in counts}  # noqa: E731
    dm_args = {"core_tiles": counts, "start_page": starts, "block_tiles": blocks}
    reader_args = {
        **dm_args,
        "head_dim_tiles": per_core(head_dim_tiles),
        "heads": per_core(heads),
        "token_rows": per_core(token_rows),
    }
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel=K_READER, runtime_arg_values=reader_args),
            E.KernelRunArgs(kernel=K_WRITER, runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {TP_INPUT: 0, TP_OUTPUT: 1}
