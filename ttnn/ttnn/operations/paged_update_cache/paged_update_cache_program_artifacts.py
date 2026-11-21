# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for paged_update_cache (regime `direct_row_write`, op_design.md).

One data-movement kernel, `row_writer`, on each core that holds users of the input's height shards; no
compute kernel (nothing is computed, rule R1) and no dataflow ring (nothing is staged). Per core and call:

    load_index_block   the positions page + this core's page-table rows -> index_scratch, one read barrier
    resolve_block      per user: skip if p == -1, else (cache tile-row base, row in tile)
    write_rows_block   every non-skipped user x head x D-tile: two face-row segments, resident shard ->
                       the cache page at a byte offset, one write barrier per WRITE_TXN_CAP transfers
    release_tags       hand the hart's transaction tag back

Thread plan (R, C, W) = (0, 0, ROW_WRITER_THREADS). The kernel deals its (head, D-tile) units round-robin over
its threads, each thread reading the index block into its own scratch slot.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

TILE_DIM = 32
FACE_DIM = 16

# The thread plan: one writer thread, no reader thread, no compute thread (op_design.md, Work Distribution).
ROW_WRITER_THREADS = 1
# Writes issued between two barriers on one transaction id: half the id's outstanding-transaction counter
# (NOC_MAX_TRANSACTION_ID_COUNT = 0xFFFF).
WRITE_TXN_CAP = 32768
# Headroom kept off the unreserved L1 for co-resident tensors (the input shard) and the program's own allocations.
L1_RESERVE_BYTES = 262144

K_ROW_WRITER = "row_writer"
TP_INPUT = "input"
TP_POSITIONS = "positions"
TP_PAGE_TABLE = "page_table"
TP_CACHE = "cache"
SCRATCH_INDEX = "index_scratch"


def _align_up(value, alignment):
    return -(-value // alignment) * alignment


def _shard_layout(input_tensor):
    """(data cores in shard order, users per shard, users) of a height-sharded input."""
    shard_spec = input_tensor.memory_config().shard_spec
    users = int(input_tensor.shape[1])
    # Each user is one tile row of the input (H <= 32, padded per image), so a shard of height S holds S / 32 users.
    users_per_shard = int(shard_spec.shape[0]) // TILE_DIM
    row_major = shard_spec.orientation == ttnn.ShardOrientation.ROW_MAJOR
    cores = list(ttnn.corerange_to_cores(shard_spec.grid, None, row_major))
    num_data_cores = -(-users // users_per_shard)
    return cores[:num_data_cores], users_per_shard, users


def _index_slot_bytes(input_tensor, update_idxs_tensor, page_table):
    """(positions page bytes, page-table row bytes, one thread's scratch slot bytes)."""
    _, users_per_shard, _ = _shard_layout(input_tensor)
    pos_page_bytes = update_idxs_tensor.buffer_aligned_page_size()
    pt_row_bytes = page_table.buffer_aligned_page_size()
    return pos_page_bytes, pt_row_bytes, pos_page_bytes + users_per_shard * pt_row_bytes


def index_scratch_bytes(input_tensor, update_idxs_tensor, page_table):
    """The index scratchpad per core: one slot (positions page + the shard's page-table rows) per writer thread."""
    return ROW_WRITER_THREADS * _index_slot_bytes(input_tensor, update_idxs_tensor, page_table)[2]


def l1_index_budget():
    return ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES


def create_program_artifacts(cache_tensor, input_tensor, update_idxs_tensor, page_table):
    blocks, heads, block_size, D = (int(d) for d in cache_tensor.shape)
    dtiles = D // TILE_DIM
    element_bytes = cache_tensor.element_size()
    face_row_bytes = FACE_DIM * element_bytes
    face_bytes = FACE_DIM * FACE_DIM * element_bytes

    data_cores, users_per_shard, users = _shard_layout(input_tensor)
    pos_page_bytes, pt_row_bytes, slot_bytes = _index_slot_bytes(input_tensor, update_idxs_tensor, page_table)

    row_writer = E.KernelSpec(
        unique_id=K_ROW_WRITER,
        source=str(KERNEL_DIR / "paged_update_cache_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=ROW_WRITER_THREADS,
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name=TP_INPUT, accessor_name=TP_INPUT),
            E.TensorBinding(tensor_parameter_name=TP_POSITIONS, accessor_name=TP_POSITIONS),
            E.TensorBinding(tensor_parameter_name=TP_PAGE_TABLE, accessor_name=TP_PAGE_TABLE),
            E.TensorBinding(tensor_parameter_name=TP_CACHE, accessor_name=TP_CACHE),
        ],
        scratchpad_bindings=[
            E.ScratchpadBinding(scratchpad_spec_name=SCRATCH_INDEX, accessor_name=SCRATCH_INDEX),
        ],
        compile_time_args={
            "tile_rows": TILE_DIM,
            "face_rows": FACE_DIM,
            "writer_threads": ROW_WRITER_THREADS,
            "heads": heads,
            "block_size_tiles": block_size // TILE_DIM,
            "dtiles": dtiles,
            "face_row_bytes": face_row_bytes,
            "face_bytes": face_bytes,
            "input_page_bytes": input_tensor.buffer_aligned_page_size(),
            "pos_page_bytes": pos_page_bytes,
            "pt_row_bytes": pt_row_bytes,
            "slot_bytes": slot_bytes,
            "writes_per_barrier": WRITE_TXN_CAP,
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["first_user", "num_users_this_core"]),
    )

    spec = E.ProgramSpec(
        name="paged_update_cache",
        kernels=[row_writer],
        scratchpads=[E.ScratchpadSpec(unique_id=SCRATCH_INDEX, size_per_node=ROW_WRITER_THREADS * slot_bytes)],
        tensor_parameters=[
            E.TensorParameter(unique_id=TP_INPUT, spec=input_tensor.spec),
            E.TensorParameter(unique_id=TP_POSITIONS, spec=update_idxs_tensor.spec),
            E.TensorParameter(unique_id=TP_PAGE_TABLE, spec=page_table.spec),
            E.TensorParameter(unique_id=TP_CACHE, spec=cache_tensor.spec),
        ],
        work_units=[
            E.WorkUnitSpec(name="main", kernels=[K_ROW_WRITER], target_nodes=ttnn.CoreRangeSet(
                [ttnn.CoreRange(core, core) for core in data_cores]))
        ],
    )

    first_user = {core: i * users_per_shard for i, core in enumerate(data_cores)}
    num_users = {core: min(users_per_shard, users - first_user[core]) for core in data_cores}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel=K_ROW_WRITER,
                runtime_arg_values={"first_user": first_user, "num_users_this_core": num_users},
            )
        ]
    )
    return spec, run_args, {TP_INPUT: 0, TP_POSITIONS: 1, TP_PAGE_TABLE: 2, TP_CACHE: 3}
