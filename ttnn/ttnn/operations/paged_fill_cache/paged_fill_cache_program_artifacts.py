# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for paged_fill_cache (op_design.md: regime R-filter).

Per cluster: one reader and one writer data-movement kernel, no compute kernel (the op moves tiles and
computes nothing, rule R1).

    reader  page_table row `batch_idx` -> dfb_page_table_row (2 copies, explicit sync, user trid 0):
            entry 0 pushed to the writer at once, entry 1 kept reserved as the reader's lookup table;
            then the cluster's input tiles [core_start_tile, +core_tiles) in input order, skipping the
            tile rows whose page-table entry is -1, one implicit-sync read each into dfb_input_tiles;
            then entry 1 is pushed.
    writer  waits for entry 0, walks the same range with the same skip, one implicit-sync write per
            tile to its cache page id; then pops both row entries.

Work split: the input's tiles (heads x seq tiles x D tiles, flattened in page order) over the full grid,
`split_unit_tiles` at a time (split_work_to_cores, row-wise). A cluster's block is its whole contiguous
range, streamed through a TILE_RING_ENTRIES window. Every shape quantity is a runtime argument (common
where it is the same on every cluster), so one compiled kernel pair serves every shape.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- Knobs (single sources; op_design.md "Knob summary") -------------------------------------------
# Thread plan (R readers, C compute threads, W writers) per cluster. Data-movement bound copy: one reader
# and one writer with implicit sync, no compute kernel (C = 0).
THREAD_PLAN = (1, 0, 1)
# Tiles per core-assignment unit of the work split.
SPLIT_UNIT_TILES = 1
# Depth of the reader -> writer tile ring, per reader thread. Even: the runtime splits it into two
# transaction-id halves (implicit-sync double buffer).
TILE_RING_ENTRIES = 8
# Page-table row copies per reader thread: one for the writer, one the reader keeps (structural).
PT_RING_ENTRIES = 2

# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them.
FREE_DM_HARTS = 6
RING_MAX_BYTES = 1_048_560
TILE = 32

DFB_INPUT_TILES = "input_tiles"
DFB_PAGE_TABLE_ROW = "page_table_row"
TP_INPUT, TP_PAGE_TABLE, TP_CACHE = "input", "page_table", "cache"
K_READER, K_WRITER = "reader", "writer"


# Plans that have been run. The kernels stride their walks by the plan's counts (CT args), but
# (2, 0, 2) on STRIDED DM->DM rings is not admitted
# (probe_001.py: [1,2,160,1024] input, page table [[0,0,0],[5,-1,2]]): reader thread p's entries are not
# shown to reach writer thread p on a ring with no compute endpoint. A refinement measuring the design's
# thread-plan lamp must resolve that before admitting R = W > 1 here.
VERIFIED_THREAD_PLANS = {(1, 0, 1)}


def validate_thread_plan(threads):
    r, c, w = threads
    # The page-table ring hands reader thread p's copy to writer thread p: R == W. C = 0: no compute kernel.
    if not (c == 0 and r >= 1 and r == w and r + w <= FREE_DM_HARTS):
        raise ValueError(f"paged_fill_cache thread plan (R, C, W) = {threads}: need C = 0, R = W, R + W <= {FREE_DM_HARTS}")
    if tuple(threads) not in VERIFIED_THREAD_PLANS:
        raise ValueError(f"paged_fill_cache thread plan {threads} has not been run; verified: {VERIFIED_THREAD_PLANS}")
    if TILE_RING_ENTRIES % (2 * r) != 0:
        raise ValueError(f"TILE_RING_ENTRIES = {TILE_RING_ENTRIES} must split into 2 txn-id halves of whole rounds")


def split_tiles(num_tiles, grid_size, unit_tiles=SPLIT_UNIT_TILES):
    """(core set, {core: tiles}, {core: first tile}) over the clusters that get work, row-major.

    The split is in units of `unit_tiles`; the last unit may be short."""
    num_units = -(-num_tiles // unit_tiles)
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_units, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            tiles = min(per_core * unit_tiles, num_tiles - start)
            counts[core], starts[core] = tiles, start
            start += tiles
    return all_cores, counts, starts


def page_table_entry_bytes(page_table):
    """Entry size of the page-table row ring: the row's page, padded to the DRAM alignment (the row is read
    from DRAM straight into each entry). The one place this is derived: validate()'s ring cap and the ring
    itself both use it."""
    dram_alignment = ttnn.get_dram_alignment()
    return -(-page_table.buffer_page_size() // dram_alignment) * dram_alignment


def _dm_config(make, explicit_rings):
    hw = make()
    hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=explicit_rings)
    return hw


def create_program_artifacts(cache_tensor, input_tensor, page_table, batch_idx, *, threads=THREAD_PLAN):
    validate_thread_plan(threads)
    R, _, W = threads
    _, heads, block_size, head_dim = (int(d) for d in cache_tensor.shape)
    seq = int(input_tensor.shape[2])
    seq_tiles, block_tiles, d_tiles = -(-seq // TILE), -(-block_size // TILE), -(-head_dim // TILE)
    tensor_input_tiles = heads * seq_tiles * d_tiles

    tile_bytes = input_tensor.buffer_page_size()
    pt_page_bytes = page_table.buffer_page_size()
    pt_entry_bytes = page_table_entry_bytes(page_table)
    assert TILE_RING_ENTRIES * R * tile_bytes <= RING_MAX_BYTES
    assert PT_RING_ENTRIES * R * pt_entry_bytes <= RING_MAX_BYTES

    core_set, counts, starts = split_tiles(tensor_input_tiles, input_tensor.device().compute_with_storage_grid_size())

    dfbs = [
        E.DataflowBufferSpec(
            unique_id=DFB_INPUT_TILES,
            entry_size=tile_bytes,
            num_entries=TILE_RING_ENTRIES * R,
            data_format_metadata=input_tensor.dtype,
        ),
        E.DataflowBufferSpec(
            unique_id=DFB_PAGE_TABLE_ROW,
            entry_size=pt_entry_bytes,
            num_entries=PT_RING_ENTRIES * R,
            data_format_metadata=ttnn.int32,
        ),
    ]
    tensor_parameters = [
        E.TensorParameter(unique_id=TP_INPUT, spec=input_tensor.spec),
        E.TensorParameter(unique_id=TP_PAGE_TABLE, spec=page_table.spec),
        E.TensorParameter(unique_id=TP_CACHE, spec=cache_tensor.spec),
    ]
    # Compile-time args: the plan's counts and the row-copy count, so the kernels' reserve / push / pop
    # counts on the row ring come from the same constant as the ring's size.
    plan = {"reader_threads": R, "writer_threads": W, "page_table_row_copies": PT_RING_ENTRIES}
    walk_args = ["num_heads", "seq_tiles", "block_tiles", "d_tiles"]
    walk_values = {"num_heads": heads, "seq_tiles": seq_tiles, "block_tiles": block_tiles, "d_tiles": d_tiles}
    per_core = ["core_start_tile", "core_tiles"]

    reader = E.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "paged_fill_cache_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, [DFB_PAGE_TABLE_ROW]),
        num_threads=R,
        dfb_bindings=[
            E.producer_of(DFB_INPUT_TILES, "input_tiles"),
            E.producer_of(DFB_PAGE_TABLE_ROW, "page_table_row"),
        ],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name=TP_INPUT, accessor_name="input"),
            E.TensorBinding(tensor_parameter_name=TP_PAGE_TABLE, accessor_name="page_table"),
        ],
        compile_time_args=plan,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=per_core, common_runtime_arg_names=walk_args + ["batch_idx", "pt_page_bytes"]
        ),
    )
    writer = E.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "paged_fill_cache_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, [DFB_PAGE_TABLE_ROW]),
        num_threads=W,
        dfb_bindings=[
            E.consumer_of(DFB_INPUT_TILES, "input_tiles"),
            E.consumer_of(DFB_PAGE_TABLE_ROW, "page_table_row"),
        ],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_CACHE, accessor_name="cache")],
        compile_time_args=plan,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=per_core, common_runtime_arg_names=walk_args),
    )
    spec = E.ProgramSpec(
        name="paged_fill_cache",
        kernels=[reader, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=[K_READER, K_WRITER], target_nodes=core_set)],
    )
    dm_args = {"core_start_tile": starts, "core_tiles": counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel=K_READER,
                runtime_arg_values=dm_args,
                common_runtime_arg_values={**walk_values, "batch_idx": batch_idx, "pt_page_bytes": pt_page_bytes},
            ),
            E.KernelRunArgs(kernel=K_WRITER, runtime_arg_values=dm_args, common_runtime_arg_values=walk_values),
        ]
    )
    # io_tensors = [input_tensor, page_table, cache_tensor]: the cache is last, the tensor generic_op returns.
    return spec, run_args, {TP_INPUT: 0, TP_PAGE_TABLE: 1, TP_CACHE: 2}
