# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for expand (regime R-H: tile layout, rows repeated).

Read once, build once, write many. The output's tiles are numbered source-major,

    L = s * fan_out_tiles + r,   s = p * tensor_w_tiles + j   (source tile s, output tile row r)

so a source tile's whole fan-out (the output tile rows it feeds) is contiguous in L. Every cluster takes one
contiguous run of L (`ttnn.split_work_to_cores`). Its reader reads each source tile of the run once from DRAM
into a ring entry and replicates the tile's row 0 over its 32 rows in place (self-aimed NoC copies); its
writer writes that one entry to every output tile of the run the source feeds. No compute kernel: the op
moves bytes and computes nothing, so a float32 value never passes a narrower format (R1).

Thread plan (R, C, W) = (1, 0, 1): the op is bound by data movement (DRAM writes), for which the target page's
*Choosing the plan* gives one thread of each kind; there is no compute kernel, hence no NEO.

Knobs (each defined once here; every dependent quantity derives from them):
    block_src_tiles  source tiles resident per block (ring entries per block), default: all of the busiest
                     cluster's sources, clamped by the ring cap
    buffer_depth     blocks in flight in the ring (clamped to the most blocks any cluster runs)
    max_cores        clusters engaged (default: the whole queried grid)
"""

from math import prod
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

TILE = 32

# Thread plan (R readers, C compute threads, W writers): the one host constant the kernels' counts come from.
THREAD_PLAN = (1, 0, 1)

# Blocks in flight in the ring (double buffering: the reader builds block b+1 while the writer fans out b).
BUFFER_DEPTH = 2

# A DataflowBuffer ring holds at most 1 MiB - 16 B (target page, cluster model); Python has no query for it.
DFB_RING_MAX_BYTES = 1_048_560

# L1 held back from the budget the ring may spend.
L1_RESERVE_BYTES = 0

# The ring's name, as the kernels see it (`dfb::cb_row_bcast_tiles`).
RING = "cb_row_bcast_tiles"


def split_runs(total_tiles, device, max_cores=None):
    """(core set, {core: run tiles}, {core: run start}) over the clusters that get work, row-major."""
    grid = device.compute_with_storage_grid_size()
    if max_cores is not None:
        grid = ttnn.num_cores_to_corerangeset(min(max_cores, grid.x * grid.y), grid, True)
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, total_tiles, row_wise=True)
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    assert start == total_tiles, (start, total_tiles)
    return all_cores, counts, starts


def sources_of_run(run_start, run_tiles, fan_out_tiles):
    """How many source tiles a run of L touches."""
    return (run_start + run_tiles - 1) // fan_out_tiles - run_start // fan_out_tiles + 1


def block_src_tiles_cap(page_size, buffer_depth):
    """The largest block the ring can hold at this depth: ring maximum and the L1 budget."""
    l1_budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
    return max(1, min(DFB_RING_MAX_BYTES, l1_budget) // (buffer_depth * page_size))


def geometry(output, result_shape):
    """The index space of regime R-H (see the module docstring)."""
    h_out, w = int(result_shape[-2]), int(result_shape[-1])
    num_planes = prod(int(d) for d in result_shape[:-2])
    tensor_w_tiles = -(-w // TILE)
    output_h_tiles = -(-h_out // TILE)  # per plane
    total_out_tiles = num_planes * output_h_tiles * tensor_w_tiles
    assert total_out_tiles == output.buffer_num_pages(), (total_out_tiles, output.buffer_num_pages())
    assert total_out_tiles < 2**32
    return num_planes, tensor_w_tiles, output_h_tiles, total_out_tiles


def create_program_artifacts(
    input_tensor, output, result_shape, *, block_src_tiles=None, buffer_depth=BUFFER_DEPTH, max_cores=None
):
    """(ProgramSpec, ProgramRunArgs, tensor indices) for one `ttnn.generic_op([input, output], ...)`.

    `block_src_tiles`: an upper bound on the block (tests force small blocks to exercise the multi-block path
    and the ring wrap); by default every source of the busiest cluster's run, clamped by the ring cap."""
    R, C, W = THREAD_PLAN
    page_size = output.buffer_page_size()
    assert input_tensor.buffer_page_size() == page_size
    _, tensor_w_tiles, output_h_tiles, total_out_tiles = geometry(output, result_shape)
    fan_out_tiles = output_h_tiles

    core_set, counts, starts = split_runs(total_out_tiles, input_tensor.device(), max_cores)
    src_tiles = {c: sources_of_run(starts[c], counts[c], fan_out_tiles) for c in counts}
    max_src_tiles_per_core = max(src_tiles.values())
    block = min(max_src_tiles_per_core, block_src_tiles_cap(page_size, buffer_depth))
    if block_src_tiles is not None:
        block = max(1, min(block, int(block_src_tiles)))
    # Blocks in flight never exceed the blocks a cluster runs: with one block per cluster (the default whenever
    # the busiest run fits the cap) a second ring half would never be live.
    max_blocks_per_core = max(-(-n // block) for n in src_tiles.values())
    depth = min(int(buffer_depth), max_blocks_per_core)

    ring = E.DataflowBufferSpec(
        unique_id=RING,
        entry_size=page_size,
        num_entries=depth * block,  # a multiple of the block: no block straddles the wrap
        data_format_metadata=input_tensor.dtype,
    )
    tensor_parameters = [
        E.TensorParameter(unique_id="input", spec=input_tensor.spec),
        E.TensorParameter(unique_id="output", spec=output.spec),
    ]
    common_ct = {
        "block_src_tiles": block,
        "fan_out_tiles": fan_out_tiles,
        "tensor_w_tiles": tensor_w_tiles,
        "reader_threads": R,
        "writer_threads": W,
    }
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["run_start", "run_tiles"])

    def dm_config(make):
        # Both kernels keep the ring's credits themselves (explicit sync): the reader assembles each entry from
        # a DRAM read and self-aimed copies; the writer writes one entry many times.
        hw = make()
        hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=[RING])
        return hw

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "expand_reader.cpp"),
        hw_config=dm_config(E.create_reader_datamovement_config),
        num_threads=R,
        dfb_bindings=[E.producer_of(RING, RING)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args=common_ct,
        runtime_arg_schema=dm_schema,
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "expand_writer.cpp"),
        hw_config=dm_config(E.create_writer_datamovement_config),
        num_threads=W,
        dfb_bindings=[E.consumer_of(RING, RING)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=common_ct,
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name="expand",
        kernels=[reader, writer],
        dataflow_buffers=[ring],
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=core_set)],
    )
    dm_args = {"run_start": starts, "run_tiles": counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}
