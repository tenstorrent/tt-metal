# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for typecast (op_design.md, regime `tile_stream_float`).

    reader  --cb_input_tiles-->  compute (unpack -> 32-bit DEST -> pack in the output format)  --cb_output_tiles-->  writer

Blocking model: one axis, the flat tile (page) index of the tensor. It is split into contiguous ranges over the
clusters (`split_work_to_cores`, row-major); within a cluster tiles are dealt round-robin to the plan's threads
(STRIDED rings), and each NEO converts its share in blocks of `block_tiles`, with `BUFFER_DEPTH` blocks in flight
on each ring.

Knobs, each defined once:
    THREAD_PLAN    (R, C, W) reader / compute / writer threads per cluster (kernel CT args)
    BUFFER_DEPTH   blocks in flight per ring per NEO
    block_tiles    derived per call (`choose_block_tiles`), clamped by the DEST section, ring and L1 caps;
                   compute CT arg
    grid           the device's compute grid (query)
Ring capacities are `BUFFER_DEPTH * block_tiles * C` entries; nothing is sized from a whole-tensor dimension.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# Thread plan (R readers, C NEOs, W writers). The op is bound by data movement (a format conversion per tile,
# no SFPU arithmetic for the float pairs): one thread of each kind.
# One constant: the kernels get the counts as compile-time args.
THREAD_PLAN = (1, 1, 1)
# Blocks in flight on each ring, per NEO: reader fills block b+1 while compute converts block b.
BUFFER_DEPTH = 2
# L1 kept back from the ring budget for everything that is not a ring.
L1_RESERVE_BYTES = 65_536
# Cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them.
DFB_RING_MAX_BYTES = 1_048_560  # compute side, per ring
NEOS = 4
FREE_DM_HARTS = 6
# DEST configuration of the compute kernel: 32-bit (exact fp32 and an exact bf16 widen), double buffered (half
# sync). One source for the ComputeHardwareConfig and for the block cap below.
ENABLE_32_BIT_DEST = True
DOUBLE_BUFFER_DEST = True

CB_INPUT = "cb_input_tiles"
CB_OUTPUT = "cb_output_tiles"


def validate_thread_plan(threads):
    r, c, w = threads
    if not (c in (1, 2, 4) and c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(f"thread plan (R, C, W) = {threads}: need C in {{1,2,4}}, R + W <= {FREE_DM_HARTS}, R | C, W | C")


def split_tiles(num_tiles, grid_size):
    """(core set, {core: tiles}, {core: first tile}) over the clusters that get work, row-major."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid_size, num_tiles, row_wise=True)
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    return all_cores, counts, starts


def dest_tiles(enable_32_bit_dest=ENABLE_32_BIT_DEST, double_buffer_dest=DOUBLE_BUFFER_DEST):
    """Tiles one DEST section holds (kernel_lib dest_helpers.hpp get_dest_limit)."""
    full = 8 if enable_32_bit_dest else 16
    return full // 2 if double_buffer_dest else full


def block_tiles_cap(in_page, out_page, compute_threads, buffer_depth):
    """Largest block that fits a DEST section, the ring size limit and the L1 budget.

    DEST: the eltwise chain clamps a PerBlockSize block to its DEST capacity at run time (kernel_lib
    eltwise/core/chain.inl, `block_size = max_block`) and then synchronises the rings in windows of that size.
    A larger block would leave the ring capacity (`buffer_depth * block_tiles`) not a multiple of the window,
    so a window would straddle the ring's end.
    """
    l1_budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
    per_block_ring = compute_threads * buffer_depth * max(in_page, out_page)
    per_block_l1 = compute_threads * buffer_depth * (in_page + out_page)
    return max(1, min(dest_tiles(), DFB_RING_MAX_BYTES // per_block_ring, l1_budget // per_block_l1))


def choose_block_tiles(core_tiles_max, in_page, out_page, compute_threads, buffer_depth):
    """The coarsest block that still keeps `buffer_depth` blocks in flight on the busiest NEO, clamped by the caps."""
    neo_tiles_max = -(-core_tiles_max // compute_threads)
    want = -(-neo_tiles_max // buffer_depth)
    return max(1, min(want, block_tiles_cap(in_page, out_page, compute_threads, buffer_depth)))


def unpack_mode_for(dtype):
    # 32-bit input: unpack straight to the 32-bit DEST, all 32 bits (SrcA would keep 19).
    # bfloat16 input: through SrcA; the A2D copy widens exactly.
    return E.UnpackMode.UnpackToDest if dtype == ttnn.float32 else E.UnpackMode.UnpackToSrc


def create_program_artifacts(input_tensor, output, *, threads=THREAD_PLAN, buffer_depth=BUFFER_DEPTH, block_tiles=None):
    """`threads`, `buffer_depth` and `block_tiles` default to the op's knobs; tests turn them."""
    validate_thread_plan(threads)
    R, C, W = threads
    num_tiles = output.buffer_num_pages()
    in_page = input_tensor.buffer_page_size()
    out_page = output.buffer_page_size()
    core_set, counts, starts = split_tiles(num_tiles, input_tensor.device().compute_with_storage_grid_size())
    core_tiles_max = max(counts.values())
    if block_tiles is None:
        block_tiles = choose_block_tiles(core_tiles_max, in_page, out_page, C, buffer_depth)
    if not 1 <= block_tiles <= block_tiles_cap(in_page, out_page, C, buffer_depth):
        raise ValueError(f"block_tiles={block_tiles} outside 1..{block_tiles_cap(in_page, out_page, C, buffer_depth)}")
    entries_per_neo = buffer_depth * block_tiles

    # A STRIDED ring is one sub-ring per NEO: capacity counted per NEO.
    dfbs = [
        E.DataflowBufferSpec(
            unique_id=CB_INPUT, entry_size=in_page, num_entries=entries_per_neo * C,
            data_format_metadata=input_tensor.dtype,
        ),
        E.DataflowBufferSpec(
            unique_id=CB_OUTPUT, entry_size=out_page, num_entries=entries_per_neo * C,
            data_format_metadata=output.dtype,
        ),
    ]
    tensor_parameters = [
        E.TensorParameter(unique_id="input", spec=input_tensor.spec),
        E.TensorParameter(unique_id="output", spec=output.spec),
    ]
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"])

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "typecast_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        dfb_bindings=[E.producer_of(CB_INPUT, CB_INPUT)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "typecast_compute.cpp"),
        hw_config=E.ComputeHardwareConfig(
            # A datacopy has no multiply: LoFi. Set explicitly.
            fpu_math_fidelity=ttnn.MathFidelity.LoFi,
            enable_32_bit_dest=ENABLE_32_BIT_DEST,
            double_buffer_dest=DOUBLE_BUFFER_DEST,
            unpack_modes={CB_INPUT: unpack_mode_for(input_tensor.dtype)},
        ),
        num_threads=C,
        dfb_bindings=[E.consumer_of(CB_INPUT, CB_INPUT), E.producer_of(CB_OUTPUT, CB_OUTPUT)],
        compile_time_args={**plan, "block_tiles": int(block_tiles)},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "typecast_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of(CB_OUTPUT, CB_OUTPUT)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name="typecast",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )
    dm_args = {"num_tiles": counts, "start_id": starts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_tiles": counts}),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}
