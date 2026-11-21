# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for sigmoid (regime R-stream of op_design.md).

Work distribution. The tile pages of the tensor (`batch * ceil(H/32) * ceil(W/32)`, page order = row-major
tile order) form one flat, independent `page` axis. It is dealt to the clusters of the grid in whole
*rounds* of C pages (C = the plan's compute threads): each cluster gets a contiguous run of rounds, which is
its block, and only the cluster holding the last round is shorter.

Thread plan (R, C, W) on every cluster. Both rings are STRIDED, so with the cluster's pages numbered
0..n-1 in ring order

    reader thread p   reads    pages p, p+R, ...
    NEO t             computes pages t, t+C, ...   in sync blocks of `block_tiles` (one DEST window)
    writer thread w   writes   pages w, w+W, ...

Knobs, each defined once here: THREAD_PLAN, BUFFER_DEPTH, DEST_TILES (-> block_tiles). Every ring size,
loop bound and core count derives from them and from the tensor, never from a literal in a kernel.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- knobs -----------------------------------------------------------------------------------------------
# (R readers, C NEO compute threads, W writers) per cluster. Compute-bound op (accurate SFPU body, ~35
# instructions per row), one input, one output: C = 4, R = W = 2 (target page, *Choosing the plan*).
THREAD_PLAN = (2, 4, 2)
# Sync blocks of slack per NEO in each ring: 2 = the reader fills block k+1 while the NEO computes block k.
BUFFER_DEPTH = 2
# Tiles one DEST window holds, keyed by (fp32_dest_acc_en, dst_full_sync_en): half-sync (the default,
# double_buffer_dest=True) halves DEST so the packer drains one half while math fills the other. block_tiles =
# the coarsest sync block DEST holds; the chain clamps to the same bound (kernel_lib/dest_helpers.hpp,
# get_dest_limit()).
DEST_TILES = {(True, False): 4, (False, False): 8, (True, True): 8, (False, True): 16}

# --- the cluster model of the target page (.claude/references/targets/quasar.md); no Python query exists ---
FREE_DM_THREADS = 6  # of 8 DM cores: DM0 serves the ring interrupts, DM1 is the remapper
NEOS = 4
MAX_RING_BYTES = 1_048_560  # one DFB ring, compute side


def validate_thread_plan(threads):
    r, c, w = threads
    if not (c in (1, 2, 4) and c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_THREADS and c % r == 0 and c % w == 0):
        raise ValueError(
            f"thread plan (R, C, W) = {threads}: need C in {{1, 2, 4}}, R + W <= {FREE_DM_THREADS}, R | C, W | C"
        )


def split_rounds(num_tiles, compute_threads, grid_size):
    """(core set, {core: pages}, {core: first page}): whole rounds of `compute_threads` pages per cluster,
    contiguous, row-major over the grid. Only the cluster holding the final round may get fewer pages."""
    num_rounds = -(-num_tiles // compute_threads)  # ceil
    _, all_cores, group_1, group_2, rounds_1, rounds_2 = ttnn.split_work_to_cores(grid_size, num_rounds, row_wise=True)
    counts, starts, start_round = {}, {}, 0
    for group, rounds in ((group_1, rounds_1), (group_2, rounds_2)):
        if rounds == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            start_id = start_round * compute_threads
            counts[core] = min(rounds * compute_threads, num_tiles - start_id)
            starts[core] = start_id
            start_round += rounds
    return all_cores, counts, starts


def _sfpu_precision(math_approx_mode):
    return E.Precision.Approximate if math_approx_mode else E.Precision.Precise


def create_program_artifacts(input_tensor, output, compute_kernel_config, *, threads=THREAD_PLAN, probe_tag_neo=False):
    """`threads` and `probe_tag_neo` are for the thread-plan tests only (the public entry point never sets
    them): `probe_tag_neo` makes odd NEOs compute sigmoid(-x), so a test can see which NEO computed a page."""
    validate_thread_plan(threads)
    R, C, W = threads
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)
    full_sync = bool(compute_kernel_config.dst_full_sync_en)
    block_tiles = DEST_TILES[(fp32_dest, full_sync)]

    num_tiles = input_tensor.buffer_num_pages()
    in_page = input_tensor.buffer_page_size()
    out_page = output.buffer_page_size()
    core_set, counts, starts = split_rounds(num_tiles, C, input_tensor.device().compute_with_storage_grid_size())

    # A STRIDED ring is sliced into one sub-ring per NEO: capacity counts per NEO.
    entries = BUFFER_DEPTH * block_tiles * C
    for page in (in_page, out_page):
        if entries * page > MAX_RING_BYTES:
            raise ValueError(f"sigmoid ring of {entries} x {page} B exceeds {MAX_RING_BYTES} B")
    dfbs = [
        E.DataflowBufferSpec(
            unique_id="cb_input_tiles", entry_size=in_page, num_entries=entries, data_format_metadata=input_tensor.dtype
        ),
        E.DataflowBufferSpec(
            unique_id="cb_output_tiles", entry_size=out_page, num_entries=entries, data_format_metadata=output.dtype
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
        source=str(KERNEL_DIR / "sigmoid_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        dfb_bindings=[E.producer_of("cb_input_tiles", "cb_input_tiles")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "sigmoid_compute.cpp"),
        hw_config=E.ComputeHardwareConfig(
            # Always explicit: ttsim aborts a compute kernel with the fidelity unset (library gaps).
            fpu_math_fidelity=compute_kernel_config.math_fidelity,
            sfpu_precision_mode=_sfpu_precision(compute_kernel_config.math_approx_mode),
            enable_32_bit_dest=fp32_dest,
            # The ComputeConfigDescriptor field is the inverse (tt-metal's compute_kernel_config.cpp mapping).
            double_buffer_dest=not full_sync,
        ),
        num_threads=C,
        dfb_bindings=[
            E.consumer_of("cb_input_tiles", "cb_input_tiles"),
            E.producer_of("cb_output_tiles", "cb_output_tiles"),
        ],
        compile_time_args={**plan, "block_tiles": block_tiles, "probe_tag_neo": int(bool(probe_tag_neo))},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "sigmoid_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of("cb_output_tiles", "cb_output_tiles")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name="sigmoid",
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
