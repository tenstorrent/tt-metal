# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for clone (regime `interleaved_page_stream`, op_design.md).

Every engaged cluster copies one contiguous range of the input's pages to the same range of the
output's pages (page p -> page p; a page is a tile in TILE layout, a row in ROW_MAJOR):

    reader (R harts)  input page start+i -> ring `staged_pages`   (implicit-sync tagged read)
    writer (W harts)  ring `staged_pages` -> output page start+i  (implicit-sync tagged write)

The ring is STRIDED at both ends with R = W, so reader hart t and writer hart t share one tile counter
and both handle the cluster's pages t, t+R, ... The unit dealt to harts is the ring entry itself, so a
ragged last round is legal. There is no compute kernel: nothing is computed, and a float32 page must
not pass through DEST (R1).

Knobs, each defined once here and passed to the kernels as compile-time / runtime args:
    THREAD_PLAN              (R, W) harts per cluster
    RING_ENTRIES_PER_THREAD  ring depth per reader/writer hart pair
    grid                     device.compute_with_storage_grid_size() unless overridden
Derived: entry_bytes (from the page size), core_pages / start_page (from the split).
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# The thread plan (R readers, W writers) per cluster. Bound by data movement: one hart of each kind
# with implicit sync reaches the DRAM floor (target page, "Choosing the plan"). No compute kernel.
THREAD_PLAN = (1, 1)
# Ring depth per reader/writer hart pair: >= 2 implicit-sync transaction-id slices of >= 2 entries.
RING_ENTRIES_PER_THREAD = 4

# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query.
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 the remapper
DFB_MAX_RING_BYTES = 1_048_560  # one ring, at most 1 MiB - 16 B

DFB_STAGED = "staged_pages"
TP_INPUT = "input"
TP_OUTPUT = "output"
K_READER = "reader"
K_WRITER = "writer"


def entry_bytes_of(tensor) -> int:
    """One ring entry = one whole page, rounded up to L1 alignment (the DFB stores entry sizes in
    L1-alignment units). Never one side's slot size: a 64-B DRAM slot would overrun a 16-B L1 slot."""
    align = ttnn.get_l1_alignment()
    page_bytes = tensor.buffer_page_size()
    return (page_bytes + align - 1) // align * align


def validate_thread_plan(threads):
    r, w = threads
    if not (r == w and r >= 1 and r + w <= FREE_DM_HARTS):
        raise ValueError(f"clone thread plan (R, W) = {threads}: need R = W and R + W <= {FREE_DM_HARTS}")


def ring_entries_per_thread_for(knob: int, entry_bytes: int, threads: int) -> int:
    """The depth knob clamped to the DFB ring cap, kept even (two transaction-id slices that divide
    evenly over the harts) and at least 2."""
    cap = DFB_MAX_RING_BYTES // (entry_bytes * threads)
    depth = min(knob, cap)
    depth -= depth % 2
    if depth < 2:
        raise NotImplementedError(f"clone: a ring of 2 entries of {entry_bytes} B per hart does not fit one DFB")
    return depth


def split_pages(tensor_pages: int, grid_size, threads: int):
    """(core set, {core: pages}, {core: first page}) over the clusters that get work, row-major.

    Pages are dealt to clusters in whole rounds of `threads` pages, so only the last engaged cluster can
    end on a short round. `tensor_pages` comes from buffer_num_pages(), which counts tile padding.
    """
    rounds = (tensor_pages + threads - 1) // threads
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid_size, rounds, row_wise=True)
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            pages = min(per_core * threads, tensor_pages - start)
            counts[core], starts[core] = pages, start
            start += pages
    assert start == tensor_pages, (start, tensor_pages)
    return all_cores, counts, starts


def create_program_artifacts(
    input, output, *, threads=THREAD_PLAN, ring_entries_per_thread=RING_ENTRIES_PER_THREAD, grid=None
):
    """(ProgramSpec, ProgramRunArgs, tensor_indices) for one ttnn.generic_op([input, output], ...) call.

    `output` is pre-allocated by the caller. The keyword knobs default to the module constants; tests and
    perf lamps turn them without editing code.
    """
    validate_thread_plan(threads)
    R, W = threads

    tensor_pages = input.buffer_num_pages()
    if output.buffer_num_pages() != tensor_pages or output.buffer_page_size() != input.buffer_page_size():
        raise ValueError(
            f"clone: page geometry differs: input {tensor_pages} x {input.buffer_page_size()} B, "
            f"output {output.buffer_num_pages()} x {output.buffer_page_size()} B"
        )
    entry_bytes = entry_bytes_of(input)
    # The write of a whole entry must stay inside the output page's slot (and the read inside the input's).
    assert entry_bytes <= min(input.buffer_aligned_page_size(), output.buffer_aligned_page_size()), (
        entry_bytes,
        input.buffer_aligned_page_size(),
        output.buffer_aligned_page_size(),
    )
    depth = ring_entries_per_thread_for(ring_entries_per_thread, entry_bytes, R)

    if grid is None:
        grid = input.device().compute_with_storage_grid_size()
    core_set, counts, starts = split_pages(tensor_pages, grid, R)

    plan = {"reader_threads": R, "writer_threads": W}
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["core_pages", "start_page"])

    reader = E.KernelSpec(
        unique_id=K_READER,
        source=str(KERNEL_DIR / "clone_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        dfb_bindings=[E.producer_of(DFB_STAGED, "staged_pages")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_INPUT, accessor_name="input")],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    writer = E.KernelSpec(
        unique_id=K_WRITER,
        source=str(KERNEL_DIR / "clone_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of(DFB_STAGED, "staged_pages")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_OUTPUT, accessor_name="output")],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name="clone",
        kernels=[reader, writer],
        dataflow_buffers=[
            E.DataflowBufferSpec(
                unique_id=DFB_STAGED,
                entry_size=entry_bytes,
                num_entries=depth * R,
                data_format_metadata=input.dtype,
            )
        ],
        tensor_parameters=[
            E.TensorParameter(unique_id=TP_INPUT, spec=input.spec),
            E.TensorParameter(unique_id=TP_OUTPUT, spec=output.spec),
        ],
        work_units=[E.WorkUnitSpec(name="main", kernels=[K_READER, K_WRITER], target_nodes=core_set)],
    )
    dm_args = {"core_pages": counts, "start_page": starts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel=K_READER, runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel=K_WRITER, runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {TP_INPUT: 0, TP_OUTPUT: 1}
