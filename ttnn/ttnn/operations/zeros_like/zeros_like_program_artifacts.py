# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for zeros_like (op_design.md, regime `fill_from_local_zero_page`).

Thread plan (R, C, W) = (0, 0, 1): no reader (the input is never read, spec R1) and no compute kernel (zero
bits need no FPU/SFPU). The work unit holds the writer kernel only. On every cluster that owns output pages:

    zero_source()                      once per kernel: iDMA-zero this thread's slice of the `zero_page` scratchpad
    for block in blocks of the range:  store_block(): `pages_per_barrier` tagged page writes, one trid barrier
    release_tag()                      hand the hart's tag register back (user id 0)

Writer thread t of W takes pages t, t+W, ... of each block, from its own `zero_src_bytes` slice of the
scratchpad. Pages are split over the full grid in contiguous ranges (`split_work_to_cores`, row_wise=True).

Knobs (each defined once here; the kernel derives everything else from them):
    THREAD_PLAN         the (R, C, W) plan; W goes to the writer as `writer_threads` and as num_threads
    PAGES_PER_BARRIER   block extent on `output_page`: page writes issued between two write barriers
    ZERO_SRC_MAX_BYTES  clamp on `zero_src_bytes`, the bytes per NoC write (= one scratchpad slice)
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# The thread plan of op_design.md (Work Distribution): a writer-only program, one writer hart per cluster.
THREAD_PLAN = {"reader_threads": 0, "compute_threads": 0, "writer_threads": 1}
# The cluster model of the target page: 6 of 8 DM harts are free for user kernels (no Python query exists).
FREE_DM_HARTS = 6

# Block extent on `output_page` (pages written between two trid write barriers).
PAGES_PER_BARRIER = 256
# Clamp on the zero source (bytes per write); a multiple of the DRAM alignment, so chunk offsets stay aligned.
ZERO_SRC_MAX_BYTES = 65536
# Per-trid outstanding-transaction counter capacity (NOC_MAX_TRANSACTION_ID_COUNT on Quasar), and the largest
# NoC packet a write is split into (NOC_MAX_BURST_SIZE): the trid cap of op_design.md, Mechanism caps.
TRID_COUNTER_CAP = 0xFFFF
NOC_PACKET_BYTES = 8192


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _ceil_div(a, b):
    return (a + b - 1) // b


def split_pages(num_pages, grid_size):
    """(core set, {core: pages}, {core: first page}) over the clusters that get work, row-major."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_pages, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    assert start == num_pages, f"split covers {start} of {num_pages} pages"
    return all_cores, counts, starts


def block_parameters(page_bytes, writer_threads):
    """(zero_src_bytes, chunks_per_page, pages_per_barrier) for a page of `page_bytes` bytes."""
    zero_src_bytes = min(_align_up(page_bytes, ttnn.get_l1_alignment()), ZERO_SRC_MAX_BYTES)
    chunks_per_page = _ceil_div(page_bytes, zero_src_bytes)
    packets_per_chunk = _ceil_div(zero_src_bytes, NOC_PACKET_BYTES)
    # Writes in flight on one trid per thread between two barriers must stay below the counter's capacity.
    pages_per_barrier = max(1, min(PAGES_PER_BARRIER, TRID_COUNTER_CAP // (chunks_per_page * packets_per_chunk)))
    budget = ttnn.get_max_worker_l1_unreserved_size()
    assert writer_threads * zero_src_bytes <= budget, (writer_threads, zero_src_bytes, budget)
    return zero_src_bytes, chunks_per_page, pages_per_barrier


def create_program_artifacts(out, *, thread_plan=None):
    """ProgramSpec, run args and tensor indices that fill `out` (on device) with zeros.

    Takes only the output: a test may hand it a pre-filled tensor to see that every page gets written.
    `thread_plan` overrides THREAD_PLAN (a perf lamp: W > 1 writer harts).
    """
    plan = dict(THREAD_PLAN if thread_plan is None else thread_plan)
    W = plan["writer_threads"]
    if plan["reader_threads"] != 0 or plan["compute_threads"] != 0 or not 1 <= W <= FREE_DM_HARTS:
        raise ValueError(f"zeros_like thread plan {plan}: a writer-only plan with 1 <= W <= {FREE_DM_HARTS}")

    num_pages = out.buffer_num_pages()
    page_bytes = out.buffer_page_size()
    zero_src_bytes, _, pages_per_barrier = block_parameters(page_bytes, W)
    core_set, counts, starts = split_pages(num_pages, out.device().compute_with_storage_grid_size())

    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "zeros_like_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        scratchpad_bindings=[E.KernelSpec.ScratchpadBinding(scratchpad_spec_name="zero_page", accessor_name="zero_page")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
        compile_time_args={
            "writer_threads": W,
            "pages_per_barrier": pages_per_barrier,
            "page_bytes": page_bytes,
            "zero_src_bytes": zero_src_bytes,
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["start_page", "num_pages_this_core"]),
    )
    spec = E.ProgramSpec(
        name="zeros_like",
        kernels=[writer],
        # One slice of zero bits per writer thread; raw bytes, so no data format metadata.
        scratchpads=[E.ScratchpadSpec(unique_id="zero_page", size_per_node=W * zero_src_bytes)],
        tensor_parameters=[E.TensorParameter(unique_id="out", spec=out.spec)],
        work_units=[E.WorkUnitSpec(name="main", kernels=["writer"], target_nodes=core_set)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="writer", runtime_arg_values={"start_page": starts, "num_pages_this_core": counts})
        ]
    )
    return spec, run_args, {"out": 0}
