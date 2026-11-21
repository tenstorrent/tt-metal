# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for upsample (regime `stick_stream`, op_design.md Blocking Model).

Work: the input's N*H*W sticks (one row-major page each, C values) are flattened into one axis and split in
contiguous ranges over the whole grid (`ttnn.split_work_to_cores`, row-wise). On a cluster the range is cut
into blocks of `block_sticks` sticks; a block is one entry of the ring `cb_input_sticks`.

Thread plan (R, C, W) = (READER_THREADS, 0, WRITER_THREADS): no compute kernel (the op computes nothing).
The ring is STRIDED on both sides, so block b of a cluster is

    read    by reader thread b mod R   (reserve, block_extent tagged page reads, barrier, push)
    written by writer thread b mod W   (wait, block_extent*scale_h*scale_w tagged page writes, barrier, pop)

Both kernels keep the ring's credits themselves (explicit sync: the reader assembles an entry from several
reads, the writer drains one entry with sh*sw writes per stick), so the ring is opted out of implicit sync on
both and every NoC transfer carries the user transaction id 0.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- Cluster model (target page .claude/references/targets/quasar.md); Python has no query for these. ---
FREE_DM_HARTS = 6  # of 8 DM cores: DM0 serves the ring interrupts, DM1 the remapper
RING_MAX_BYTES = 1_048_560  # a DFB ring is at most 1 MiB - 16 B

# --- Knobs (op_design.md Blocking Model). Each is defined once; everything below derives from them. ---
# Thread plan (R, W): two readers, four writers (C = 0). R + W <= FREE_DM_HARTS; on a STRIDED ring the
# smaller count divides the larger.
THREAD_PLAN = (2, 4)
# Ring depth: entries per thread slice (2 = double buffering per writer thread).
RING_ENTRIES_PER_THREAD = 2
# Cut count: a cluster's sticks are cut into about WRITER_THREADS * BLOCKS_PER_WRITER_THREAD blocks.
BLOCKS_PER_WRITER_THREAD = 2
# L1 kept back from the ring cap (reserve knob).
L1_RESERVE_BYTES = 0

RING_NAME = "cb_input_sticks"


def validate_thread_plan(plan):
    r, w = plan
    if not (r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and max(r, w) % min(r, w) == 0):
        raise ValueError(f"upsample thread plan (R, W) = {plan}: need R + W <= {FREE_DM_HARTS} and min(R, W) | max(R, W)")


def ring_entries(plan=THREAD_PLAN):
    # A STRIDED ring deals entries over max(R, W) slices; capacity is counted per slice.
    return RING_ENTRIES_PER_THREAD * max(plan)


def ring_bytes_cap():
    return min(RING_MAX_BYTES, ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES)


def stick_geometry(input_tensor):
    """(stick_bytes, stick_stride): one page's bytes, and its slot in a ring entry aligned for the NoC."""
    stick_bytes = input_tensor.buffer_page_size()
    align = max(ttnn.get_l1_alignment(), ttnn.get_dram_alignment())
    stick_stride = (stick_bytes + align - 1) // align * align
    return stick_bytes, stick_stride


def stick_stream_fits(input_tensor, plan=THREAD_PLAN):
    """Regime predicate of `stick_stream`: a ring of one-stick entries fits the ring cap."""
    _, stick_stride = stick_geometry(input_tensor)
    return ring_entries(plan) * stick_stride <= ring_bytes_cap()


def split_sticks(num_sticks, grid_size):
    """(core set, {core: sticks}, {core: first stick}) over the clusters that get work, row-major."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_sticks, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, start
            start += per_core
    return all_cores, counts, starts


def block_sticks_for(max_core_sticks, stick_stride, plan=THREAD_PLAN):
    """Sticks per block (= per ring entry): the coarsest cut that gives every writer thread
    BLOCKS_PER_WRITER_THREAD blocks, clamped so the whole ring fits the ring cap."""
    _, w = plan
    by_cut = math.ceil(max_core_sticks / (w * BLOCKS_PER_WRITER_THREAD))
    by_ring = ring_bytes_cap() // (ring_entries(plan) * stick_stride)
    return max(1, min(by_cut, by_ring))


def create_program_artifacts(input_tensor, output, scale_h, scale_w, *, reader_thread_mask=None, writer_thread_mask=None):
    """The ProgramSpec, run args and tensor indices of one upsample dispatch.

    `reader_thread_mask` / `writer_thread_mask` are test probes (bit t clear: thread t still takes its ring
    entries but moves no bytes), so a test can show which thread moved which block. Default: every thread.
    """
    plan = THREAD_PLAN
    validate_thread_plan(plan)
    R, W = plan
    if reader_thread_mask is None:
        reader_thread_mask = (1 << R) - 1
    if writer_thread_mask is None:
        writer_thread_mask = (1 << W) - 1

    _, _, in_w, _ = tuple(input_tensor.shape)
    num_sticks = input_tensor.buffer_num_pages()
    stick_bytes, stick_stride = stick_geometry(input_tensor)
    core_set, counts, starts = split_sticks(num_sticks, input_tensor.device().compute_with_storage_grid_size())
    block_sticks = block_sticks_for(max(counts.values()), stick_stride, plan)
    entries = ring_entries(plan)
    assert entries % max(plan) == 0 and entries * block_sticks * stick_stride <= ring_bytes_cap()

    ring = E.DataflowBufferSpec(
        unique_id=RING_NAME,
        entry_size=block_sticks * stick_stride,
        num_entries=entries,
        data_format_metadata=input_tensor.dtype,  # metadata only: the bytes never pass an unpacker (R1)
    )
    tensor_parameters = [
        E.TensorParameter(unique_id="input", spec=input_tensor.spec),
        E.TensorParameter(unique_id="output", spec=output.spec),
    ]
    common_ct = {"block_sticks": block_sticks, "stick_bytes": stick_bytes, "stick_stride": stick_stride}
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["start_stick", "core_sticks"])

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "upsample_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
        num_threads=R,
        dfb_bindings=[E.producer_of(RING_NAME, RING_NAME)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args={**common_ct, "reader_threads": R, "reader_thread_mask": int(reader_thread_mask)},
        runtime_arg_schema=dm_schema,
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "upsample_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True),
        num_threads=W,
        dfb_bindings=[E.consumer_of(RING_NAME, RING_NAME)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args={
            **common_ct,
            "writer_threads": W,
            "writer_thread_mask": int(writer_thread_mask),
            "scale_h": int(scale_h),
            "scale_w": int(scale_w),
            "in_w": int(in_w),
        },
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name="upsample",
        kernels=[reader, writer],
        dataflow_buffers=[ring],
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=core_set)],
    )
    dm_args = {"start_stick": starts, "core_sticks": counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}
