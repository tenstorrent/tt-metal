# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for slice (regime `page_relay`, op_design.md).

One DM->DM ring per cluster (`slice_pages`), no compute kernel:

    reader hart p   reads    the source pages of output pages p, p+R, ... of its range into the ring
    writer hart w   writes   output pages w, w+W, ... of its range out of the ring

Every output page is exactly one whole input page (within the built regime's domain), so the op's block
space folds to one axis, `out_page`, in row-major output page order. The host splits contiguous ranges of it
over the whole grid; the reader maps each output page to its source page through a 4-D page map (rank
left-padded with 1s), advanced as an odometer. Both kernels use implicit sync: one tagged NoC call per page,
the DM0 interrupt handler posts / acks one block (half the ring) per batch.

Knobs (each defined once here; everything else derives from them):
    DM_THREAD_PLAN      (R, W): reader / writer harts per cluster
    RING_DEPTH_BLOCKS   blocks resident in the ring (2: the reader fills one while the writer drains one)
    block_pages         pages per block, computed per call from the split's max_core_pages (`block_pages_for`)
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- Knobs -------------------------------------------------------------------------------------------------
DM_THREAD_PLAN = (1, 1)  # (R readers, W writers). Bound by data movement: one hart of each kind (target page).
RING_DEPTH_BLOCKS = 2  # blocks in the ring: double buffering, the only overlap a relay has
L1_RESERVE_BYTES = 0  # L1 kept back from the ring cap (the ring is the op's only L1 allocation)

# --- Cluster model and mechanism caps (target page; Python has no query for them) --------------------------
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 the remapper
DFB_RING_MAX_BYTES = 1_048_560  # a ring is at most 1 MiB - 16 B (target page, cluster model)
# One implicit-sync entry is one NoC transaction; keep it within one burst so the ISR's transaction count
# is the entry count (tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h: 256 words x 256 B).
NOC_MAX_BURST_BYTES = 65_536

RING = "slice_pages"
PAGE_MAP_RANK = 4


def ring_cap_bytes():
    return min(DFB_RING_MAX_BYTES, ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES)


def block_pages_cap(entry_bytes):
    return ring_cap_bytes() // (RING_DEPTH_BLOCKS * entry_bytes)


def check_mechanism_caps(input_tensor):
    """Refusals that come from the mechanism, not from an axis (design: Mechanism caps)."""
    entry_bytes = input_tensor.buffer_aligned_page_size()
    if entry_bytes > NOC_MAX_BURST_BYTES:
        raise NotImplementedError(
            f"slice: a page of {entry_bytes} B exceeds one NoC burst ({NOC_MAX_BURST_BYTES} B)"
        )
    if block_pages_cap(entry_bytes) < 1:
        raise NotImplementedError(f"slice: {RING_DEPTH_BLOCKS} pages of {entry_bytes} B do not fit one ring")


def validate_thread_plan(threads):
    r, w = threads
    if not (r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and (r % w == 0 or w % r == 0)):
        raise ValueError(f"slice: thread plan (R, W) = {threads}: need R + W <= {FREE_DM_HARTS}, R | W or W | R")


def block_pages_for(max_core_pages, entry_bytes, threads):
    """Pages per block: half the busiest core's share (RING_DEPTH_BLOCKS blocks cover it), clamped by the
    ring cap, and a multiple of max(R, W) so the implicit-sync batches divide over the harts."""
    step = max(threads)
    cap = block_pages_cap(entry_bytes)
    block = max(1, min(math.ceil(max_core_pages / RING_DEPTH_BLOCKS), cap))
    block = math.ceil(block / step) * step
    if block > cap:
        block = (cap // step) * step
    if block < 1:
        raise NotImplementedError(f"slice: no block of {step} pages of {entry_bytes} B fits one ring")
    return block


def page_map(input_tensor, start, out_shape):
    """(out_page_shape, in_page_stride, src_base): the 4-D page map of the built regime.

    Output page with multi-index m (row-major over out_page_shape) is input page
    src_base + sum_d m[d] * in_page_stride[d]."""
    shape = [int(d) for d in input_tensor.shape]
    rank = len(shape)
    pad = PAGE_MAP_RANK - rank
    h, w = shape[-2], shape[-1]
    h_start, w_start = start[-2], start[-1]
    h_out, w_out = out_shape[-2], out_shape[-1]
    if input_tensor.layout == ttnn.TILE_LAYOUT:
        th, tw = input_tensor.tile.tile_shape
        if h_start % th or w_start % tw:
            raise NotImplementedError("slice: a tile-layout block must start on a tile boundary (regime tile_retile)")
        in_inner = [math.ceil(h / th), math.ceil(w / tw)]
        out_inner = [math.ceil(h_out / th), math.ceil(w_out / tw)]
        start_inner = [h_start // th, w_start // tw]
    else:
        if w_start != 0 or w_out != w:
            raise NotImplementedError("slice: a row-major block must take whole rows (regime rm_column_cut)")
        in_inner, out_inner, start_inner = [h, 1], [h_out, 1], [h_start, 0]
    in_page_shape = [1] * pad + shape[:-2] + in_inner
    out_page_shape = [1] * pad + list(out_shape[:-2]) + out_inner
    page_start = [0] * pad + list(start[:-2]) + start_inner
    in_page_stride = [math.prod(in_page_shape[d + 1 :]) for d in range(PAGE_MAP_RANK)]
    src_base = sum(s * st for s, st in zip(page_start, in_page_stride))
    return out_page_shape, in_page_stride, src_base


def split_pages(num_pages, grid_size):
    """(core set, {core: pages}, {core: first page}, max pages per core) over the clusters that get work."""
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_pages, row_wise=True
    )
    counts, starts, first = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = per_core, first
            first += per_core
    return all_cores, counts, starts, max(per_core_1, per_core_2)


def create_program_artifacts(input_tensor, output, start, *, threads=DM_THREAD_PLAN):
    validate_thread_plan(threads)
    R, W = threads
    out_shape = [int(d) for d in output.shape]
    out_page_shape, in_page_stride, src_base = page_map(input_tensor, list(start), out_shape)

    num_out_pages = math.prod(out_page_shape)
    assert num_out_pages == output.buffer_num_pages(), (num_out_pages, output.buffer_num_pages())
    entry_bytes = input_tensor.buffer_aligned_page_size()
    # The implicit-sync transfer moves the whole aligned page slot; the two slots must agree.
    assert entry_bytes == output.buffer_aligned_page_size(), (entry_bytes, output.buffer_aligned_page_size())

    core_set, counts, firsts, max_core_pages = split_pages(
        num_out_pages, input_tensor.device().compute_with_storage_grid_size()
    )
    block_pages = block_pages_for(max_core_pages, entry_bytes, threads)

    ring = E.DataflowBufferSpec(
        unique_id=RING,
        entry_size=entry_bytes,
        num_entries=RING_DEPTH_BLOCKS * block_pages,
        data_format_metadata=input_tensor.dtype,  # metadata only: bytes are never unpacked
    )
    tensor_parameters = [
        E.TensorParameter(unique_id="input", spec=input_tensor.spec),
        E.TensorParameter(unique_id="output", spec=output.spec),
    ]
    per_core = ["first_out_page", "num_pages"]

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "slice_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        dfb_bindings=[E.producer_of(RING, "slice_pages")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args={"reader_threads": R},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=per_core,
            common_runtime_arg_names=["src_base"],
            common_runtime_array_args=[
                E.ArrayArgDecl(name="out_page_shape", size=PAGE_MAP_RANK),
                E.ArrayArgDecl(name="in_page_stride", size=PAGE_MAP_RANK),
            ],
        ),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "slice_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of(RING, "slice_pages")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args={"writer_threads": W},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=per_core),
    )
    spec = E.ProgramSpec(
        name="slice",
        kernels=[reader, writer],
        dataflow_buffers=[ring],
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "writer"], target_nodes=core_set)],
    )
    dm_args = {"first_out_page": firsts, "num_pages": counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(
                kernel="reader",
                runtime_arg_values=dm_args,
                common_runtime_arg_values={"src_base": src_base},
                common_runtime_array_arg_values={"out_page_shape": out_page_shape, "in_page_stride": in_page_stride},
            ),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}
