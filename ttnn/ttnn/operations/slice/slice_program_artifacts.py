# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for slice (regime `page_relay`, op_design.md).

One DM->DM ring per cluster (`slice_pages`), no compute kernel:

    reader hart p   reads    the source pages of output pages p, p+R, ... of its range into the ring
    writer hart w   writes   output pages w, w+W, ... of its range out of the ring

Every output page is exactly one input page, or one byte range of it, so the op's block space folds to one
axis, `out_page`, in row-major output page order. Two regimes (op_design.md, Regimes), selected by `regime_for`:

    page_relay      every output page is one whole input page (TILE from a tile boundary; RM with W whole)
    rm_column_cut   RM with W narrowed: output row = bytes [w_start*e, w_end*e) of its input row. The read
                    starts at that byte offset and moves only the row's bytes (the Quasar NoC needs no
                    alignment, noc_parameters.h NOC_*_ALIGNMENT_BYTES = 1), so a transfer is not one ring
                    entry and the ring takes the explicit protocol The host splits contiguous ranges of it
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
# One transfer is one NoC burst at most (tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:
# 256 words x 256 B).
NOC_MAX_BURST_BYTES = 65_536
# The overlay cuts every transfer into packets of at most 8 KiB (NOC_OVERLAY_MAX_BYTES_IN_PACKET,
# tt_metal/hw/inc/internal/tt-2xx/quasar/noc_cmd_buf_common.h). Larger pages take the ring's explicit protocol.
IMPLICIT_SYNC_MAX_ENTRY_BYTES = 8 * 1024

PAGE_RELAY, RM_COLUMN_CUT = "page_relay", "rm_column_cut"
RING = "slice_pages"  # the DFB's unique id and its kernel-side accessor name (`dfb::slice_pages`)
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
    # W > R is not needed by the design (plan (1, 1); (2, 2) tests the strided dealing) and stays refused
    # until a test covers it.
    if w > r:
        raise ValueError(f"slice: thread plan (R, W) = {threads}: W > R is not supported")


def regime_for(input_tensor, start, out_shape):
    """The design's regime predicate: RM with W narrowed is `rm_column_cut`, everything else `page_relay`."""
    w = int(input_tensor.shape[-1])
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT and (start[-1], out_shape[-1]) != (0, w):
        return RM_COLUMN_CUT
    return PAGE_RELAY


def uses_explicit_sync(entry_bytes, regime=PAGE_RELAY):
    # An implicit-sync transfer moves exactly one entry, from the entry's start: a column cut moves a part of
    # a page from an offset, and an entry is at most one NoC packet (IMPLICIT_SYNC_MAX_ENTRY_BYTES).
    return regime == RM_COLUMN_CUT or entry_bytes > IMPLICIT_SYNC_MAX_ENTRY_BYTES


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
        # A row is one page in both regimes; rm_column_cut takes a byte range of it (`row_transfer`).
        in_inner, out_inner, start_inner = [h, 1], [h_out, 1], [h_start, 0]
    in_page_shape = [1] * pad + shape[:-2] + in_inner
    out_page_shape = [1] * pad + list(out_shape[:-2]) + out_inner
    page_start = [0] * pad + list(start[:-2]) + start_inner
    in_page_stride = [math.prod(in_page_shape[d + 1 :]) for d in range(PAGE_MAP_RANK)]
    src_base = sum(s * st for s, st in zip(page_start, in_page_stride))
    return out_page_shape, in_page_stride, src_base


def row_transfer(input_tensor, start, out_shape, regime, entry_bytes):
    """(source byte offset into the page, bytes per transfer). page_relay moves the whole aligned page slot
    from its start; rm_column_cut moves the row's W_out elements from element w_start."""
    if regime != RM_COLUMN_CUT:
        return 0, entry_bytes
    elem_bytes = input_tensor.buffer_page_size() // int(input_tensor.shape[-1])  # an RM page is one row of W
    return start[-1] * elem_bytes, out_shape[-1] * elem_bytes


def _dm_config(make, explicit_sync):
    # Implicit sync is the ring's default on a data-movement kernel; the explicit protocol opts it out.
    hw = make()
    if explicit_sync:
        hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=[RING])
    return hw


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
    regime = regime_for(input_tensor, start, out_shape)
    # A ring entry is one output page slot. page_relay moves the whole aligned slot, so the input's and the
    # output's slots must agree; rm_column_cut's input rows are wider than its entries.
    entry_bytes = output.buffer_aligned_page_size()
    if regime == PAGE_RELAY:
        assert entry_bytes == input_tensor.buffer_aligned_page_size(), (
            entry_bytes,
            input_tensor.buffer_aligned_page_size(),
        )
    src_offset_bytes, transfer_bytes = row_transfer(input_tensor, start, out_shape, regime, entry_bytes)

    core_set, counts, firsts, max_core_pages = split_pages(
        num_out_pages, input_tensor.device().compute_with_storage_grid_size()
    )
    block_pages = block_pages_for(max_core_pages, entry_bytes, threads)
    explicit_sync = uses_explicit_sync(entry_bytes, regime)
    if explicit_sync and R != W:
        # Explicit sync batches a block per hart in place, which needs each hart on one tile counter per end.
        raise ValueError(
            f"slice: thread plan (R, W) = {threads}: the explicit-sync ring (pages over 8 KiB, or {RM_COLUMN_CUT})"
            " needs R == W"
        )

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
    # Explicit sync keeps the ring's credits in the kernels, which then reserve / wait a block per hart.
    # transfer_bytes is the size of every explicit-sync transfer on both ends (the reader's also starts
    # src_offset_bytes into its page); implicit-sync transfers are one whole entry by construction.
    common = {"thread_block_pages": block_pages // max(threads), "transfer_bytes": transfer_bytes}

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "slice_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, explicit_sync),
        num_threads=R,
        dfb_bindings=[E.producer_of(RING, RING)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args={"reader_threads": R, "explicit_sync": int(explicit_sync)},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=per_core,
            common_runtime_arg_names=["src_base", "src_offset_bytes", *common],
            common_runtime_array_args=[
                E.ArrayArgDecl(name="out_page_shape", size=PAGE_MAP_RANK),
                E.ArrayArgDecl(name="in_page_stride", size=PAGE_MAP_RANK),
            ],
        ),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "slice_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, explicit_sync),
        num_threads=W,
        dfb_bindings=[E.consumer_of(RING, RING)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args={"writer_threads": W, "explicit_sync": int(explicit_sync)},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=per_core, common_runtime_arg_names=list(common)
        ),
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
                common_runtime_arg_values={"src_base": src_base, "src_offset_bytes": src_offset_bytes, **common},
                common_runtime_array_arg_values={"out_page_shape": out_page_shape, "in_page_stride": in_page_stride},
            ),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args, common_runtime_arg_values=common),
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}
