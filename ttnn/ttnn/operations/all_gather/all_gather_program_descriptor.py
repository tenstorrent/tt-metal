# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_gather — MeshProgramDescriptor assembly (bidirectional store-and-forward ring).

Builds one ``ttnn.ProgramDescriptor`` per device coordinate on the (1, N) line and
packs them into a ``ttnn.MeshProgramDescriptor``. Every device runs the SAME two-core
ring role parameterized by its ``ring_index`` (its column on the line):

  * core_fwd = (0, 0): sends toward higher ring_index (d -> d+1). Owns the barrier
    (waits + resets barrier_sem) and the own-slot LOCAL write.
  * core_bwd = (0, 1): sends toward lower ring_index (d -> d-1). Issues its backward
    barrier multicast only; correctness comes from backward_sem counting.

Each core has a reader (NCRISC) + writer (BRISC), sharing a small streaming relay CB.

Routing (``ccl_dm_route``) and the fabric-connection runtime-arg block
(``append_ccl_fabric_rt_args`` layout: ``[has_forward][fwd?][has_backward][bwd?]``)
are the Python-bound CCL host helpers; the kernel-side ``FabricStreamSender`` consumes
the block. gather_dim=0 (proven primary): slice from origin j occupies the contiguous
output page range ``[j*pages_per_shard, (j+1)*pages_per_shard)``.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# Semantic CB indices (see op_design.md "Circular Buffers"). Both cores use index 0
# (distinct core ranges), so the relay CB is per-core.
_CB_RELAY = 0

# A small, shape-independent streaming relay buffer (tiles are pushed/popped one at a
# time). Bounds L1 regardless of pages_per_shard.
_CB_NUM_PAGES = 8

_LINK_IDX = 0
_FORWARD = 0
_BACKWARD = 1


def _concat_addressing(input_tensor, output_tensor, axis, ring_size):
    """Concat-by-gather_dim output walk (op_design "Dataflow Strategy" stride table).

    Returns ``(block_in, sub_page, input_page_size, output_page_size)``.

    For a slice from origin chip ``j``, input page ``in_p`` (the sequential page index
    within a shard) maps to an output page/offset as follows:

    * ``sub_page == 0`` (whole-page remap — Regime A): every input page is one output
      page at ``out_p = (in_p // block_in)*block_in*ring_size + (in_p % block_in) + j*block_in``
      where ``block_in`` = input pages from the gather axis down (mid_in * inner). Covers
      every TILE gather on a tile-aligned axis and every ROW_MAJOR non-innermost gather.
      ``axis = 0`` degenerates to ``block_in = pages_per_shard`` (contiguous slot j).
    * ``sub_page == 1`` (sub-page byte concat — Regime B): ROW_MAJOR + INNERMOST gather
      only. The concat happens WITHIN a page (a row): ``out_p == in_p`` but the payload
      lands at byte offset ``j * input_page_size`` inside the (N x larger) output page.

    (TILE + a non-tile-aligned gather axis is refused by EXCLUSIONS — it would need a
    sub-tile repack this pure-byte-movement op cannot do.)
    """
    rank = len(input_tensor.shape)
    shape = list(input_tensor.shape)
    input_page_size = input_tensor.buffer_page_size()
    pages_per_shard = input_tensor.buffer_num_pages()

    if input_tensor.layout == ttnn.TILE_LAYOUT:
        # Page grid = tile grid: last two dims collapse to tile counts, outer dims stay.
        grid = list(shape)
        grid[-2] = (shape[-2] + 31) // 32
        grid[-1] = (shape[-1] + 31) // 32
        block_in = 1
        for i in range(axis, rank):
            block_in *= grid[i]
        sub_page = 0
        output_page_size = input_page_size  # tile page size is dtype-fixed, shape-independent
    else:  # ROW_MAJOR: page = one row of the innermost dim
        if axis < rank - 1:
            block_in = 1
            for i in range(axis, rank - 1):  # pages from the gather axis down to (not incl.) the row dim
                block_in *= shape[i]
            sub_page = 0
            output_page_size = input_page_size  # innermost (row) dim unchanged
        else:  # innermost gather: concat within the row
            block_in = pages_per_shard  # unused on the sub-page path
            sub_page = 1
            output_page_size = input_page_size * ring_size

    # Sanity: the resolved output page size must match the allocated output tensor's.
    assert output_page_size == output_tensor.buffer_page_size(), (
        f"all_gather: computed output_page_size {output_page_size} != "
        f"output tensor page size {output_tensor.buffer_page_size()} (axis={axis})"
    )
    return block_in, sub_page, input_page_size, output_page_size


def _append_fabric_rt_args(rt_args_ref, src_id, neighbor_id, program, core, is_forward):
    """Mirror ttnn::ccl::dataflow::append_ccl_fabric_rt_args.

    Appends ``[has_forward][<fwd conn args> if fwd][has_backward][<bwd conn args> if bwd]``.
    The kernel records the start index as conn_arg_idx; its leading has_forward flag is
    also the send direction the FabricStreamSender binds. setup_fabric_connection mutates
    ``program`` (appends SemaphoreDescriptors).
    """
    rt_args_ref.append(int(is_forward))  # has_forward
    if is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))
    rt_args_ref.append(int(not is_forward))  # has_backward
    if not is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))


def create_mesh_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    topology: ttnn.Topology,
    gather_axis: int,
    barrier_sem_addr: int,
    forward_sem_addr: int,
    backward_sem_addr: int,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = input_tensor.device()
    ring_size = 1
    for d in tuple(mesh_device.shape):
        ring_size *= d

    l1_alignment = ttnn.get_l1_alignment()
    data_format = input_tensor.dtype

    # Per-device shard metrics. page_size is the INPUT page (one tile / one RM row); the
    # relay CB always holds input pages. output_page_size differs only for the sub-page
    # ROW_MAJOR innermost-gather path (N x larger; the concat lives inside a row).
    page_size = input_tensor.buffer_page_size()
    pages_per_shard = input_tensor.buffer_num_pages()
    aligned_page_size = ((page_size + l1_alignment - 1) // l1_alignment) * l1_alignment

    # Concat-by-gather_dim addressing (block_in / sub_page). block_in=pages_per_shard,
    # sub_page=0 reproduces the proven gather_dim=0 contiguous walk exactly.
    block_in, sub_page, _input_page_size, output_page_size = _concat_addressing(
        input_tensor, output_tensor, gather_axis, ring_size
    )

    fwd_core = ttnn.CoreCoord(0, 0)
    bwd_core = ttnn.CoreCoord(0, 1)
    fwd_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(fwd_core, fwd_core)])
    bwd_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(bwd_core, bwd_core)])

    # Physical (virtual/NOC) coords of the two worker cores — identical on every device.
    fwd_noc = mesh_device.worker_core_from_logical_core(fwd_core)
    bwd_noc = mesh_device.worker_core_from_logical_core(bwd_core)

    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()

    def _relay_cb(core_set):
        return ttnn.CBDescriptor(
            total_size=_CB_NUM_PAGES * aligned_page_size,
            core_ranges=core_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=_CB_RELAY, data_format=data_format, page_size=aligned_page_size)
            ],
        )

    mesh_pd = ttnn.MeshProgramDescriptor()

    for d in range(ring_size):
        coord = ttnn.MeshCoordinate(0, d)
        src_id = mesh_device.get_fabric_node_id(coord)

        has_fwd = d < ring_size - 1
        has_bwd = d > 0

        # ---- forward channel (core_fwd = (0,0)) ----
        num_recv_fwd = d  # forward slices this device receives (counting threshold)
        num_consume_fwd = (d + 1) if has_fwd else 1  # own + relays, or own-only (local write)
        barrier_range_fwd = ring_size - 1 - d

        fwd_reader_ct = [
            ring_size,
            d,
            _CB_RELAY,
            _FORWARD,
            pages_per_shard,
            page_size,
            int(has_fwd),
            num_recv_fwd,
            1,  # does_own_read = 1 (always, for the local write)
            block_in,
            sub_page,
            output_page_size,
        ]
        fwd_reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
        fwd_reader_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        fwd_reader_rt = ttnn.RuntimeArgs()
        fwd_reader_rt[fwd_core.x][fwd_core.y] = [input_addr, output_addr, forward_sem_addr]
        fwd_reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "all_gather_reader.cpp"),
            core_ranges=fwd_core_set,
            compile_time_args=fwd_reader_ct,
            runtime_args=fwd_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )

        fwd_writer_ct = [
            ring_size,
            d,
            _CB_RELAY,
            _FORWARD,
            pages_per_shard,
            page_size,
            int(has_fwd),
            num_consume_fwd,
            1,  # is_fwd_core = 1
            l1_alignment,
            barrier_range_fwd,
            block_in,
            sub_page,
            output_page_size,
        ]
        fwd_writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        fwd_writer_rt = ttnn.RuntimeArgs()
        # barrier target + counting target both = fwd core (0,0) for the forward channel.
        fwd_writer_rt[fwd_core.x][fwd_core.y] = [
            output_addr,
            barrier_sem_addr,
            forward_sem_addr,
            fwd_noc.x,
            fwd_noc.y,
            fwd_noc.x,
            fwd_noc.y,
        ]
        fwd_writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "all_gather_writer.cpp"),
            core_ranges=fwd_core_set,
            compile_time_args=fwd_writer_ct,
            runtime_args=fwd_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        )

        # ---- backward channel (core_bwd = (0,1)) ----
        num_recv_bwd = ring_size - 1 - d
        num_consume_bwd = (ring_size - d) if has_bwd else 0
        barrier_range_bwd = d

        bwd_reader_ct = [
            ring_size,
            d,
            _CB_RELAY,
            _BACKWARD,
            pages_per_shard,
            page_size,
            int(has_bwd),
            num_recv_bwd,
            int(has_bwd),  # does_own_read only if it sends
            block_in,
            sub_page,
            output_page_size,
        ]
        bwd_reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
        bwd_reader_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        bwd_reader_rt = ttnn.RuntimeArgs()
        bwd_reader_rt[bwd_core.x][bwd_core.y] = [input_addr, output_addr, backward_sem_addr]
        bwd_reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "all_gather_reader.cpp"),
            core_ranges=bwd_core_set,
            compile_time_args=bwd_reader_ct,
            runtime_args=bwd_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )

        bwd_writer_ct = [
            ring_size,
            d,
            _CB_RELAY,
            _BACKWARD,
            pages_per_shard,
            page_size,
            int(has_bwd),
            num_consume_bwd,
            0,  # is_fwd_core = 0
            l1_alignment,
            barrier_range_bwd,
            block_in,
            sub_page,
            output_page_size,
        ]
        bwd_writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        bwd_writer_rt = ttnn.RuntimeArgs()
        # barrier target = fwd core (0,0); counting target = bwd core (0,1) (backward_sem owner).
        bwd_writer_rt[bwd_core.x][bwd_core.y] = [
            output_addr,
            barrier_sem_addr,
            backward_sem_addr,
            fwd_noc.x,
            fwd_noc.y,
            bwd_noc.x,
            bwd_noc.y,
        ]
        bwd_writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "all_gather_writer.cpp"),
            core_ranges=bwd_core_set,
            compile_time_args=bwd_writer_ct,
            runtime_args=bwd_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        )

        program = ttnn.ProgramDescriptor(
            kernels=[fwd_reader, fwd_writer, bwd_reader, bwd_writer],
            semaphores=[],
            cbs=[_relay_cb(fwd_core_set), _relay_cb(bwd_core_set)],
        )

        # Append the fabric-connection RT block to each writer that has a neighbour
        # (kernel index 1 = fwd writer at (0,0); index 3 = bwd writer at (0,1)).
        if has_fwd:
            fwd_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord, ttnn.MeshCoordinate(0, d + 1), topology)
            _append_fabric_rt_args(
                program.kernels[1].runtime_args[fwd_core.x][fwd_core.y],
                src_id,
                fwd_route.neighbor_id,
                program,
                fwd_core,
                fwd_route.is_forward,
            )
        if has_bwd:
            bwd_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord, ttnn.MeshCoordinate(0, d - 1), topology)
            _append_fabric_rt_args(
                program.kernels[3].runtime_args[bwd_core.x][bwd_core.y],
                src_id,
                bwd_route.neighbor_id,
                program,
                bwd_core,
                bwd_route.is_forward,
            )

        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program

    return mesh_pd
