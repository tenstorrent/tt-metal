# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""reduce_scatter — MeshProgramDescriptor assembly (two phases).

Two ordered ``ttnn.generic_op`` dispatches share the device command queue:

  * Phase A (gather, fabric): line store-and-forward gather of every device's FULL
    shard into an op-internal ``gather_buffer`` (block c at pages
    ``[c*P_shard, (c+1)*P_shard)``) — identical structure to the proven all_reduce
    Phase A. Per device, two worker cores run a per-direction role:
        forward core  (0, 0): flow rightward, fabric connection -> chip i+1
        backward core (0, 1): flow leftward,  fabric connection -> chip i-1
    Each core runs a reader (NCRISC) + writer (BRISC). Fabric egress goes through
    the CCL helper (writer); the receive ingress / relay read-back / counting wait /
    cache-reuse re-arm are op-owned NoC calls (reader). Cross-device sync uses ONE
    op-internal GlobalSemaphore (parked on this descriptor by the entry point).

  * Phase B (scatter-reduce, local compute): THE SCATTER IS THE ADDRESSING. P_out
    output-tile positions are split across the compute grid; each core's reader
    walks slice ``my_chip_id``'s source pages (shared-schedule SliceRowWalker),
    pulls the N gathered tiles per position, the compute kernel sums them
    (``sum_blocks``), and the writer lands the reduced tile at its dense output
    page. No fabric, no cross-device sync.

Five single-purpose kernel sources (deliberate divergence from all_reduce's
shared-source-with-phase-CT-arg pattern — removes the uniform-CT-superset
zero-padding footgun). CT-arg layout per kernel: all scalar CT args first,
TensorAccessorArgs last.

Program-cache discipline: everything per-call (buffer addresses, page counts,
sem_addr, num_hops, start_tile/n, slice_Wt/Wt) is a RUNTIME arg; CT args are
structural (CB indices, direction, my_chip_id, N, dim, l1_alignment, accessors).
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic names; see op_design.md "Circular Buffers").
_CB_RELAY_PAGES = 16  # Phase A reader -> writer: seed pages + store-and-forward read-backs
_CB_SELF_COPY_SCRATCH = 24  # Phase A forward reader scratch for the local self-copy (reserve-only)
_CB_GATHERED_SLICES = 0  # Phase B reader -> compute: N gathered tiles for one output position
_CB_REDUCED_SLICE = 16  # Phase B compute -> writer: the summed tile

_LINK_IDX = 0  # single-link transfer

_TILE = 32  # tile edge (elements)


def _round_up(value: int, mult: int) -> int:
    return ((value + mult - 1) // mult) * mult


def _num_line_devices(mesh_device) -> int:
    """Number of devices on the 1-D line (mesh view is (1, N))."""
    n = 1
    for d in tuple(mesh_device.shape):
        n *= d
    return n


# ===========================================================================
# Phase A — line store-and-forward gather (fabric)
# ===========================================================================


def _append_fabric_rt_args(rt_args_ref, src_id, neighbor_id, program, core, is_forward):
    """Mirror ttnn::ccl::dataflow::append_ccl_fabric_rt_args.

    After the call the block beginning at the current rt_args length is:
        [has_forward][<forward conn args> if fwd][has_backward][<backward conn args> if bwd]
    The kernel records that start index as conn_arg_idx; its leading has_forward
    flag also equals the send direction, which the kernel peeks for is_forward.
    ``setup_fabric_connection`` also mutates ``program`` (appends SemaphoreDescriptors).
    """
    rt_args_ref.append(int(is_forward))  # has_forward
    if is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))
    rt_args_ref.append(int(not is_forward))  # has_backward
    if not is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))


def _build_gather_device_program(
    mesh_device,
    i,
    num_devices,
    input_tensor,
    gather_buffer,
    sem_addr,
    topology,
    fwd_noc,
    bwd_noc,
    page_size,
    aligned_page_size,
    pages_per_shard,
    l1_alignment,
    data_format,
    input_ta,
    gather_buffer_ta,
):
    """Build the Phase-A ProgramDescriptor for device ``i`` on the line."""
    num_targets_fwd = num_devices - 1 - i  # devices reachable rightward (i+1..N-1)
    num_targets_bwd = i  # devices reachable leftward (0..i-1)

    coord_i = ttnn.MeshCoordinate(0, i)
    fabric_id_i = mesh_device.get_fabric_node_id(coord_i)

    forward_core = ttnn.CoreCoord(0, 0)
    backward_core = ttnn.CoreCoord(0, 1)
    fwd_set = ttnn.CoreRangeSet([ttnn.CoreRange(forward_core, forward_core)])
    bwd_set = ttnn.CoreRangeSet([ttnn.CoreRange(backward_core, backward_core)])
    both_set = ttnn.CoreRangeSet([ttnn.CoreRange(forward_core, backward_core)])

    # ----- circular buffers (on both worker cores) -----
    cb_relay_pages = ttnn.CBDescriptor(
        total_size=2 * aligned_page_size,  # double-buffered streaming page
        core_ranges=both_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_RELAY_PAGES, data_format=data_format, page_size=aligned_page_size)
        ],
    )
    cb_self_copy_scratch = ttnn.CBDescriptor(
        total_size=2 * aligned_page_size,  # forward reader's self-copy scratch (reserve-only)
        core_ranges=both_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_CB_SELF_COPY_SCRATCH, data_format=data_format, page_size=aligned_page_size
            )
        ],
    )

    # ----- readers (NCRISC): scalar CT args first, accessors last -----
    def reader_ct(direction):
        return (
            [
                _CB_RELAY_PAGES,
                _CB_SELF_COPY_SCRATCH,
                direction,
                i,  # my_chip_id
                num_targets_fwd,
                num_targets_bwd,
            ]
            + input_ta
            + gather_buffer_ta
        )

    reader_rt_vals = [
        input_tensor.buffer_address(),
        gather_buffer.buffer_address(),
        pages_per_shard,
        page_size,
        sem_addr,
    ]

    fwd_reader_rt = ttnn.RuntimeArgs()
    fwd_reader_rt[forward_core.x][forward_core.y] = list(reader_rt_vals)
    bwd_reader_rt = ttnn.RuntimeArgs()
    bwd_reader_rt[backward_core.x][backward_core.y] = list(reader_rt_vals)

    fwd_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_gather_reader.cpp"),
        core_ranges=fwd_set,
        compile_time_args=reader_ct(0),
        runtime_args=fwd_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    bwd_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_gather_reader.cpp"),
        core_ranges=bwd_set,
        compile_time_args=reader_ct(1),
        runtime_args=bwd_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- writers (BRISC): scalar CT args first, accessor last -----
    def writer_ct(direction):
        return [
            _CB_RELAY_PAGES,
            direction,
            i,  # my_chip_id
            num_targets_fwd,
            num_targets_bwd,
            l1_alignment,
        ] + gather_buffer_ta

    fwd_writer_rt = ttnn.RuntimeArgs()
    bwd_writer_rt = ttnn.RuntimeArgs()
    fwd_route = None
    bwd_route = None
    if num_targets_fwd > 0:
        coord_next = ttnn.MeshCoordinate(0, i + 1)
        fwd_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_next, topology)
        fwd_writer_rt[forward_core.x][forward_core.y] = [
            gather_buffer.buffer_address(),
            pages_per_shard,
            page_size,
            fwd_route.num_hops,
            sem_addr,
            fwd_noc.x,  # neighbour forward core noc x (counting-sem target)
            fwd_noc.y,
        ]
    else:
        fwd_writer_rt[forward_core.x][forward_core.y] = []  # early-return writer reads nothing

    if num_targets_bwd > 0:
        coord_prev = ttnn.MeshCoordinate(0, i - 1)
        bwd_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_prev, topology)
        bwd_writer_rt[backward_core.x][backward_core.y] = [
            gather_buffer.buffer_address(),
            pages_per_shard,
            page_size,
            bwd_route.num_hops,
            sem_addr,
            bwd_noc.x,  # neighbour backward core noc x (counting-sem target)
            bwd_noc.y,
        ]
    else:
        bwd_writer_rt[backward_core.x][backward_core.y] = []

    fwd_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_gather_writer.cpp"),
        core_ranges=fwd_set,
        compile_time_args=writer_ct(0),
        runtime_args=fwd_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    bwd_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_gather_writer.cpp"),
        core_ranges=bwd_set,
        compile_time_args=writer_ct(1),
        runtime_args=bwd_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[fwd_reader, fwd_writer, bwd_reader, bwd_writer],
        semaphores=[],
        cbs=[cb_relay_pages, cb_self_copy_scratch],
    )

    # Fabric connection args live on the writers (kernel idx 1 = fwd, idx 3 = bwd);
    # appended AFTER ProgramDescriptor construction by mutating runtime_args in place.
    if num_targets_fwd > 0:
        fabric_id_next = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, i + 1))
        ref = program.kernels[1].runtime_args[forward_core.x][forward_core.y]
        _append_fabric_rt_args(ref, fabric_id_i, fabric_id_next, program, forward_core, fwd_route.is_forward)
    if num_targets_bwd > 0:
        fabric_id_prev = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(0, i - 1))
        ref = program.kernels[3].runtime_args[backward_core.x][backward_core.y]
        _append_fabric_rt_args(ref, fabric_id_i, fabric_id_prev, program, backward_core, bwd_route.is_forward)

    return program


def create_gather_mesh_program_descriptor(
    input_tensor: ttnn.Tensor,
    gather_buffer: ttnn.Tensor,
    topology: ttnn.Topology,
    sem_addr: int,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = input_tensor.device()
    num_devices = _num_line_devices(mesh_device)

    l1_alignment = ttnn.get_l1_alignment()
    data_format = input_tensor.dtype
    page_size = input_tensor.buffer_page_size()
    pages_per_shard = input_tensor.buffer_num_pages()
    aligned_page_size = _round_up(page_size, l1_alignment)

    # NoC coords of the two worker cores (uniform across the mesh) — the
    # counting-sem targets the SAME logical core on the neighbour device.
    forward_core = ttnn.CoreCoord(0, 0)
    backward_core = ttnn.CoreCoord(0, 1)
    fwd_noc = mesh_device.worker_core_from_logical_core(forward_core)
    bwd_noc = mesh_device.worker_core_from_logical_core(backward_core)

    input_ta = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    gather_buffer_ta = list(ttnn.TensorAccessorArgs(gather_buffer).get_compile_time_args())

    mesh_pd = ttnn.MeshProgramDescriptor()
    for i in range(num_devices):
        coord_i = ttnn.MeshCoordinate(0, i)
        program = _build_gather_device_program(
            mesh_device,
            i,
            num_devices,
            input_tensor,
            gather_buffer,
            sem_addr,
            topology,
            fwd_noc,
            bwd_noc,
            page_size,
            aligned_page_size,
            pages_per_shard,
            l1_alignment,
            data_format,
            input_ta,
            gather_buffer_ta,
        )
        mesh_pd[ttnn.MeshCoordinateRange(coord_i, coord_i)] = program

    return mesh_pd


# ===========================================================================
# Phase B — scatter-reduce (local compute; the scatter IS the addressing)
# ===========================================================================


def _build_reduce_device_program(
    my_chip_id,
    num_devices,
    scatter_dim,
    gather_buffer,
    output_tensor,
    grid,
    tile_size,
    pages_per_shard,
    pages_per_output,
    slice_Wt,
    tensor_Wt,
    gather_buffer_ta,
    output_ta,
):
    """Build the Phase-B ProgramDescriptor for device ``my_chip_id``.

    Per-device programs differ ONLY in the reader's my_chip_id CT arg (the slice
    base) — the scatter is pure local addressing.
    """
    # P_out output-tile positions split across the compute grid.
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        tiles_per_core_g1,
        tiles_per_core_g2,
    ) = ttnn.split_work_to_cores(grid, pages_per_output)

    # ----- circular buffers (on the work cores) -----
    # cb_gathered_slices capacity 2*N tiles: sum_blocks waits its whole N-tile
    # input up front, so capacity must be >= N; 2x double-buffers the reader.
    cb_gathered_slices = ttnn.CBDescriptor(
        total_size=2 * num_devices * tile_size,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_CB_GATHERED_SLICES, data_format=gather_buffer.dtype, page_size=tile_size
            )
        ],
    )
    cb_reduced_slice = ttnn.CBDescriptor(
        total_size=2 * tile_size,  # double buffer
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_CB_REDUCED_SLICE, data_format=output_tensor.dtype, page_size=tile_size
            )
        ],
    )

    # ----- per-core runtime args (each core owns [start_tile, start_tile + n)) -----
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    cores = ttnn.corerange_to_cores(all_cores, num_cores, True)
    g1_count = core_group_1.num_cores()
    gb_addr = gather_buffer.buffer_address()
    out_addr = output_tensor.buffer_address()
    start = 0
    for idx, core in enumerate(cores):
        n = tiles_per_core_g1 if idx < g1_count else tiles_per_core_g2
        # reader: [gather_buffer_addr, page_size, P_shard, start_tile, num_tiles, slice_Wt, tensor_Wt]
        reader_rt[core.x][core.y] = [gb_addr, tile_size, pages_per_shard, start, n, slice_Wt, tensor_Wt]
        # writer: [output_addr, page_size, start_tile, num_tiles]
        writer_rt[core.x][core.y] = [out_addr, tile_size, start, n]
        # compute: [num_tiles]
        compute_rt[core.x][core.y] = [n]
        start += n

    # ----- kernels -----
    reader_ct = [
        _CB_GATHERED_SLICES,
        num_devices,  # N
        my_chip_id,  # slice index (the scatter)
        scatter_dim,
    ] + gather_buffer_ta

    writer_ct = [_CB_REDUCED_SLICE] + output_ta

    compute_ct = [_CB_GATHERED_SLICES, _CB_REDUCED_SLICE, num_devices]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_reduce_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_reduce_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reduce_scatter_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        # fp32 DST accumulation covers both the bf16 sum-of-N rounding budget and
        # the float32 dtype; sum_blocks' internal DEST_AUTO_LIMIT chunking derives
        # the matching 4-tile f32 capacity kernel-side.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],
        semaphores=[],
        cbs=[cb_gathered_slices, cb_reduced_slice],
    )


def create_reduce_mesh_program_descriptor(
    gather_buffer: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    num_devices: int,
    scatter_dim: int,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = gather_buffer.device()

    grid = mesh_device.compute_with_storage_grid_size()
    tile_size = output_tensor.buffer_page_size()
    pages_per_output = output_tensor.buffer_num_pages()  # P_out output-tile positions
    pages_per_shard = gather_buffer.buffer_num_pages() // num_devices  # P_shard (one full block)

    # Slice geometry (dim 3): tiles per full-shard row and per output-slice row.
    tensor_Wt = output_tensor.shape[3] * num_devices // _TILE  # = input W / 32
    slice_Wt = output_tensor.shape[3] // _TILE

    gather_buffer_ta = list(ttnn.TensorAccessorArgs(gather_buffer).get_compile_time_args())
    output_ta = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    mesh_pd = ttnn.MeshProgramDescriptor()
    for i in range(num_devices):
        coord_i = ttnn.MeshCoordinate(0, i)
        program = _build_reduce_device_program(
            i,
            num_devices,
            scatter_dim,
            gather_buffer,
            output_tensor,
            grid,
            tile_size,
            pages_per_shard,
            pages_per_output,
            slice_Wt,
            tensor_Wt,
            gather_buffer_ta,
            output_ta,
        )
        mesh_pd[ttnn.MeshCoordinateRange(coord_i, coord_i)] = program
    return mesh_pd
