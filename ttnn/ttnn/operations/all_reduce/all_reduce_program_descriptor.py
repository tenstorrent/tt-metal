# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""all_reduce — MeshProgramDescriptor assembly (two phases).

Two ordered ``ttnn.generic_op`` dispatches share the device command queue:

  * Phase A (gather): line store-and-forward gather into an op-internal
    ``gather_buffer`` — structurally the all_gather ``gather_dim=0`` pattern. Per
    device, two worker cores run a bidirectional ring role:
        forward core  (0, 0): flow rightward, fabric connection -> chip i+1
        backward core (0, 1): flow leftward,  fabric connection -> chip i-1
    Each core runs a reader (NCRISC) + writer (BRISC). Fabric egress goes through
    the CCL helper (writer); the receive ingress / relay read-back / counting wait
    are op-owned NoC calls (reader). Cross-device sync uses ONE op-internal
    GlobalSemaphore (parked on this descriptor).

  * Phase B (reduce): local element-wise N-way tile sum. ``P`` output-tile
    positions are split across the compute grid; each core reads the N shard
    tiles at a position (one from each gather block), sums them on TRISC, and
    writes the reduced tile back to ``output``. No fabric, no cross-device sync.

ONE reader.cpp / writer.cpp source serves BOTH phases, selected by a leading
``phase`` compile-time arg. To keep the discarded ``if constexpr`` branch's
compile-time-arg reads in-bounds (get_compile_time_arg_val static-asserts on the
index, even in a dead branch), both phases use a UNIFORM compile-time-arg
superset: 7 scalar CT args after ``phase``, then a fixed number of
TensorAccessorArgs (reader: 2, writer: 1). Unused scalar slots are zero-padded;
the reduce reader passes a second (unused-by-it) accessor purely to keep the dead
gather branch's second-accessor offset valid.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

# Topology lives on the C++ module; reference it directly to stay safe at
# eager-import time (the top-level ttnn.Topology alias binds later).
from ttnn._ttnn.operations.ccl import Topology as _Topology

KERNEL_DIR = Path(__file__).parent / "kernels"

# Phase selectors (leading CT arg of the shared reader / writer sources).
_PHASE_GATHER = 0
_PHASE_REDUCE = 1

# CB indices (semantic; see op_design.md "Circular Buffers").
_CB_RELAY_PAGES = 16  # Phase A reader -> writer: seed pages + store-and-forward read-backs
_CB_SELF_COPY = 24  # Phase A forward reader scratch for the local self-copy
_CB_GATHERED_SHARDS = 0  # Phase B reader -> compute: N shard tiles for one position
_CB_REDUCED = 16  # Phase B compute -> writer: the summed tile

_LINK_IDX = 0  # single-link transfer

# Number of scalar CT args after `phase`, uniform across both phases.
_NUM_SCALAR_CT = 7


def _round_up(value: int, mult: int) -> int:
    return ((value + mult - 1) // mult) * mult


# ===========================================================================
# Phase A — line store-and-forward gather
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
    cb_relay = ttnn.CBDescriptor(
        total_size=2 * aligned_page_size,  # double-buffered streaming chunk
        core_ranges=both_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_RELAY_PAGES, data_format=data_format, page_size=aligned_page_size)
        ],
    )
    cb_self = ttnn.CBDescriptor(
        total_size=2 * aligned_page_size,  # forward reader's self-copy scratch
        core_ranges=both_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_SELF_COPY, data_format=data_format, page_size=aligned_page_size)
        ],
    )

    # ----- readers (NCRISC) -----
    # Uniform CT superset: [phase][7 scalars][input accessor][gather_buffer accessor].
    def reader_ct(direction):
        return (
            [
                _PHASE_GATHER,
                _CB_RELAY_PAGES,  # scalar 1
                _CB_SELF_COPY,  # scalar 2
                direction,  # scalar 3
                i,  # scalar 4 (my_chip_id)
                num_devices,  # scalar 5 (ring_size)
                num_targets_fwd,  # scalar 6
                num_targets_bwd,  # scalar 7
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
        kernel_source=str(KERNEL_DIR / "all_reduce_reader.cpp"),
        core_ranges=fwd_set,
        compile_time_args=reader_ct(0),
        runtime_args=fwd_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    bwd_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_reduce_reader.cpp"),
        core_ranges=bwd_set,
        compile_time_args=reader_ct(1),
        runtime_args=bwd_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- writers (BRISC) -----
    # Uniform CT superset: [phase][7 scalars][gather_buffer accessor].
    def writer_ct(direction):
        return [
            _PHASE_GATHER,
            _CB_RELAY_PAGES,  # scalar 1
            direction,  # scalar 2
            i,  # scalar 3 (my_chip_id)
            num_devices,  # scalar 4 (ring_size)
            num_targets_fwd,  # scalar 5
            num_targets_bwd,  # scalar 6
            l1_alignment,  # scalar 7
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
        kernel_source=str(KERNEL_DIR / "all_reduce_writer.cpp"),
        core_ranges=fwd_set,
        compile_time_args=writer_ct(0),
        runtime_args=fwd_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    bwd_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_reduce_writer.cpp"),
        core_ranges=bwd_set,
        compile_time_args=writer_ct(1),
        runtime_args=bwd_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[fwd_reader, fwd_writer, bwd_reader, bwd_writer],
        semaphores=[],
        cbs=[cb_relay, cb_self],
    )

    # Fabric connection args live on the writers (kernel idx 1 = fwd, idx 3 = bwd).
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
# Phase B — local element-wise N-way tile sum
# ===========================================================================


def _build_reduce_device_program(
    gather_buffer,
    output_tensor,
    num_devices,
    grid,
    tile_size,
    pages_per_shard,
    gather_buffer_ta,
    output_ta,
):
    """Build the Phase-B ProgramDescriptor (identical on every device — pure local
    compute)."""
    # P output-tile positions split across the compute grid.
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        tiles_per_core_g1,
        tiles_per_core_g2,
    ) = ttnn.split_work_to_cores(grid, pages_per_shard)

    # ----- circular buffers (on the work cores) -----
    cb_gathered = ttnn.CBDescriptor(
        total_size=2 * num_devices * tile_size,  # double-buffered block of N shard tiles
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_CB_GATHERED_SHARDS, data_format=gather_buffer.dtype, page_size=tile_size
            )
        ],
    )
    cb_reduced = ttnn.CBDescriptor(
        total_size=2 * tile_size,  # double buffer
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_REDUCED, data_format=output_tensor.dtype, page_size=tile_size)
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
        # reader: [gather_buffer_addr, page_size, start_tile, num_tiles]
        reader_rt[core.x][core.y] = [gb_addr, tile_size, start, n]
        # writer: [output_addr, page_size, start_tile, num_tiles]
        writer_rt[core.x][core.y] = [out_addr, tile_size, start, n]
        # compute: [num_tiles]
        compute_rt[core.x][core.y] = [n]
        start += n

    # ----- kernels -----
    # Uniform CT superset: reader [phase][7 scalars][gb accessor][output accessor].
    reader_ct = (
        [
            _PHASE_REDUCE,
            _CB_GATHERED_SHARDS,  # scalar 1
            num_devices,  # scalar 2 (N)
            pages_per_shard,  # scalar 3 (P)
            0,  # scalar 4 (pad)
            0,  # scalar 5 (pad)
            0,  # scalar 6 (pad)
            0,  # scalar 7 (pad)
        ]
        + gather_buffer_ta
        + output_ta
    )

    # writer [phase][7 scalars][output accessor].
    writer_ct = [
        _PHASE_REDUCE,
        _CB_REDUCED,  # scalar 1
        0,  # scalar 2 (pad)
        0,  # scalar 3 (pad)
        0,  # scalar 4 (pad)
        0,  # scalar 5 (pad)
        0,  # scalar 6 (pad)
        0,  # scalar 7 (pad)
    ] + output_ta

    compute_ct = [_CB_GATHERED_SHARDS, _CB_REDUCED, num_devices]

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_reduce_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_reduce_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "all_reduce_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        # fp32_dest_acc: accumulate the N-way sum in fp32 DST (precision for both
        # the bf16 sum-of-N rounding budget and the float32 secondary dtype).
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
        cbs=[cb_gathered, cb_reduced],
    )


def create_reduce_mesh_program_descriptor(
    gather_buffer: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    num_devices: int,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = gather_buffer.device()

    grid = mesh_device.compute_with_storage_grid_size()
    tile_size = output_tensor.buffer_page_size()
    pages_per_shard = output_tensor.buffer_num_pages()  # P output-tile positions

    gather_buffer_ta = list(ttnn.TensorAccessorArgs(gather_buffer).get_compile_time_args())
    output_ta = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Pure local compute — the program is identical on every device (uniform mesh
    # buffer addresses), but build a fresh ProgramDescriptor per device to mirror
    # the proven all_gather idiom (one program object per MeshCoordinateRange).
    mesh_pd = ttnn.MeshProgramDescriptor()
    for i in range(num_devices):
        coord_i = ttnn.MeshCoordinate(0, i)
        program = _build_reduce_device_program(
            gather_buffer,
            output_tensor,
            num_devices,
            grid,
            tile_size,
            pages_per_shard,
            gather_buffer_ta,
            output_ta,
        )
        mesh_pd[ttnn.MeshCoordinateRange(coord_i, coord_i)] = program
    return mesh_pd


def _num_line_devices(mesh_device) -> int:
    """Number of devices on the 1-D line (mesh view is (1, N))."""
    n = 1
    for d in tuple(mesh_device.shape):
        n *= d
    return n
