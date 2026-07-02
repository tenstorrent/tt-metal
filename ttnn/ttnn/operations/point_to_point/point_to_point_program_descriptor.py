# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""point_to_point — MeshProgramDescriptor assembly (single-hop fabric unicast).

Builds exactly two ``ttnn.ProgramDescriptor``s parked on one
``ttnn.MeshProgramDescriptor``:

  * Send program at ``sender_coord``   — worker core (0,0): sender_reader (NCRISC)
    loads the input shard pages into cb_shard_pages; sender_writer (BRISC)
    coalesces pages into packets, fabric-unicasts them into the receiver device's
    INTERMEDIATE buffer, and signals "done".
  * Receive program at ``receiver_coord`` — worker core (0,0): receiver_reader
    (NCRISC) signals "ready", waits "done", reads its LOCAL intermediate copy back
    and de-coalesces the packets into cb_shard_out; receiver_writer (BRISC) stores
    cb_shard_out into the output shard.

No program runs on any other mesh coordinate. Cross-device coordination uses ONE
op-internal ``GlobalSemaphore`` (parked on the descriptor by the caller): the
receiver's ready-inc and the sender's done-inc are the SENDING halves owned by
the CCL helper; each side's ``noc_semaphore_wait_min`` (+ reset) is the op-owned
WAITING half.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic; see op_design.md "Circular Buffers").
_CB_SHARD_PAGES = 0  # send:    reader -> writer, input shard pages
_CB_SHARD_OUT = 16  # receive: reader -> writer, output shard pages
_CB_PACKET_SEND = 24  # send:    writer scratch, one coalesced packet
_CB_PACKET_RECV = 24  # receive: reader scratch, one landed packet

_LINK_IDX = 0  # single-link transfer


@dataclass
class PacketDims:
    packet_size_bytes: int
    pages_per_packet: int
    page_segments: int
    total_packets: int


def _round_up(value: int, mult: int) -> int:
    return ((value + mult - 1) // mult) * mult


def _append_fabric_rt_args(rt_ref, src_id, neighbor_id, program, core, is_forward):
    """Mirror ttnn::ccl::dataflow::append_ccl_fabric_rt_args.

    After the call the block beginning at the current rt_args length is:
        [has_forward][<forward conn args> if fwd][has_backward][<backward conn args> if bwd]
    The kernel records that start index as conn_arg_idx; its leading has_forward
    flag also equals the send direction, which the kernel peeks for is_forward.
    ``setup_fabric_connection`` also mutates ``program`` (appends SemaphoreDescriptors).
    """
    rt_ref.append(int(is_forward))  # has_forward
    if is_forward:
        rt_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))
    rt_ref.append(int(not is_forward))  # has_backward
    if not is_forward:
        rt_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))


def _build_send_program(
    mesh_device,
    input_tensor,
    intermediate_tensor,
    sender_coord,
    receiver_coord,
    topology,
    sem_addr,
    packet_dims,
    data_format,
    page_size,
    aligned_page_size,
    num_pages,
    l1_alignment,
):
    core = ttnn.CoreCoord(0, 0)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    input_ta = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    intermediate_ta = list(ttnn.TensorAccessorArgs(intermediate_tensor).get_compile_time_args())

    # ----- circular buffers -----
    cb_shard_pages = ttnn.CBDescriptor(
        total_size=2 * aligned_page_size,  # double-buffered streaming
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_SHARD_PAGES, data_format=data_format, page_size=aligned_page_size)
        ],
    )
    cb_packet_send = ttnn.CBDescriptor(
        total_size=packet_dims.packet_size_bytes,  # one coalesced packet scratch
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_CB_PACKET_SEND, data_format=data_format, page_size=packet_dims.packet_size_bytes
            )
        ],
    )

    # ----- sender_reader (NCRISC): input shard pages -> cb_shard_pages -----
    reader_ct = [_CB_SHARD_PAGES] + input_ta
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[core.x][core.y] = [
        input_tensor.buffer_address(),
        num_pages,
        page_size,
    ]
    sender_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_sender_reader.cpp"),
        core_ranges=core_set,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- sender_writer (BRISC): coalesce -> fabric write -> done inc -----
    route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, sender_coord, receiver_coord, topology)
    writer_ct = [_CB_SHARD_PAGES, _CB_PACKET_SEND, l1_alignment] + intermediate_ta
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[core.x][core.y] = [
        intermediate_tensor.buffer_address(),  # 0: receiver intermediate base (same addr on all devices)
        0,  # 1: page_idx_start
        num_pages,  # 2: page_idx_end
        route.num_hops,  # 3: dst_num_hops
        page_size,  # 4: raw page size
        packet_dims.packet_size_bytes,  # 5: payload size (== packet size)
        packet_dims.pages_per_packet,  # 6: max pages per packet
        packet_dims.page_segments,  # 7: page segments
        sem_addr,  # 8: (receiver) done-semaphore address
    ]
    sender_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_sender_writer.cpp"),
        core_ranges=core_set,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[sender_reader, sender_writer],
        semaphores=[],
        cbs=[cb_shard_pages, cb_packet_send],
    )

    # Fabric connection args live on the writer (kernel idx 1). Appended in place
    # after program creation so setup_fabric_connection can mutate the program.
    src_id = mesh_device.get_fabric_node_id(sender_coord)
    ref = program.kernels[1].runtime_args[core.x][core.y]
    _append_fabric_rt_args(ref, src_id, route.neighbor_id, program, core, route.is_forward)

    return program


def _build_receive_program(
    mesh_device,
    intermediate_tensor,
    output_tensor,
    sender_coord,
    receiver_coord,
    topology,
    sem_addr,
    packet_dims,
    data_format,
    page_size,
    aligned_page_size,
    num_pages,
    l1_alignment,
):
    core = ttnn.CoreCoord(0, 0)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    intermediate_ta = list(ttnn.TensorAccessorArgs(intermediate_tensor).get_compile_time_args())
    output_ta = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ----- circular buffers -----
    cb_packet_recv = ttnn.CBDescriptor(
        total_size=packet_dims.packet_size_bytes,  # one landed packet scratch
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_CB_PACKET_RECV, data_format=data_format, page_size=packet_dims.packet_size_bytes
            )
        ],
    )
    cb_shard_out = ttnn.CBDescriptor(
        total_size=3 * packet_dims.pages_per_packet * aligned_page_size,  # pipelined
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_SHARD_OUT, data_format=data_format, page_size=aligned_page_size)
        ],
    )

    # ----- receiver_reader (NCRISC): ready + wait done + read local intermediate + de-coalesce -----
    route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, receiver_coord, sender_coord, topology)
    reader_ct = [_CB_PACKET_RECV, _CB_SHARD_OUT, l1_alignment] + intermediate_ta
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[core.x][core.y] = [
        0,  # 0: page_idx_start
        num_pages,  # 1: page_idx_end
        packet_dims.pages_per_packet,  # 2: max pages per packet
        intermediate_tensor.buffer_address(),  # 3: local intermediate base
        packet_dims.packet_size_bytes,  # 4: packet size
        page_size,  # 5: raw output page size
        packet_dims.page_segments,  # 6: page segments
        sem_addr,  # 7: (sender) semaphore address
        route.num_hops,  # 8: sender_num_hops (for the ready ack route)
    ]
    receiver_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_receiver_reader.cpp"),
        core_ranges=core_set,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ----- receiver_writer (BRISC): cb_shard_out -> output shard -----
    writer_ct = [_CB_SHARD_OUT] + output_ta
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_pages,
        0,  # start_page_id
        page_size,
    ]
    receiver_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_receiver_writer.cpp"),
        core_ranges=core_set,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[receiver_reader, receiver_writer],
        semaphores=[],
        cbs=[cb_packet_recv, cb_shard_out],
    )

    # Fabric connection args live on the reader (kernel idx 0) for the ready ack.
    src_id = mesh_device.get_fabric_node_id(receiver_coord)
    ref = program.kernels[0].runtime_args[core.x][core.y]
    _append_fabric_rt_args(ref, src_id, route.neighbor_id, program, core, route.is_forward)

    return program


def create_mesh_program_descriptor(
    input_tensor: ttnn.Tensor,
    intermediate_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    sender_coord: ttnn.MeshCoordinate,
    receiver_coord: ttnn.MeshCoordinate,
    topology: ttnn.Topology,
    sem_addr: int,
    packet_dims: PacketDims,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = input_tensor.device()

    l1_alignment = ttnn.get_l1_alignment()
    data_format = input_tensor.dtype
    page_size = input_tensor.buffer_page_size()
    aligned_page_size = _round_up(page_size, l1_alignment)
    num_pages = input_tensor.buffer_num_pages()

    send_program = _build_send_program(
        mesh_device,
        input_tensor,
        intermediate_tensor,
        sender_coord,
        receiver_coord,
        topology,
        sem_addr,
        packet_dims,
        data_format,
        page_size,
        aligned_page_size,
        num_pages,
        l1_alignment,
    )
    receive_program = _build_receive_program(
        mesh_device,
        intermediate_tensor,
        output_tensor,
        sender_coord,
        receiver_coord,
        topology,
        sem_addr,
        packet_dims,
        data_format,
        page_size,
        aligned_page_size,
        num_pages,
        l1_alignment,
    )

    mesh_pd = ttnn.MeshProgramDescriptor()
    mesh_pd[ttnn.MeshCoordinateRange(sender_coord, sender_coord)] = send_program
    mesh_pd[ttnn.MeshCoordinateRange(receiver_coord, receiver_coord)] = receive_program
    return mesh_pd
