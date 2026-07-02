# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""point_to_point — MeshProgramDescriptor assembly (single-link 1-D fabric unicast).

Builds exactly TWO ``ttnn.ProgramDescriptor`` entries in a
``ttnn.MeshProgramDescriptor``:

  * a SEND program pinned to ``MeshCoordinateRange(sender_coord, sender_coord)``
  * a RECEIVE program pinned to ``MeshCoordinateRange(receiver_coord, receiver_coord)``

Every other mesh device runs no program (its output shard is left as the seeded
copy of its input). Each participating device uses a single Tensix core (0, 0).

Data path (pure byte movement, no compute thread):

    SENDER                                          RECEIVER
      input DRAM shard                                output DRAM shard
        | sender_reader (noc_async_read)                ^ receiver_writer (noc_async_write)
        v                                                |
      cb_send_pages                                   cb_recv_pages
        | sender_writer (coalesce -> packet)             ^ receiver_reader (de-coalesce)
        v                                                |
      cb_send_packet                                  cb_recv_packet
        | FabricStream unicast write --- fabric ----> intermediate DRAM (receiver copy)
        +--------------------------------------------->  (read locally by receiver_reader)

Packet framing (owned by ccl_packet_dims): a page smaller than a fabric transfer
unit packs N pages per packet (pages_per_packet > 1, page_segments == 1); a larger
page is split across page_segments packets. The kernels loop `total_packets` and
clamp the final packet; both sides recompute packet boundaries from the shared
scalars, so their loop arithmetic must stay identical.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic; see op_design.md "Circular Buffers").
_CB_SEND_PAGES = 0  # send: reader -> writer, one input shard page per slot
_CB_SEND_PACKET = 24  # send: writer coalesce scratch (one packet)
_CB_RECV_PAGES = 16  # receive: reader -> writer, one output shard page per slot
_CB_RECV_PACKET = 24  # receive: reader local-read scratch (one packet)

_LINK_IDX = 0  # single-link transfer
_CORE = ttnn.CoreCoord(0, 0)


def _round_up(value: int, mult: int) -> int:
    return ((value + mult - 1) // mult) * mult


def _append_fabric_rt_args(rt_args_ref, src_id, neighbor_id, program, core, is_forward):
    """Mirror ttnn::ccl::dataflow::append_ccl_fabric_rt_args.

    After the call the block beginning at the current rt_args length is:
        [has_forward][<forward conn args> if fwd][has_backward][<backward conn args> if bwd]
    The kernel records that start index as conn_arg_idx; its leading has_forward flag
    also equals the send direction, which the kernel peeks for `is_forward`.
    ``setup_fabric_connection`` also mutates ``program`` (appends SemaphoreDescriptors).
    """
    rt_args_ref.append(int(is_forward))  # has_forward
    if is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))
    rt_args_ref.append(int(not is_forward))  # has_backward
    if not is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, _LINK_IDX, program, core))


def _single_core_set():
    return ttnn.CoreRangeSet([ttnn.CoreRange(_CORE, _CORE)])


def _build_send_program(
    input_tensor,
    intermediate_tensor,
    sender_fabric_id,
    send_route,
    sem_addr,
    page_size,
    aligned_page_size,
    num_pages,
    packet_size_bytes,
    pages_per_packet,
    page_segments,
    l1_alignment,
    data_format,
    input_ta,
    inter_ta,
):
    core_set = _single_core_set()

    cb_send_pages = ttnn.CBDescriptor(
        total_size=2 * aligned_page_size,  # double-buffered streaming page
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_SEND_PAGES, data_format=data_format, page_size=aligned_page_size)
        ],
    )
    cb_send_packet = ttnn.CBDescriptor(
        total_size=2 * packet_size_bytes,  # coalesce scratch (writer reserves 1, reuses)
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_SEND_PACKET, data_format=data_format, page_size=packet_size_bytes)
        ],
    )

    # --- sender reader (NCRISC): input DRAM -> cb_send_pages ---
    reader_ct = [_CB_SEND_PAGES] + input_ta
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[_CORE.x][_CORE.y] = [input_tensor.buffer_address(), num_pages, 0, page_size]
    sender_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_sender_reader.cpp"),
        core_ranges=core_set,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- sender writer (BRISC): coalesce -> fabric write to receiver intermediate ---
    writer_ct = [_CB_SEND_PAGES, _CB_SEND_PACKET, l1_alignment] + inter_ta
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[_CORE.x][_CORE.y] = [
        intermediate_tensor.buffer_address(),  # [0] receiver_base_address (intermediate)
        0,  # [1] page_idx_start
        num_pages,  # [2] page_idx_end
        send_route.num_hops,  # [3] dst_num_hops
        page_size,  # [4] page_size_bytes
        packet_size_bytes,  # [5] payload_size_bytes
        pages_per_packet,  # [6] max_pages_per_packet
        page_segments,  # [7] page_segments
        sem_addr,  # [8] receive_semaphore_addr
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
        cbs=[cb_send_pages, cb_send_packet],
    )

    # Fabric connection args live on the writer (kernel idx 1); block begins at RT index 9.
    ref = program.kernels[1].runtime_args[_CORE.x][_CORE.y]
    _append_fabric_rt_args(ref, sender_fabric_id, send_route.neighbor_id, program, _CORE, send_route.is_forward)

    return program


def _build_receive_program(
    output_tensor,
    intermediate_tensor,
    receiver_fabric_id,
    recv_route,
    sem_addr,
    page_size,
    aligned_page_size,
    num_pages,
    packet_size_bytes,
    pages_per_packet,
    page_segments,
    l1_alignment,
    data_format,
    output_ta,
    inter_ta,
):
    core_set = _single_core_set()

    cb_recv_packet = ttnn.CBDescriptor(
        total_size=2 * packet_size_bytes,  # local-read scratch (reader reserves 1, reuses)
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_RECV_PACKET, data_format=data_format, page_size=packet_size_bytes)
        ],
    )
    recv_pages_slots = max(2, 3 * pages_per_packet)
    cb_recv_pages = ttnn.CBDescriptor(
        total_size=recv_pages_slots * aligned_page_size,  # pipelined de-coalesced pages
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_RECV_PAGES, data_format=data_format, page_size=aligned_page_size)
        ],
    )

    # --- receiver reader (NCRISC): fabric ack + local read intermediate -> cb_recv_pages ---
    reader_ct = [_CB_RECV_PACKET, _CB_RECV_PAGES, l1_alignment] + inter_ta
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[_CORE.x][_CORE.y] = [
        0,  # [0] page_idx_start
        num_pages,  # [1] page_idx_end
        pages_per_packet,  # [2] max_pages_per_packet
        intermediate_tensor.buffer_address(),  # [3] intermediate_base_addr
        packet_size_bytes,  # [4] packet_size_bytes
        page_size,  # [5] page_size_bytes
        page_segments,  # [6] page_segments
        sem_addr,  # [7] sender_semaphore_addr
        recv_route.num_hops,  # [8] sender_num_hops
    ]
    receiver_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_receiver_reader.cpp"),
        core_ranges=core_set,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- receiver writer (BRISC): cb_recv_pages -> output DRAM ---
    writer_ct = [_CB_RECV_PAGES] + output_ta
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[_CORE.x][_CORE.y] = [output_tensor.buffer_address(), num_pages, 0, page_size]
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
        cbs=[cb_recv_packet, cb_recv_pages],
    )

    # Fabric connection args live on the reader (kernel idx 0); block begins at RT index 9.
    ref = program.kernels[0].runtime_args[_CORE.x][_CORE.y]
    _append_fabric_rt_args(ref, receiver_fabric_id, recv_route.neighbor_id, program, _CORE, recv_route.is_forward)

    return program


def create_mesh_program_descriptor(
    input_tensor: ttnn.Tensor,
    intermediate_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    sender_coord,
    receiver_coord,
    topology,
    sem_addr: int,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = input_tensor.device()

    l1_alignment = ttnn.get_l1_alignment()
    data_format = input_tensor.dtype
    page_size = input_tensor.buffer_page_size()
    num_pages = input_tensor.buffer_num_pages()
    aligned_page_size = _round_up(page_size, l1_alignment)

    pd = ttnn._ttnn.fabric.ccl_packet_dims(data_format, page_size, num_pages, l1_alignment)
    packet_size_bytes = pd.packet_size_bytes
    pages_per_packet = pd.pages_per_packet
    page_segments = pd.page_segments

    # Routes: the sender routes toward the receiver; the receiver routes toward the
    # sender for its "ready" ack. ccl_dm_route owns the fwd/bwd sign reversal + ring
    # short-way; each side's num_hops must agree so the atomic-inc lands on the right cell.
    send_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, sender_coord, receiver_coord, topology)
    recv_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, receiver_coord, sender_coord, topology)

    sender_fabric_id = mesh_device.get_fabric_node_id(sender_coord)
    receiver_fabric_id = mesh_device.get_fabric_node_id(receiver_coord)

    input_ta = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    inter_ta = list(ttnn.TensorAccessorArgs(intermediate_tensor).get_compile_time_args())
    output_ta = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    send_program = _build_send_program(
        input_tensor,
        intermediate_tensor,
        sender_fabric_id,
        send_route,
        sem_addr,
        page_size,
        aligned_page_size,
        num_pages,
        packet_size_bytes,
        pages_per_packet,
        page_segments,
        l1_alignment,
        data_format,
        input_ta,
        inter_ta,
    )
    receive_program = _build_receive_program(
        output_tensor,
        intermediate_tensor,
        receiver_fabric_id,
        recv_route,
        sem_addr,
        page_size,
        aligned_page_size,
        num_pages,
        packet_size_bytes,
        pages_per_packet,
        page_segments,
        l1_alignment,
        data_format,
        output_ta,
        inter_ta,
    )

    mesh_pd = ttnn.MeshProgramDescriptor()
    mesh_pd[ttnn.MeshCoordinateRange(sender_coord, sender_coord)] = send_program
    mesh_pd[ttnn.MeshCoordinateRange(receiver_coord, receiver_coord)] = receive_program
    return mesh_pd
