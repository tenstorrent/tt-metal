# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""point_to_point — MeshProgramDescriptor assembly.

Builds the two-program ``ttnn.MeshProgramDescriptor`` for the fabric transfer:
a SEND program at ``sender_coord`` and a RECEIVE program at ``receiver_coord``.
No program is emitted for any other mesh coordinate, so only the two endpoints
run kernels.

Packet framing (``ccl_packet_dims``) and 1-D routing (``ccl_dm_route``) are the
Python-bound CCL host helpers — they own the bf16 ``bit_floor`` packet sizing,
both page<->packet regimes, and the fabric forward/backward sign reversal +
ring shorter-way. The fabric-connection runtime-arg block is laid out exactly
as ``append_ccl_fabric_rt_args`` does (``[has_forward][fwd?][has_backward][bwd?]``),
which the kernel-side ``FabricStreamSender`` consumes.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# CB indices (semantic; see op_design.md "Circular Buffers").
_CB_INPUT_PAGES = 0  # sender: input shard pages (reader -> writer)
_CB_OUTPUT_PAGES = 16  # receiver: de-coalesced output pages (reader -> writer)
_CB_PACKET_SCRATCH = 24  # both: L1 working buffer for one coalesced packet

_LINK_IDX = 0  # single-link transfer


def _packet_dims(input_tensor: ttnn.Tensor):
    """Frame the per-device shard into fabric packets (owns the bf16 bit_floor)."""
    l1_alignment = ttnn.get_l1_alignment()
    return ttnn._ttnn.fabric.ccl_packet_dims(
        input_tensor.dtype,
        input_tensor.buffer_page_size(),
        input_tensor.buffer_num_pages(),
        l1_alignment,
    )


def resolve_intermediate_spec(input_tensor: ttnn.Tensor) -> ttnn.TensorSpec:
    """Resolve the op-internal staging-tensor spec.

    The intermediate is addressed PER-PACKET (page index = packet_idx, page size
    overridden to packet_size_bytes), carrying raw bytes only. We stage it as a
    row-major uint32 interleaved buffer of ``[total_packets, packet_size_bytes/4]``
    in the input's buffer_type, which holds exactly ``total_packets`` packets for
    EVERY dtype (uint32 sidesteps ``element_size`` being undefined for block-float
    formats such as bfloat8_b). packet_size_bytes is always a multiple of the L1
    alignment (>= 16), hence of 4.
    """
    pd = _packet_dims(input_tensor)
    packet_page_dim_u32 = pd.packet_size_bytes // 4
    shape = ttnn.Shape([pd.total_packets, packet_page_dim_u32])
    buffer_type = input_tensor.memory_config().buffer_type
    return ttnn.TensorSpec(shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, buffer_type)


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


def create_mesh_program_descriptor(
    input_tensor: ttnn.Tensor,
    intermediate_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    sender_coord: ttnn.MeshCoordinate,
    receiver_coord: ttnn.MeshCoordinate,
    topology: ttnn.Topology,
    sem_addr: int,
) -> ttnn.MeshProgramDescriptor:
    mesh_device = input_tensor.device()
    l1_alignment = ttnn.get_l1_alignment()
    data_format = input_tensor.dtype

    # ----- per-device page + packet metrics -----
    input_page_size = input_tensor.buffer_page_size()
    num_pages = input_tensor.buffer_num_pages()
    output_page_size = output_tensor.buffer_page_size()
    aligned_input_page_size = ((input_page_size + l1_alignment - 1) // l1_alignment) * l1_alignment

    pd = _packet_dims(input_tensor)
    packet_size_bytes = pd.packet_size_bytes
    pages_per_packet = pd.pages_per_packet
    page_segments = pd.page_segments

    # ----- 1-D routes (ccl_dm_route owns the fwd/bwd sign reversal + ring short-way) -----
    send_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, sender_coord, receiver_coord, topology)
    recv_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, receiver_coord, sender_coord, topology)

    sender_fabric_id = mesh_device.get_fabric_node_id(sender_coord)
    receiver_fabric_id = mesh_device.get_fabric_node_id(receiver_coord)

    core = ttnn.CoreCoord(0, 0)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    mesh_pd = ttnn.MeshProgramDescriptor()

    # ============================ SEND PROGRAM ============================
    sender_input_cb = ttnn.CBDescriptor(
        total_size=2 * aligned_input_page_size,  # streaming double-buffer
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_CB_INPUT_PAGES, data_format=data_format, page_size=aligned_input_page_size
            )
        ],
    )
    sender_packet_cb = ttnn.CBDescriptor(
        total_size=packet_size_bytes,  # one coalesced packet of working L1
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_CB_PACKET_SCRATCH, data_format=data_format, page_size=packet_size_bytes
            )
        ],
    )

    sender_reader_ct = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    sender_reader_rt = ttnn.RuntimeArgs()
    sender_reader_rt[core.x][core.y] = [input_tensor.buffer_address(), num_pages, 0, input_page_size]
    sender_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_sender_reader.cpp"),
        core_ranges=core_set,
        compile_time_args=sender_reader_ct,
        runtime_args=sender_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    sender_writer_ct = [_CB_INPUT_PAGES, _CB_PACKET_SCRATCH, l1_alignment]
    sender_writer_ct.extend(ttnn.TensorAccessorArgs(intermediate_tensor).get_compile_time_args())
    sender_writer_rt = ttnn.RuntimeArgs()
    sender_writer_rt[core.x][core.y] = [
        intermediate_tensor.buffer_address(),  # 0: receiver_base_address
        0,  # 1: page_idx_start
        num_pages,  # 2: page_idx_end
        send_route.num_hops,  # 3: dst_num_hops
        input_page_size,  # 4: page_size_bytes
        packet_size_bytes,  # 5: payload_size_bytes (per-packet on-wire size)
        pages_per_packet,  # 6: max_pages_per_packet
        page_segments,  # 7: page_segments
        sem_addr,  # 8: receive_semaphore_addr (global sem address)
    ]
    sender_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_sender_writer.cpp"),
        core_ranges=core_set,
        compile_time_args=sender_writer_ct,
        runtime_args=sender_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    sender_program = ttnn.ProgramDescriptor(
        kernels=[sender_reader, sender_writer],
        semaphores=[],
        cbs=[sender_input_cb, sender_packet_cb],
    )
    # Fabric connection args live on the SEND writer (kernel index 1), starting
    # at runtime-arg index 9 (right after the 9 scalar args above).
    writer_rt_ref = sender_program.kernels[1].runtime_args[core.x][core.y]
    _append_fabric_rt_args(
        writer_rt_ref, sender_fabric_id, send_route.neighbor_id, sender_program, core, send_route.is_forward
    )
    mesh_pd[ttnn.MeshCoordinateRange(sender_coord, sender_coord)] = sender_program

    # ========================== RECEIVE PROGRAM ==========================
    recv_packet_cb = ttnn.CBDescriptor(
        total_size=packet_size_bytes,  # one landed packet of working L1
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=_CB_PACKET_SCRATCH, data_format=data_format, page_size=packet_size_bytes
            )
        ],
    )
    recv_output_cb = ttnn.CBDescriptor(
        total_size=3 * pages_per_packet * output_page_size,  # de-coalesced output stream
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=_CB_OUTPUT_PAGES, data_format=data_format, page_size=output_page_size)
        ],
    )

    recv_reader_ct = [_CB_PACKET_SCRATCH, _CB_OUTPUT_PAGES, l1_alignment]
    recv_reader_ct.extend(ttnn.TensorAccessorArgs(intermediate_tensor).get_compile_time_args())
    recv_reader_rt = ttnn.RuntimeArgs()
    recv_reader_rt[core.x][core.y] = [
        0,  # 0: page_idx_start
        num_pages,  # 1: page_idx_end
        pages_per_packet,  # 2: max_pages_per_packet
        intermediate_tensor.buffer_address(),  # 3: intermediate_base_addr (local landing buffer)
        packet_size_bytes,  # 4: packet_size_bytes
        output_page_size,  # 5: page_size_bytes
        page_segments,  # 6: page_segments
        sem_addr,  # 7: sender_semaphore_addr (global sem address)
        recv_route.num_hops,  # 8: sender_num_hops (ack route receiver->sender)
    ]
    recv_reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_receiver_reader.cpp"),
        core_ranges=core_set,
        compile_time_args=recv_reader_ct,
        runtime_args=recv_reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    recv_writer_ct = [_CB_OUTPUT_PAGES]
    recv_writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    recv_writer_rt = ttnn.RuntimeArgs()
    recv_writer_rt[core.x][core.y] = [output_tensor.buffer_address(), num_pages, 0, output_page_size]
    recv_writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "point_to_point_receiver_writer.cpp"),
        core_ranges=core_set,
        compile_time_args=recv_writer_ct,
        runtime_args=recv_writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    receiver_program = ttnn.ProgramDescriptor(
        kernels=[recv_reader, recv_writer],
        semaphores=[],
        cbs=[recv_packet_cb, recv_output_cb],
    )
    # Fabric connection args live on the RECEIVE reader (kernel index 0), starting
    # at runtime-arg index 9 (right after the 9 scalar args above).
    reader_rt_ref = receiver_program.kernels[0].runtime_args[core.x][core.y]
    _append_fabric_rt_args(
        reader_rt_ref, receiver_fabric_id, recv_route.neighbor_id, receiver_program, core, recv_route.is_forward
    )
    mesh_pd[ttnn.MeshCoordinateRange(receiver_coord, receiver_coord)] = receiver_program

    return mesh_pd
