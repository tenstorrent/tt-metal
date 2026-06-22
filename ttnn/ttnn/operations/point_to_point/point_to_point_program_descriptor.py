# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
point_to_point — MeshProgramDescriptor builder.

Builds the two-program MeshProgramDescriptor described in op_design.md:

    send    @ sender_coord    : reader_send  + writer_send
    receive @ receiver_coord  : reader_receive + writer_receive

Every other mesh device runs nothing. The fabric-connection runtime args are framed as
``[has_forward][fwd conn args][has_backward][bwd conn args]`` at the index the kernel-side
FabricStreamSender records (conn_arg_idx = 9), exactly mirroring the C++ host helper
``ttnn::ccl::dataflow::append_ccl_fabric_rt_args`` (ccl_helpers_dataflow_host.hpp:219-237).
``ttnn.setup_fabric_connection`` is the Python equivalent of that helper's inner
``append_fabric_connection_rt_args`` (one direction) and also appends the connection
SemaphoreDescriptors to the program it is given.
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels" / "dataflow"

# Send program CB indices.
CB_SENDER_PAGES = 0  # c_0: input shard pages (reader_send -> writer_send)
CB_PACKET_SCRATCH = 1  # c_1: page->packet coalescing scratch (writer_send single owner)

# Receive program CB indices.
CB_PACKET_LANDING = 0  # c_0: one landed packet, local NoC read (reader_receive single owner)
CB_RECEIVER_PAGES = 1  # c_1: de-coalesced pages (reader_receive -> writer_receive)

LINK_IDX = 0  # single-link implementation


def _append_fabric_block(rt_args_ref, src_id, neighbor_id, program, core, is_forward):
    """Append the fabric-connection RT-arg block in the exact layout FabricStreamSender consumes.

    Mirrors append_ccl_fabric_rt_args:
        [has_forward][<forward conn args> if fwd][has_backward][<backward conn args> if bwd]
    The leading has_forward flag also encodes the send direction (the kernel peeks it).
    """
    rt_args_ref.append(int(is_forward))  # has_forward
    if is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, LINK_IDX, program, core))
    rt_args_ref.append(int(not is_forward))  # has_backward
    if not is_forward:
        rt_args_ref.extend(ttnn.setup_fabric_connection(src_id, neighbor_id, LINK_IDX, program, core))


def create_mesh_program_descriptor(
    mesh_device,
    input_tensor,
    intermediate_tensor,
    output_tensor,
    sender_coord,
    receiver_coord,
    topology,
    sem_addr,
    packet_dims,
):
    """Build the two-program MeshProgramDescriptor for one sender->receiver transfer."""
    l1_alignment = ttnn.get_l1_alignment()
    input_dtype = input_tensor.dtype
    output_dtype = output_tensor.dtype

    input_page_size_bytes = input_tensor.buffer_page_size()
    input_num_pages = input_tensor.buffer_num_pages()
    output_page_size_bytes = output_tensor.buffer_page_size()
    output_num_pages = output_tensor.buffer_num_pages()

    packet_size_bytes = packet_dims.packet_size_bytes
    pages_per_packet = packet_dims.pages_per_packet
    page_segments = packet_dims.page_segments

    aligned_input_page_size_bytes = ((input_page_size_bytes + l1_alignment - 1) // l1_alignment) * l1_alignment

    # Routes. The sender routes to the receiver; the receiver computes the reverse route
    # for its "ready" inc back to the sender. ccl_dm_route owns the fwd/bwd sign reversal
    # and the Ring short-way choice.
    sender_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, sender_coord, receiver_coord, topology)
    receiver_route = ttnn._ttnn.fabric.ccl_dm_route(mesh_device, receiver_coord, sender_coord, topology)
    sender_fabric_id = mesh_device.get_fabric_node_id(sender_coord)
    receiver_fabric_id = mesh_device.get_fabric_node_id(receiver_coord)

    core = ttnn.CoreCoord(0, 0)
    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    mesh_desc = ttnn.MeshProgramDescriptor()

    # ===================== SEND PROGRAM (@ sender_coord) =====================
    cb_sender_pages = ttnn.CBDescriptor(
        total_size=2 * aligned_input_page_size_bytes,  # double-buffer streaming
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_SENDER_PAGES,
                data_format=input_dtype,
                page_size=aligned_input_page_size_bytes,
            )
        ],
    )
    cb_packet_scratch = ttnn.CBDescriptor(
        total_size=packet_size_bytes,  # one scratch packet
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_PACKET_SCRATCH,
                data_format=input_dtype,
                page_size=packet_size_bytes,
            )
        ],
    )

    reader_send_ct = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    reader_send_rt = ttnn.RuntimeArgs()
    reader_send_rt[core.x][core.y] = [
        input_tensor.buffer_address(),
        input_num_pages,
        0,  # page_idx_start
        input_page_size_bytes,
    ]
    reader_send_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_send.cpp"),
        core_ranges=core_set,
        compile_time_args=reader_send_ct,
        runtime_args=reader_send_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_send_ct = [CB_SENDER_PAGES, CB_PACKET_SCRATCH, l1_alignment]
    writer_send_ct.extend(ttnn.TensorAccessorArgs(intermediate_tensor).get_compile_time_args())
    writer_send_rt = ttnn.RuntimeArgs()
    writer_send_rt[core.x][core.y] = [
        intermediate_tensor.buffer_address(),  # 0
        0,  # 1: page_idx_start
        input_num_pages,  # 2: page_idx_end
        sender_route.num_hops,  # 3
        input_page_size_bytes,  # 4
        packet_size_bytes,  # 5: payload size for each armed write
        pages_per_packet,  # 6
        page_segments,  # 7
        sem_addr,  # 8
        # fabric block appended below at index 9
    ]
    writer_send_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_send.cpp"),
        core_ranges=core_set,
        compile_time_args=writer_send_ct,
        runtime_args=writer_send_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    send_program = ttnn.ProgramDescriptor(
        kernels=[reader_send_kernel, writer_send_kernel],
        semaphores=[],
        cbs=[cb_sender_pages, cb_packet_scratch],
    )
    # writer_send is kernel index 1; append its fabric block (mutates send_program.semaphores).
    ws_rt_ref = send_program.kernels[1].runtime_args[core.x][core.y]
    _append_fabric_block(
        ws_rt_ref, sender_fabric_id, sender_route.neighbor_id, send_program, core, sender_route.is_forward
    )

    mesh_desc[ttnn.MeshCoordinateRange(sender_coord, sender_coord)] = send_program

    # ===================== RECEIVE PROGRAM (@ receiver_coord) =====================
    cb_packet_landing = ttnn.CBDescriptor(
        total_size=packet_size_bytes,  # one landed packet
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_PACKET_LANDING,
                data_format=input_dtype,
                page_size=packet_size_bytes,
            )
        ],
    )
    receiver_cb_num_pages = 3 * pages_per_packet
    cb_receiver_pages = ttnn.CBDescriptor(
        total_size=receiver_cb_num_pages * output_page_size_bytes,
        core_ranges=core_set,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=CB_RECEIVER_PAGES,
                data_format=output_dtype,
                page_size=output_page_size_bytes,
            )
        ],
    )

    reader_receive_ct = [CB_PACKET_LANDING, CB_RECEIVER_PAGES, l1_alignment]
    reader_receive_ct.extend(ttnn.TensorAccessorArgs(intermediate_tensor).get_compile_time_args())
    reader_receive_rt = ttnn.RuntimeArgs()
    reader_receive_rt[core.x][core.y] = [
        0,  # 0: page_idx_start
        output_num_pages,  # 1: page_idx_end
        pages_per_packet,  # 2
        intermediate_tensor.buffer_address(),  # 3
        packet_size_bytes,  # 4
        output_page_size_bytes,  # 5
        page_segments,  # 6
        sem_addr,  # 7
        receiver_route.num_hops,  # 8: hops back to the sender (for the "ready" inc)
        # fabric block appended below at index 9
    ]
    reader_receive_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_receive.cpp"),
        core_ranges=core_set,
        compile_time_args=reader_receive_ct,
        runtime_args=reader_receive_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_receive_ct = [CB_RECEIVER_PAGES]
    writer_receive_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_receive_rt = ttnn.RuntimeArgs()
    writer_receive_rt[core.x][core.y] = [
        output_tensor.buffer_address(),
        output_num_pages,
        0,  # page_idx_start
        output_page_size_bytes,
    ]
    writer_receive_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_receive.cpp"),
        core_ranges=core_set,
        compile_time_args=writer_receive_ct,
        runtime_args=writer_receive_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    receive_program = ttnn.ProgramDescriptor(
        kernels=[reader_receive_kernel, writer_receive_kernel],
        semaphores=[],
        cbs=[cb_packet_landing, cb_receiver_pages],
    )
    # reader_receive is kernel index 0; append its fabric block (mutates receive_program.semaphores).
    rr_rt_ref = receive_program.kernels[0].runtime_args[core.x][core.y]
    _append_fabric_block(
        rr_rt_ref, receiver_fabric_id, receiver_route.neighbor_id, receive_program, core, receiver_route.is_forward
    )

    mesh_desc[ttnn.MeshCoordinateRange(receiver_coord, receiver_coord)] = receive_program

    return mesh_desc
