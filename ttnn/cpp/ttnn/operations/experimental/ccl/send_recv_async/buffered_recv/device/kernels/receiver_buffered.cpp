// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "ttnn/operations/experimental/ccl/send_recv_async/buffered_common/buffered_async_types.hpp"
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t handshake_page_size = get_compile_time_arg_val(1);
// Number of output tensors that make up the ring of receive buffers.
constexpr uint32_t num_output_tensors = get_compile_time_arg_val(2);

// Handshake landing-zone layout (must match buffered_send/device/kernels/sender_buffered.cpp).
// The OutputTensorInfo struct is bulk-written at INFO_STRUCT_OFFSET (kept 0 so the NoC write
// destination is aligned), then the 4-byte valid flag in the last word of the handshake page is set
// last so the sender never observes a partially-written struct.
constexpr uint32_t INFO_STRUCT_OFFSET = 0;
constexpr uint32_t INFO_VALID_OFFSET = handshake_page_size - sizeof(uint32_t);

FORCE_INLINE void fabric_inline_write_upstream(
    const SocketReceiverInterface& receiver_socket,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_addr,
    uint64_t dst_noc_addr,
    uint32_t value) {
    fabric_set_unicast_route(packet_header_addr, receiver_socket);
    packet_header_addr->to_noc_unicast_inline_write(NocUnicastInlineWriteCommandHeader{dst_noc_addr, value});
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)packet_header_addr, sizeof(PACKET_HEADER_TYPE));
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t rt_args_idx = 0;
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_page_size = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_pages = get_arg_val<uint32_t>(rt_args_idx++);
    // Persistent L1_SMALL buffer (zero-initialized on the host) used to coordinate receive-buffer
    // availability. Replaces the previously caller-provided global semaphore.
    uint32_t coordination_buffer_addr = get_arg_val<uint32_t>(rt_args_idx++);

    // Base address of each output tensor in the ring of receive buffers.
    uint32_t output_base_addrs[num_output_tensors];
    for (uint32_t i = 0; i < num_output_tensors; ++i) {
        output_base_addrs[i] = get_arg_val<uint32_t>(rt_args_idx++);
    }

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    fabric_connection.open();

    // The coordination buffer stores OutputTensorInfo, including the receive-buffer ring addresses
    // and the monotonic read/write counters shared with the sender.
    DPRINT("coordination_buffer_addr = {}\n", coordination_buffer_addr);
    auto* output_tensor_info = reinterpret_cast<volatile tt_l1_ptr OutputTensorInfo*>(coordination_buffer_addr);

    DPRINT("Output tensor info size = {}\n", sizeof(OutputTensorInfo));

    // Create Socket Interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, handshake_page_size);

    invalidate_l1_cache();
    if (output_tensor_info->num_tensors == 0) {
        DPRINT("Coordination buffer is zero-initialized\n");

        DPRINT("Num output tensors: {}\n", num_output_tensors);
        //////////////////////////////////////////////////
        // STEP 1: receive the sender's advertised handshake-buffer address
        //////////////////////////////////////////////////
        DPRINT("Waiting for handshake address from sender\n");
        socket_wait_for_pages(receiver_socket, 1);
        uint32_t sender_handshake_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_socket.read_ptr)[0];
        socket_pop_pages(receiver_socket, 1);
        fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
        DPRINT("Received handshake address  {} from sender \n", sender_handshake_addr);

        //////////////////////////////////////////////////
        // STEP 2: write the destination tensor info back into the sender's handshake buffer, advertising
        // the whole ring of receive-buffer base addresses. Bulk-write the struct first, then set the
        // valid flag last so the sender never observes a partially-written struct.
        //
        // The sender uses the advertised base addresses with monotonic read/write counters to pick
        // an available receive buffer.
        //////////////////////////////////////////////////
        uint32_t upstream_noc_x = receiver_socket.d2d.upstream_noc_x;
        uint32_t upstream_noc_y = receiver_socket.d2d.upstream_noc_y;

        // Stage the OutputTensorInfo struct in its dedicated CB (L1) so it can be used directly as the
        // source of the NoC payload write back to the sender.
        output_tensor_info->num_tensors = num_output_tensors;
        output_tensor_info->page_size = output_page_size;
        output_tensor_info->num_pages = num_pages;
        output_tensor_info->write_index[0] = 0;
        output_tensor_info->read_index[0] = 0;
        output_tensor_info->sender_config_l1_addr = sender_handshake_addr;
        output_tensor_info->receiver_config_l1_addr = coordination_buffer_addr;
        for (uint32_t i = 0; i < num_output_tensors; ++i) {
            output_tensor_info->base_addr[i] = output_base_addrs[i];
        }
        DPRINT("output_tensor_info_addr = {}\n", coordination_buffer_addr);
        uint64_t struct_dst_noc_addr = get_noc_addr(upstream_noc_x, upstream_noc_y, sender_handshake_addr);
        fabric_set_unicast_route(socket_packet_header_addr, receiver_socket);
        socket_packet_header_addr->to_noc_unicast_write(
            NocUnicastCommandHeader{struct_dst_noc_addr}, sizeof(OutputTensorInfo));
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_without_header_non_blocking_from_address(
            coordination_buffer_addr, sizeof(OutputTensorInfo));
        fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)socket_packet_header_addr, sizeof(PACKET_HEADER_TYPE));

        DPRINT("Wrote destination tensor info to sender's handshake buffer at address {}\n", sender_handshake_addr);
        update_socket_config(receiver_socket);
    }
    //////////////////////////////////////////////////
    // STEP 3: wait for the completion token from the sender
    //////////////////////////////////////////////////
    DPRINT(
        "Write Index addr = {}, Read Index addr = {}\n",
        (uint32_t)output_tensor_info->write_index,
        (uint32_t)output_tensor_info->read_index);
    DPRINT(
        "On receiver: Write Index = {}, Read Index = {}\n",
        output_tensor_info->write_index[0],
        output_tensor_info->read_index[0]);
    do {
        invalidate_l1_cache();
    } while (output_tensor_info->read_index[0] == output_tensor_info->write_index[0]);
    output_tensor_info->read_index[0] = output_tensor_info->read_index[0] + 1;

    uint32_t write_l1_addr = output_tensor_info->sender_config_l1_addr + offsetof(OutputTensorInfo, read_index);
    uint64_t write_noc_addr =
        get_noc_addr(receiver_socket.d2d.upstream_noc_x, receiver_socket.d2d.upstream_noc_y, write_l1_addr);
    fabric_set_unicast_route(socket_packet_header_addr, receiver_socket);
    socket_packet_header_addr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{write_noc_addr, 1});
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)socket_packet_header_addr, sizeof(PACKET_HEADER_TYPE));
    fabric_connection.close();
    DPRINT("Closed fabric connection\n");
}
