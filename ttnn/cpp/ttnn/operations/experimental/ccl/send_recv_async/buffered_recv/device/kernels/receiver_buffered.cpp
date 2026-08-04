// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "ttnn/operations/experimental/ccl/send_recv_async/buffered_common/buffered_async_types.hpp"
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t handshake_page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_output_tensors = get_compile_time_arg_val(2);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t rt_args_idx = 0;
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_page_size = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_pages = get_arg_val<uint32_t>(rt_args_idx++);
    // Persistent L1_SMALL buffer, zero-initialized on the host.
    uint32_t coordination_buffer_addr = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t output_base_addrs[num_output_tensors];
    for (uint32_t i = 0; i < num_output_tensors; ++i) {
        output_base_addrs[i] = get_arg_val<uint32_t>(rt_args_idx++);
    }

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    fabric_connection.open();

    auto* output_tensor_info = reinterpret_cast<volatile tt_l1_ptr OutputTensorInfo*>(coordination_buffer_addr);

    // Create Socket Interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, handshake_page_size);

    invalidate_l1_cache();
    if (output_tensor_info->num_tensors == 0) {
        //////////////////////////////////////////////////
        // STEP 1: receive the sender's advertised handshake-buffer address
        //////////////////////////////////////////////////
        socket_wait_for_pages(receiver_socket, 1);
        uint32_t sender_handshake_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_socket.read_ptr)[0];
        socket_pop_pages(receiver_socket, 1);
        fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);

        //////////////////////////////////////////////////
        // STEP 2: advertise the whole ring by copying OutputTensorInfo into the sender's landing
        // zone in one fabric write. The sender polls num_tensors, the struct's first field, so this
        // relies on the write landing as a unit.
        //////////////////////////////////////////////////
        uint32_t upstream_noc_x = receiver_socket.d2d.upstream_noc_x;
        uint32_t upstream_noc_y = receiver_socket.d2d.upstream_noc_y;

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
        uint64_t struct_dst_noc_addr = get_noc_addr(upstream_noc_x, upstream_noc_y, sender_handshake_addr);
        fabric_set_unicast_route(socket_packet_header_addr, receiver_socket);
        socket_packet_header_addr->to_noc_unicast_write(
            NocUnicastCommandHeader{struct_dst_noc_addr}, sizeof(OutputTensorInfo));
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_without_header_non_blocking_from_address(
            coordination_buffer_addr, sizeof(OutputTensorInfo));
        fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)socket_packet_header_addr, sizeof(PACKET_HEADER_TYPE));

        update_socket_config(receiver_socket);
    }
    //////////////////////////////////////////////////
    // STEP 3: wait for the sender to signal a filled buffer, then mirror the consumed count back so
    // it can reuse the slot.
    //////////////////////////////////////////////////
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
}
