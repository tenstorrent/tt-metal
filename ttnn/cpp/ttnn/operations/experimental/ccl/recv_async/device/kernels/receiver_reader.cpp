// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "dataflow_api.h"
#include "socket_api.h"
#include "tt_metal/hw/inc/accessor/tensor_accessor.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t scratch_buffer_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t socket_block_size = get_compile_time_arg_val(2);  // This is assumed to be aligned
constexpr uint32_t socket_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_blocks = get_compile_time_arg_val(4);
constexpr uint32_t num_pages_per_block = get_compile_time_arg_val(5);
constexpr uint32_t block_remainder_pages = get_compile_time_arg_val(6);
constexpr bool is_dram = get_compile_time_arg_val(7);

template <uint32_t num_pages_per_block, uint32_t page_size, uint32_t cb_id, bool is_dram>
FORCE_INLINE void read_data_from_remote_core(
    SocketReceiverInterface& receiver_socket,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    uint32_t bank_id,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr) {
    constexpr uint32_t block_size = num_pages_per_block * page_size;
    socket_wait_for_pages(receiver_socket, 1);
    cb_reserve_back(cb_id, num_pages_per_block);
    auto remote_read_addr = get_noc_addr_from_bank_id<is_dram>(bank_id, receiver_socket.read_ptr);
    auto l1_write_addr = get_write_ptr(cb_id);
    noc_async_read<block_size>(remote_read_addr, l1_write_addr, block_size);
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages_per_block);
    socket_pop_pages(receiver_socket, 1);
    fabric_socket_notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr);
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    // Setup Fabric Headers and Connections
    size_t rt_args_idx = 0;
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t bank_id = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    // This kernel relies on two fabric headers stored in fabric_packet_header_cb:
    //  - data_packet_header: Used for issuing reads from upstream data cores
    //  - socket_packet_header: Used by socket APIs for control flow
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    fabric_connection.open();

    // Create Socket Interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, socket_block_size);

    for (uint32_t i = 0; i < num_blocks; ++i) {
        read_data_from_remote_core<num_pages_per_block, socket_page_size, scratch_buffer_cb_id, is_dram>(
            receiver_socket, fabric_connection, bank_id, socket_packet_header_addr);
    }
    if (block_remainder_pages > 0) {
        read_data_from_remote_core<block_remainder_pages, socket_page_size, scratch_buffer_cb_id, is_dram>(
            receiver_socket, fabric_connection, bank_id, socket_packet_header_addr);
    }
    update_socket_config(receiver_socket);
    fabric_connection.close();
}
