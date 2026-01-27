
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/debug/dprint.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t fabric_packet_header_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t output_page_size = get_compile_time_arg_val(1);  // This is assumed to be aligned
constexpr uint32_t socket_block_size = get_compile_time_arg_val(2);
constexpr uint32_t num_pages_per_ack = get_compile_time_arg_val(3);

constexpr uint32_t output_args_cta_idx = 4;
constexpr uint32_t output_args_crta_idx = 0;

FORCE_INLINE void notify_sender(
    SocketReceiverInterface& receiver_socket,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr,
    uint64_t upstream_bytes_acked_noc_addr) {
    // noc_async_writes_flushed();
    fabric_socket_notify_sender_stateful(
        receiver_socket, fabric_connection, socket_packet_header_addr, upstream_bytes_acked_noc_addr);
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // Setup Fabric Headers and Connections
    size_t rt_args_idx = 0;
    uint32_t socket_config_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_base_addr = get_arg_val<uint32_t>(rt_args_idx++);

    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* socket_packet_header_addr =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(get_write_ptr(fabric_packet_header_cb_id));
    fabric_connection.open();

    // Create Socket Interface
    SocketReceiverInterface receiver_socket = create_receiver_socket_interface(socket_config_addr);
    set_receiver_socket_page_size(receiver_socket, socket_block_size);

    auto output_addr_gen_args = TensorAccessorArgs<output_args_cta_idx, output_args_crta_idx>();
    auto output_addr_gen = TensorAccessor(output_addr_gen_args, output_base_addr, output_page_size);
    bool ack_sent = false;

    fabric_set_unicast_route(socket_packet_header_addr, receiver_socket);

    uint64_t upstream_bytes_acked_noc_addr = get_noc_addr(
        receiver_socket.upstream_noc_x, receiver_socket.upstream_noc_y, receiver_socket.upstream_bytes_acked_addr);

    constexpr uint32_t fwd_credit_addr = 1565632;
    constexpr uint32_t bwd_credit_addr = 1565632 + 64;

    volatile tt_l1_ptr uint32_t* credit_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_credit_addr);
    uint64_t remote_credit_addr =
        get_noc_addr(receiver_socket.upstream_noc_x, receiver_socket.upstream_noc_y, bwd_credit_addr);

    while (*credit_addr == 0) {
        invalidate_l1_cache();
    }

    socket_packet_header_addr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{remote_credit_addr, 1});
    fabric_connection.wait_for_empty_write_slot();
    fabric_connection.send_payload_flush_blocking_from_address(
        (uint32_t)socket_packet_header_addr, sizeof(PACKET_HEADER_TYPE));

    auto noc_write_addr = output_addr_gen.get_noc_addr(0);
    uint32_t measurement_addr = 1565632;
    for (int i = 0; i < 200; i++) {
        uint64_t start_timestamp = get_timestamp();
        socket_wait_for_pages(receiver_socket, 1);
        uint64_t end_timestamp = get_timestamp();
        // noc_async_write<output_page_size>(receiver_socket.read_ptr, noc_write_addr, output_page_size);
        socket_pop_pages(receiver_socket, 1);
        if (i % num_pages_per_ack == 0) {
            notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr, upstream_bytes_acked_noc_addr);
            ack_sent = true;
        }

        uint64_t latency = end_timestamp - start_timestamp;
        *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(measurement_addr) = latency;
        measurement_addr += sizeof(uint64_t);
    }
    if (!ack_sent) {
        notify_sender(receiver_socket, fabric_connection, socket_packet_header_addr, upstream_bytes_acked_noc_addr);
    }
    update_socket_config(receiver_socket);
    fabric_connection.close();
}
