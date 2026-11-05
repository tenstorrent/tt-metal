// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "../d2d_send.h"

namespace fabric_api = tt::tt_fabric::mesh::experimental;

// persistent_moe_request (see PersistentMoeRequestParams): one request to the persistent fused_experts
// op from this chip. The page -- a 64-byte header, the layer id in word 0, then the input row -- is
// built in cb_page and sent to the request socket's receivers (d2d_send.h), then bytes_sent. Unless the layer is the
// stop, the reduced row then comes back on the response socket into the output, the response is acked to its sender,
// and -- the receivers do not ack the request, the response being proof they are all past it -- the request socket's
// acks are set here.
//
// Compile-time args:
//   0: request socket config address   1: response socket config address   2: row_bytes
//   3: max_payload (fabric packet bytes)   4: cb_page   5: stop layer id
//   6+: TensorAccessorArgs(input), then TensorAccessorArgs(output)
//
// Runtime args:
//   0: layer   1: input address   2: output address   3: num_routes   4+: the request's routes
//   (d2d_send.h), then the fabric connection they leave by (link 0), then the one to the response's
//   sender (link 1)

void kernel_main() {
    constexpr uint32_t request_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t response_config_addr = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t max_payload = get_compile_time_arg_val(3);
    constexpr uint32_t cb_page = get_compile_time_arg_val(4);
    constexpr uint32_t stop_layer = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t kHdrBytes = 64;
    constexpr uint32_t page_bytes = kHdrBytes + row_bytes;
    constexpr uint32_t kMaxRoutes = 16;

    size_t arg_idx = 0;
    const uint32_t layer = get_arg_val<uint32_t>(arg_idx++);
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(arg_idx++));
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(arg_idx++));
    const uint32_t num_routes = get_arg_val<uint32_t>(arg_idx++);
    D2DRoute routes[kMaxRoutes];
    for (uint32_t r = 0; r < num_routes; ++r) {
        routes[r] = read_d2d_route(arg_idx);
    }
    tt::tt_fabric::RoutingPlaneConnectionManager request_connection;
    open_connections(request_connection, 1, arg_idx);
    tt::tt_fabric::RoutingPlaneConnectionManager ack_connection;
    open_connections(ack_connection, 1, arg_idx);
    auto* request_header = PacketHeaderPool::allocate_header();
    const uint8_t ack_route = PacketHeaderPool::allocate_header_n(1);

    const uint32_t page_l1 = get_write_ptr(cb_page);
    auto* header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_l1);
    for (uint32_t w = 0; w < kHdrBytes / 4u; ++w) {
        header[w] = 0;
    }
    header[0] = layer;
    noc_async_read(input.get_noc_addr(0), page_l1 + kHdrBytes, row_bytes);
    noc_async_read_barrier();

    SocketSenderInterface request = create_sender_socket_interface(request_config_addr);
    set_sender_socket_page_size(request, page_bytes);
    {
        DeviceZoneScopedN("MOE_REQ_SEND");
        socket_reserve_pages(request, 1);
        const auto receiver = get_downstream_encoding(request, 0);
        const uint32_t rx_x = receiver.d2d.downstream_noc_x;
        const uint32_t rx_y = receiver.d2d.downstream_noc_y;
        auto* sender = &request_connection.get(0).sender;
        const uint32_t dst = request.downstream_fifo_addr + request.write_ptr;
        for (uint32_t r = 0; r < num_routes; ++r) {
            d2d_send<max_payload>(sender, request_header, routes[r], page_l1, page_bytes, rx_x, rx_y, dst);
        }
        socket_push_pages(request, 1);
        for (uint32_t r = 0; r < num_routes; ++r) {
            d2d_send_word(
                sender, request_header, routes[r], rx_x, rx_y, request.downstream_bytes_sent_addr, request.bytes_sent);
        }
    }

    if (layer != stop_layer) {
        SocketReceiverInterface response = create_receiver_socket_interface(response_config_addr);
        set_receiver_socket_page_size(response, row_bytes);
        {
            DeviceZoneScopedN("MOE_REQ_RESPONSE_WAIT");
            socket_wait_for_pages(response, 1);
        }
        DeviceZoneScopedN("MOE_REQ_RESPONSE");
        noc_async_write(response.read_ptr, output.get_noc_addr(0), row_bytes);
        noc_async_write_barrier();
        socket_pop_pages(response, 1);
        fabric_api::fabric_unicast_noc_unicast_inline_write(
            ack_connection,
            ack_route,
            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                safe_get_noc_addr(
                    response.d2d.upstream_noc_x,
                    response.d2d.upstream_noc_y,
                    response.d2d.upstream_bytes_acked_addr,
                    0),
                response.bytes_acked});
        for (uint32_t i = 0; i < request.num_downstreams; ++i) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                request.bytes_acked_base_addr + i * bytes_acked_size_bytes) = request.bytes_sent;
        }
        update_socket_config(response);
    }

    update_socket_config(request);
    noc_async_write_barrier();
    close_connections(request_connection);
    close_connections(ack_connection);
}
