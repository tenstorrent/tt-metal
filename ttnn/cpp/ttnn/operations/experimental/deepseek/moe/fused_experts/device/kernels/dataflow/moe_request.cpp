// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/socket_api.h"
#include "api/tensor/tensor_accessor.h"
#include "../d2d_send.h"

// persistent_moe_request (see PersistentMoeRequestParams): one request to the persistent fused_experts
// op from this chip. The page -- a 64-byte header, the layer id in word 0, then the input row -- is
// built in cb_page and sent to the request socket's receivers (d2d_send.h); the last packet of each send bumps
// their bytes_sent. The op's chips send their partials of the row straight to this chip's sum cores
// (moe_partials_sum.cpp), which write the output. The receivers do not ack the request: the op takes a
// requester's next page only after this one's partials, which this op waits for, so the request socket's acks
// are set here.
//
// The page may go over two fabric links: the socket's sender core (the main one) sends bytes [0, send_bytes)
// over link 0 and a partner core the rest over link 1. The partner takes the page's FIFO address, the
// receivers' bytes_sent address and their core from the main core (its semaphores 0, 1, 2), and bumps
// bytes_sent by its own bytes; the main core bumps it by the rest of what the socket push added. The page counts
// at the receivers only once both halves have landed.
//
// A one-row request to an op built for blocks of block_rows rows (the socket's page a block's): its row goes into
// row 0 of the block's (rows x 32) tiles -- each 32 B face row of the input every 32 * block_rows B of the page,
// read straight from DRAM -- with kFeHdrRows (header word 9) set to 1; the page's other rows are never used. The main
// core sends the whole page (no partner).
//
// Compile-time args:
//   0: request socket config address   1: row_bytes (the input's)   2: max_payload (fabric packet bytes)   3: cb_page
//   4: page_row_bytes (the socket page's row: row_bytes, or a block's)   5: block_rows (page_row_bytes / row_bytes)
//   6+: TensorAccessorArgs(input)
//
// Runtime args:
//   0: layer   1: input address   2: send_start (0 on the main core)   3: send_bytes   4: the partner's NoC x
//   (kNoPartner: none; main core only)   5: its NoC y   6: num_routes   7+: the request's routes (d2d_send.h),
//   then the fabric connection they leave by

constexpr uint32_t kNoPartner = 0xFFFFFFFFu;

void kernel_main() {
    constexpr uint32_t request_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t max_payload = get_compile_time_arg_val(2);
    constexpr uint32_t cb_page = get_compile_time_arg_val(3);
    constexpr uint32_t page_row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t block_rows = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();
    constexpr uint32_t kHdrBytes = 64;
    constexpr uint32_t page_bytes = kHdrBytes + page_row_bytes;
    // persistent.h kFeHdrRows.
    constexpr uint32_t kHdrRowsWord = 9;
    constexpr uint32_t kFaceRowBytes = 32;
    constexpr uint32_t kMaxRoutes = 16;
    constexpr uint32_t kSendHeaders = 8;  // of the RISC's 12 in the pool

    size_t arg_idx = 0;
    const uint32_t layer = get_arg_val<uint32_t>(arg_idx++);
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(arg_idx++));
    const uint32_t send_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t send_bytes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t partner_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t partner_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_routes = get_arg_val<uint32_t>(arg_idx++);
    D2DRoute routes[kMaxRoutes];
    for (uint32_t r = 0; r < num_routes; ++r) {
        routes[r] = read_d2d_route(arg_idx);
    }
    tt::tt_fabric::RoutingPlaneConnectionManager request_connection;
    open_connections(request_connection, 1, arg_idx);
    // The page's packets go out back to back, each with its own header.
    D2DHeaderRing<kSendHeaders> headers;
    headers.init();
    auto* sender = &request_connection.get(0).sender;
    const uint32_t page_l1 = get_write_ptr(cb_page);

    if (send_start > 0u) {
        // The partner: its part of the row, once the main core has the page's place.
        noc_async_read(input.get_noc_addr(0) + (send_start - kHdrBytes), page_l1 + send_start, send_bytes);
        auto* place = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        auto* sent_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
        auto* rx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
        noc_async_read_barrier();
        DeviceZoneScopedN("MOE_REQ_SEND");
        do {
            invalidate_l1_cache();
        } while (*place == 0u || *sent_addr == 0u || *rx == 0u);
        headers.reset();
        for (uint32_t r = 0; r < num_routes; ++r) {
            d2d_send_flagged<max_payload>(
                sender,
                headers,
                routes[r],
                page_l1 + send_start,
                send_bytes,
                *rx >> 16,
                *rx & 0xFFFFu,
                *place + send_start,
                *sent_addr,
                send_bytes);
        }
        noc_async_write_barrier();
        close_connections(request_connection);
        return;
    }

    auto* header = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_l1);
    for (uint32_t w = 0; w < kHdrBytes / 4u; ++w) {
        header[w] = 0;
    }
    header[0] = layer;
    if constexpr (block_rows > 1u) {
        header[kHdrRowsWord] = 1u;
        const uint64_t src = input.get_noc_addr(0);
        for (uint32_t c = 0; c < row_bytes / kFaceRowBytes; ++c) {
            noc_async_read(
                src + c * kFaceRowBytes, page_l1 + kHdrBytes + c * block_rows * kFaceRowBytes, kFaceRowBytes);
        }
    } else {
        noc_async_read(input.get_noc_addr(0), page_l1 + kHdrBytes, send_bytes - kHdrBytes);
    }
    noc_async_read_barrier();

    SocketSenderInterface request = create_sender_socket_interface(request_config_addr);
    set_sender_socket_page_size(request, page_bytes);
    {
        DeviceZoneScopedN("MOE_REQ_SEND");
        socket_reserve_pages(request, 1);
        const auto receiver = get_downstream_encoding(request, 0);
        const uint32_t rx_x = receiver.d2d.downstream_noc_x;
        const uint32_t rx_y = receiver.d2d.downstream_noc_y;
        const uint32_t dst = request.downstream_fifo_addr + request.write_ptr;
        if (partner_x != kNoPartner) {
            noc_inline_dw_write(get_noc_addr(partner_x, partner_y, get_semaphore(0)), dst);
            noc_inline_dw_write(
                get_noc_addr(partner_x, partner_y, get_semaphore(1)), request.downstream_bytes_sent_addr);
            noc_inline_dw_write(get_noc_addr(partner_x, partner_y, get_semaphore(2)), (rx_x << 16) | rx_y);
        }
        const uint32_t sent_before = request.bytes_sent;
        socket_push_pages(request, 1);
        // The receivers' bytes_sent goes up by what the push added, fused into each send's last packet: the
        // partner's bytes from it, the rest (with any wrap padding) from here.
        const uint32_t pushed = request.bytes_sent - sent_before;
        const uint32_t mine = pushed - (page_bytes - send_bytes);
        headers.reset();
        for (uint32_t r = 0; r < num_routes; ++r) {
            d2d_send_flagged<max_payload>(
                sender,
                headers,
                routes[r],
                page_l1,
                send_bytes,
                rx_x,
                rx_y,
                dst,
                request.downstream_bytes_sent_addr,
                mine);
        }
    }
    // The next request's page goes out only after this op, whose sum cores wait for this page's partials.
    for (uint32_t i = 0; i < request.num_downstreams; ++i) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(request.bytes_acked_base_addr + i * bytes_acked_size_bytes) =
            request.bytes_sent;
    }

    update_socket_config(request);
    noc_async_write_barrier();
    close_connections(request_connection);
}
