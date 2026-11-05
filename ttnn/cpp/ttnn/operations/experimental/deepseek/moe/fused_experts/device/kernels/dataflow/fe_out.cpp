// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/socket_api.h"
#include "../d2d_send.h"
#include "../persistent.h"

// The persistent op's output (RISCV_1 of the root chip's control core, NoC 1). Iteration i's
// num_out_cores reduced output shards land in half i & 1 of the staging, each bumping sem_out_count;
// once all of them have, the row goes out as one page of the output D2D socket of the iteration's
// requester -- the control core's source ring names it -- and sem_out_done counts it, which is what
// lets the control core reuse that half two iterations later. The control core sets sem_out_stop to
// kStopBit | (pages to push) at the stop.
//
// The page and then bytes_sent go to the socket's receivers over its routes (d2d_send.h), all through
// the one fabric connection.
//
// Compile-time args:
//   0: num_requesters   1: row_bytes (one socket page)   2: num_out_cores   3: sem_out_count
//   4: sem_out_done   5: sem_out_stop   6: max_payload (fabric packet bytes)   7: cb_ctrl
//   8: the source ring's offset in cb_ctrl
//
// Runtime args:
//   0: staging address (two rows), then per requester its output socket config address, its number of
//   routes and the routes, then the fabric connection

constexpr uint32_t kStopBit = 0x80000000u;

void kernel_main() {
    constexpr uint32_t num_requesters = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_out_cores = get_compile_time_arg_val(2);
    constexpr uint32_t sem_out_count_id = get_compile_time_arg_val(3);
    constexpr uint32_t sem_out_done_id = get_compile_time_arg_val(4);
    constexpr uint32_t sem_out_stop_id = get_compile_time_arg_val(5);
    constexpr uint32_t max_payload = get_compile_time_arg_val(6);
    constexpr uint32_t cb_ctrl_id = get_compile_time_arg_val(7);
    constexpr uint32_t source_ring_offset = get_compile_time_arg_val(8);

    size_t arg_idx = 0;
    const uint32_t staging_addr = get_arg_val<uint32_t>(arg_idx++);
    SocketSenderInterface out[num_requesters];
    // Where each requester's routes start in the runtime args, and how many there are.
    uint32_t route_args[num_requesters];
    uint32_t num_routes[num_requesters];
    for (uint32_t k = 0; k < num_requesters; ++k) {
        out[k] = create_sender_socket_interface(get_arg_val<uint32_t>(arg_idx++));
        set_sender_socket_page_size(out[k], row_bytes);
        num_routes[k] = get_arg_val<uint32_t>(arg_idx++);
        route_args[k] = arg_idx;
        arg_idx += num_routes[k] * kD2DRouteArgs;
    }
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, 1, arg_idx);
    auto* sender = &fabric_connection.get(0).sender;
    auto* header = PacketHeaderPool::allocate_header();

    auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctrl_id) + source_ring_offset);
    auto* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_out_count_id));
    auto* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_out_done_id));
    auto* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_out_stop_id));

    for (uint32_t page = 0;; ++page) {
        // The stop comes only once every page's shards are in, so a count that is still short at the
        // stop means there is no such page.
        bool stopped = false;
        {
            DeviceZoneScopedN("FE_OUT_SHARDS_WAIT");
            while (true) {
                invalidate_l1_cache();
                if (*count >= num_out_cores * (page + 1u)) {
                    break;
                }
                const uint32_t s = *stop;
                if ((s & kStopBit) != 0u && (s & ~kStopBit) == page) {
                    stopped = true;
                    break;
                }
            }
        }
        if (stopped) {
            break;
        }

        DeviceZoneScopedN("FE_OUT_PUSH");
        const uint32_t k = source[page % kFeSourceRing];
        SocketSenderInterface& socket = out[k];
        socket_reserve_pages(socket, 1);
        const auto receiver = get_downstream_encoding(socket, 0);
        const uint32_t rx_x = receiver.d2d.downstream_noc_x;
        const uint32_t rx_y = receiver.d2d.downstream_noc_y;
        const uint32_t src = staging_addr + (page & 1u) * row_bytes;
        const uint32_t dst = socket.downstream_fifo_addr + socket.write_ptr;
        size_t idx = route_args[k];
        for (uint32_t r = 0; r < num_routes[k]; ++r) {
            d2d_send<max_payload>(sender, header, read_d2d_route(idx), src, row_bytes, rx_x, rx_y, dst);
        }
        socket_push_pages(socket, 1);
        idx = route_args[k];
        for (uint32_t r = 0; r < num_routes[k]; ++r) {
            d2d_send_word(
                sender, header, read_d2d_route(idx), rx_x, rx_y, socket.downstream_bytes_sent_addr, socket.bytes_sent);
        }
        // The staging half is refilled two pages on, so the row has to have left it.
        noc_async_writes_flushed();
        *done = page + 1u;
    }

    for (uint32_t k = 0; k < num_requesters; ++k) {
        update_socket_config(out[k]);
    }
    noc_async_write_barrier();
    close_connections(fabric_connection);
}
