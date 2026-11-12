// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "../d2d_send.h"
#include "../persistent.h"
#include <cstdint>

// The persistent op's partials out to the requester, on one fabric sender core per link (see persistent.h for
// the iteration protocol). The op does no cross-chip reduce: every chip sends its partial row straight to the
// requesting chips, which sum the partials of every chip themselves (persistent_moe_request). Link L's
// sender carries the row's L-th part -- its output cores own those columns, and stage their partials here,
// in column order, in the iteration's half of the staging -- to sum core L of every chip of the iteration's
// requester (the control core writes the requester into this core's source ring before the go): into this
// chip's slot of that core's partials buffer, then a ready flag in this chip's flag word after the slots. The
// flag goes on the same route as the slot, after it, so it lands after it. The requester clears its flags
// once it has summed them, before it sends its next request, so a slot is free whenever a row of that
// requester is sent.
//
// The senders are their own stage, a row behind the worker grid: they report ready on FE_SEM_READY (the
// control core's reduce-stage count), which the grid's go for iteration i waits on for iteration i - 2, so
// the staging half a row is sent from is refilled only after it has left.
//
// Compile-time args:
//   0: my_chip (this chip's slot in the requesters' partials buffers)   1: part_bytes (this link's part of a
//   row, its output cores' partials together)   2: sem_ar_ready   3: sem_go   4: cb_scratch (the staging, at
//   the scratch address)   5: num_requesters   6: max_payload (fabric packet bytes)   7: num_slots (chips,
//   the slots of a partials buffer)   8: num_connections (one per direction the partials leave in)
//
// Runtime args: num_cores (this link's output cores), the source ring's address, then per requester: the
// connection its sends leave by, its partials buffer address, this link's sum core's NoC x and y, its number
// of routes and the routes (d2d_send.h); then the fabric connections.

constexpr uint32_t my_chip = get_compile_time_arg_val(0);
constexpr uint32_t part_bytes = get_compile_time_arg_val(1);
constexpr uint32_t sem_ar_ready_id = get_compile_time_arg_val(2);
constexpr uint32_t sem_go_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_scratch_id = get_compile_time_arg_val(4);
constexpr uint32_t num_requesters = get_compile_time_arg_val(5);
constexpr uint32_t max_payload = get_compile_time_arg_val(6);
constexpr uint32_t num_slots = get_compile_time_arg_val(7);
constexpr uint32_t num_connections = get_compile_time_arg_val(8);
constexpr uint32_t kStopBit = 0x80000000u;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    uint32_t connection[num_requesters];
    uint32_t partials_addr[num_requesters];
    uint32_t sum_x[num_requesters];
    uint32_t sum_y[num_requesters];
    uint32_t route_args[num_requesters];
    uint32_t num_routes[num_requesters];
    for (uint32_t k = 0; k < num_requesters; ++k) {
        connection[k] = get_arg_val<uint32_t>(arg_idx++);
        partials_addr[k] = get_arg_val<uint32_t>(arg_idx++);
        sum_x[k] = get_arg_val<uint32_t>(arg_idx++);
        sum_y[k] = get_arg_val<uint32_t>(arg_idx++);
        num_routes[k] = get_arg_val<uint32_t>(arg_idx++);
        route_args[k] = arg_idx;
        arg_idx += num_routes[k] * kD2DRouteArgs;
    }
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_idx);
    auto* header = PacketHeaderPool::allocate_header();
    const uint32_t staging_addr = get_write_ptr(cb_scratch_id);

    Semaphore<> ar_ready(sem_ar_ready_id);
    auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go_id));
    for (uint32_t iter = 0;; ++iter) {
        if (iter > 0) {
            noc_semaphore_inc(get_noc_addr(FE_CTRL_X, FE_CTRL_Y, get_semaphore(FE_SEM_READY)), 1);
            noc_async_atomic_barrier();
        }
        uint32_t v;
        do {
            invalidate_l1_cache();
            v = *go;
        } while ((v & ~kStopBit) < iter + 1u);
        if ((v & kStopBit) != 0u) {
            break;
        }

        {
            DeviceZoneScopedN("FE_RED_PARTIALS_WAIT");
            ar_ready.wait_min(num_cores * (iter + 1u));
        }
        DeviceZoneScopedN("FE_RED_SEND");
        invalidate_l1_cache();
        const uint32_t k = source[iter % kFeSourceRing];
        const uint32_t src = staging_addr + (iter & 1u) * part_bytes;
        const uint32_t slot = partials_addr[k] + my_chip * part_bytes;
        const uint32_t flag = partials_addr[k] + num_slots * part_bytes + my_chip * 4u;
        auto* sender = &fabric_connection.get(connection[k]).sender;
        size_t idx = route_args[k];
        for (uint32_t r = 0; r < num_routes[k]; ++r) {
            const D2DRoute route = read_d2d_route(idx);
            d2d_send<max_payload>(sender, header, route, src, part_bytes, sum_x[k], sum_y[k], slot);
            d2d_send_word(sender, header, route, sum_x[k], sum_y[k], flag, 1u);
        }
        // The staging half is refilled two iterations on, so every packet has to have left it.
        noc_async_writes_flushed();
    }
    noc_async_write_barrier();
    close_connections(fabric_connection);
}
