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
// chip's slot of that core's partials buffer, then +1 on this chip's flag word after the slots (0 -> 1, the ready
// flag), fused into the slot's last packet after its write, so it counts only once the slot has
// landed. Every sub-part's packets go out back to back (a ring of headers). The requester clears its flags
// once it has summed them, before it sends its next request, so a slot is free whenever a row of that
// requester is sent.
//
// The senders are their own stage, a row behind the worker grid: they report ready on FE_SEM_READY (the
// control core's reduce-stage count), which the grid's go for iteration i waits on for iteration i - 2, so
// the staging half a row is sent from is refilled only after it has left.
//
// The link-0 sender also carries the sharded router's candidate exchange (router_reduce.cpp, num_exchange_routes >
// 0): the reducer leaves this chip's candidates of each iteration in its slot of the candidate slots here and bumps
// the ready count; whenever this kernel waits, it forwards every candidate slot it has not yet forwarded, in order,
// along the whole-mesh branches to the same slot of every other chip's route merge core, each slot one fused
// write + increment of that core's sem_router per branch, then bumps the reducer's forwarded count. The router stage
// runs a row ahead of the grid and this stage a row behind it, so it is idle on its own work when candidates come.
//
// Compile-time args:
//   0: my_chip (this chip's slot in the requesters' partials buffers)   1: part_bytes (this link's part of a
//   row, its output cores' partials together; sub_parts sum cores' worth of columns, each its own packet)   2:
//   sem_ar_ready   3: sem_go   4: cb_scratch (the staging, at the scratch address)   5: num_requesters   6: max_payload
//   (fabric packet bytes)   7: num_slots (chips, the slots of a partials buffer)   8: cand_slot_bytes   9: sem_router
//   (on the route merge) 10: sub_parts (the requester's sum cores per link: this link's part of a row splits evenly
//   over them)
//
// Runtime args: num_cores (this link's output cores), the source ring's address, then per requester: the
// connection its sends leave by, its partials buffer address, this link's sub_parts sum cores' NoC xs, then ys,
// its number
// of routes and the routes (d2d_send.h); then num_exchange_routes (0 off link 0) and, with an exchange, the
// candidate slots address, the route merge's and the reducer's NoC x and y, the ready count address (on this
// core), the reducer's forwarded count address, and per exchange route its connection and the route; then
// num_connections (one per direction the partials or the candidates leave in) and the fabric connections.

constexpr uint32_t my_chip = get_compile_time_arg_val(0);
constexpr uint32_t part_bytes = get_compile_time_arg_val(1);
constexpr uint32_t sem_ar_ready_id = get_compile_time_arg_val(2);
constexpr uint32_t sem_go_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_scratch_id = get_compile_time_arg_val(4);
constexpr uint32_t num_requesters = get_compile_time_arg_val(5);
constexpr uint32_t max_payload = get_compile_time_arg_val(6);
constexpr uint32_t num_slots = get_compile_time_arg_val(7);
constexpr uint32_t cand_slot_bytes = get_compile_time_arg_val(8);
constexpr uint32_t sem_router_id = get_compile_time_arg_val(9);
constexpr uint32_t sub_parts = get_compile_time_arg_val(10);
constexpr uint32_t sub_bytes = part_bytes / sub_parts;
constexpr uint32_t kMaxExchangeRoutes = 4;
// The partials' header ring: with the exchange's kMaxExchangeRoutes, within a RISC's 12 pool headers.
constexpr uint32_t kSendHeaders = 6;
constexpr uint32_t kStopBit = 0x80000000u;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    auto* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    uint32_t connection[num_requesters];
    uint32_t partials_addr[num_requesters];
    uint32_t sum_x[num_requesters][sub_parts];
    uint32_t sum_y[num_requesters][sub_parts];
    uint32_t route_args[num_requesters];
    uint32_t num_routes[num_requesters];
    for (uint32_t k = 0; k < num_requesters; ++k) {
        connection[k] = get_arg_val<uint32_t>(arg_idx++);
        partials_addr[k] = get_arg_val<uint32_t>(arg_idx++);
        for (uint32_t p = 0; p < sub_parts; ++p) {
            sum_x[k][p] = get_arg_val<uint32_t>(arg_idx++);
        }
        for (uint32_t p = 0; p < sub_parts; ++p) {
            sum_y[k][p] = get_arg_val<uint32_t>(arg_idx++);
        }
        num_routes[k] = get_arg_val<uint32_t>(arg_idx++);
        route_args[k] = arg_idx;
        arg_idx += num_routes[k] * kD2DRouteArgs;
    }
    uint32_t cand_addr = 0;
    uint32_t merge_x = 0;
    uint32_t merge_y = 0;
    uint32_t reducer_x = 0;
    uint32_t reducer_y = 0;
    volatile tt_l1_ptr uint32_t* cand_ready = nullptr;
    uint32_t forwarded_addr = 0;
    const uint32_t num_exchange_routes = get_arg_val<uint32_t>(arg_idx++);
    uint32_t exchange_connection[kMaxExchangeRoutes];
    D2DRoute exchange_route[kMaxExchangeRoutes];
    if (num_exchange_routes > 0u) {
        cand_addr = get_arg_val<uint32_t>(arg_idx++);
        merge_x = get_arg_val<uint32_t>(arg_idx++);
        merge_y = get_arg_val<uint32_t>(arg_idx++);
        reducer_x = get_arg_val<uint32_t>(arg_idx++);
        reducer_y = get_arg_val<uint32_t>(arg_idx++);
        cand_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
        forwarded_addr = get_arg_val<uint32_t>(arg_idx++);
        for (uint32_t r = 0; r < num_exchange_routes; ++r) {
            exchange_connection[r] = get_arg_val<uint32_t>(arg_idx++);
            exchange_route[r] = read_d2d_route(arg_idx);
        }
    }
    const uint32_t num_connections = get_arg_val<uint32_t>(arg_idx++);
    tt::tt_fabric::RoutingPlaneConnectionManager fabric_connection;
    open_connections(fabric_connection, num_connections, arg_idx);
    // The partials' packets: each sub-part is one fused write + flag increment per route, issued back to back.
    D2DHeaderRing<kSendHeaders> send_headers;
    send_headers.init();
    const uint32_t staging_addr = get_write_ptr(cb_scratch_id);

    // The candidate exchange: one header per route, built once (route, sem_router increment, payload size), so a
    // forward is one fused write + increment packet per route, every route's in flight at once. The merge core's
    // router issues the increment after the write, in order (no flush: with one per chip arriving at each merge core,
    // flushes held its router for each write's ack and serialized them), so the slot has landed when sem_router
    // counts it. Only the slot
    // address (the iteration's half) changes from one forward to the next.
    namespace fabric_api = tt::tt_fabric::mesh::experimental;
    using tt::tt_fabric::common::experimental::UnicastFusedAtomicIncUpdateMask;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* exchange_header[kMaxExchangeRoutes];
    for (uint32_t r = 0; r < num_exchange_routes; ++r) {
        const D2DRoute& route = exchange_route[r];
        exchange_header[r] = PacketHeaderPool::allocate_header();
        fabric_api::fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state<UnicastFusedAtomicIncUpdateMask::All>(
            exchange_header[r],
            route.dst_chip,
            route.dst_mesh,
            route.range,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                safe_get_noc_addr(merge_x, merge_y, cand_addr, 0),
                safe_get_noc_addr(merge_x, merge_y, get_semaphore(sem_router_id), 0),
                1,
                false},
            cand_slot_bytes);
    }

    // Forwards every candidate slot the reducer has left here and this kernel has not yet forwarded.
    uint32_t num_forwarded = 0;
    auto pump = [&]() {
        if (num_exchange_routes > 0u) {
            invalidate_l1_cache();
            const uint32_t ready = *cand_ready;
            while (num_forwarded < ready) {
                DeviceZoneScopedN("FE_RED_EXCHANGE");
                const uint32_t slot = cand_addr + ((num_forwarded & 1u) * num_slots + my_chip) * cand_slot_bytes;
                // The headers are rewritten in place, so the previous forward's (and the partials') must have left.
                noc_async_writes_flushed();
                for (uint32_t r = 0; r < num_exchange_routes; ++r) {
                    fabric_api::fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state<
                        UnicastFusedAtomicIncUpdateMask::WriteDstAddr>(
                        &fabric_connection.get(exchange_connection[r]).sender,
                        exchange_header[r],
                        slot,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            safe_get_noc_addr(merge_x, merge_y, slot, 0), 0, 0, false});
                }
                // The slot's bytes have left before the reducer may refill it (two forwards on).
                noc_async_writes_flushed();
                ++num_forwarded;
                noc_semaphore_inc(get_noc_addr(reducer_x, reducer_y, forwarded_addr), 1);
                noc_async_atomic_barrier();
            }
        }
    };

    auto* ar_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ar_ready_id));
    auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go_id));
    uint32_t iter = 0;
    for (;; ++iter) {
        if (iter > 0) {
            noc_semaphore_inc(get_noc_addr(FE_CTRL_X, FE_CTRL_Y, get_semaphore(FE_SEM_READY)), 1);
            noc_async_atomic_barrier();
        }
        uint32_t v;
        do {
            pump();
            invalidate_l1_cache();
            v = *go;
        } while ((v & ~kStopBit) < iter + 1u);
        if ((v & kStopBit) != 0u) {
            break;
        }

        {
            DeviceZoneScopedN("FE_RED_PARTIALS_WAIT");
            while (true) {
                pump();
                invalidate_l1_cache();
                if (*ar_ready >= num_cores * (iter + 1u)) {
                    break;
                }
            }
        }
        DeviceZoneScopedN("FE_RED_SEND");
        invalidate_l1_cache();
        const uint32_t k = source[iter % kFeSourceRing];
        const uint32_t src = staging_addr + (iter & 1u) * part_bytes;
        // Every sum core of the requester holds its own columns' slots, at the same address.
        const uint32_t slot = partials_addr[k] + my_chip * sub_bytes;
        const uint32_t flag = partials_addr[k] + num_slots * sub_bytes + my_chip * 4u;
        auto* sender = &fabric_connection.get(connection[k]).sender;
        size_t idx = route_args[k];
        send_headers.reset();
        for (uint32_t r = 0; r < num_routes[k]; ++r) {
            const D2DRoute route = read_d2d_route(idx);
            for (uint32_t p = 0; p < sub_parts; ++p) {
                d2d_send_flagged<max_payload>(
                    sender, send_headers, route, src + p * sub_bytes, sub_bytes, sum_x[k][p], sum_y[k][p], slot, flag);
            }
        }
        // The staging half is refilled two iterations on, so every packet has to have left it.
        noc_async_writes_flushed();
    }
    // The router stage ran every iteration a row ahead, so its last candidates may still be waiting here.
    if (num_exchange_routes > 0u) {
        while (num_forwarded < iter) {
            pump();
        }
        *cand_ready = 0;
    }
    noc_async_write_barrier();
    close_connections(fabric_connection);
}
