// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

#include "pump.hpp"

// Multicast pump: sends every slot on all C connections (the builder copies it into our own output).
//
// Two semaphores, so a peer's next-invocation init can never pass for this invocation's completion:
//   init: every peer's output is allocated. Sent once on route variant 0 of each connection; the variant-0
//         routes cover every other device exactly once, so we wait for init_expect = N - 1.
//   done: every peer's data has landed here. Sent after the data on every (connection, variant): each route
//         that carried data carries its own flushed inc behind it, so we wait for V * (N - 1).
void kernel_main() {
    constexpr uint32_t C = SlotCt::C;
    constexpr bool alternate = get_named_compile_time_arg_val("alternate") != 0;
    constexpr uint32_t variants = alternate ? 2 : 1;

    uint32_t idx = 0;
    const uint32_t total_units = get_arg_val<uint32_t>(idx++);
    const uint32_t init_sem = get_arg_val<uint32_t>(idx++);
    const uint32_t done_sem = get_arg_val<uint32_t>(idx++);
    const uint32_t init_expect = get_arg_val<uint32_t>(idx++);
    const uint32_t self_x = get_arg_val<uint32_t>(idx++);
    const uint32_t self_y = get_arg_val<uint32_t>(idx++);
    RouteArgs routes[C];
    for (uint32_t c = 0; c < C; ++c) {
        routes[c] = RouteArgs::read(idx);
    }

    tt::tt_fabric::RoutingPlaneConnectionManager conns;
    size_t fabric_idx = idx;
    open_connections(conns, C, fabric_idx);
    IncSender inc[C][variants];
    for (uint32_t c = 0; c < C; ++c) {
        for (uint32_t v = 0; v < variants; ++v) {
            inc[c][v].init<true>(conns.get(c).sender, routes[c], v);
        }
    }

    // Every device uses the same core and semaphore addresses, so our own coordinates name the target.
    for (uint32_t c = 0; c < C; ++c) {
        inc[c][0].send(self_x, self_y, init_sem, 1);
    }
    sem_wait_consume(init_sem, init_expect);

    Pump<SlotCt> pump(conns);
    pump.run(total_units, [](uint32_t, uint32_t) {}, []() {});
    pump.finish();

    for (uint32_t c = 0; c < C; ++c) {
        for (uint32_t v = 0; v < variants; ++v) {
            inc[c][v].send(self_x, self_y, done_sem, 1);
        }
    }
    sem_wait_consume(done_sem, variants * init_expect);

    close_connections(conns);
    noc_async_write_barrier();
}
