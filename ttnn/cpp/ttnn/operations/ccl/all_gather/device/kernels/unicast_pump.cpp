// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

#include <cstdint>

#include "pump.hpp"
#include "route.hpp"

// Store-and-forward pump: sends the builder's packets one hop, copies iteration 0 (our own data) into our
// output, and counts delivered units into the downstream builder's data_valid.
void kernel_main() {
    constexpr uint32_t cb = get_named_compile_time_arg_val("cb");
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t slot_bytes = get_named_compile_time_arg_val("slot_bytes");
    constexpr uint32_t payload_off = get_named_compile_time_arg_val("payload_off");
    constexpr uint32_t hdr_bytes = get_named_compile_time_arg_val("hdr_bytes");
    constexpr uint32_t C = get_named_compile_time_arg_val("C");
    constexpr uint32_t A = get_named_compile_time_arg_val("A");
    constexpr uint32_t signal_units = get_named_compile_time_arg_val("signal_units");

    uint32_t idx = 0;
    const uint32_t I = get_arg_val<uint32_t>(idx++);
    const uint32_t iter_units = get_arg_val<uint32_t>(idx++);
    const uint32_t last_units = get_arg_val<uint32_t>(idx++);
    const bool local_copy = get_arg_val<uint32_t>(idx++) != 0;
    const uint32_t init_sem = get_arg_val<uint32_t>(idx++);
    const uint32_t dv_sem = get_arg_val<uint32_t>(idx++);
    const uint32_t self_x = get_arg_val<uint32_t>(idx++);
    const uint32_t self_y = get_arg_val<uint32_t>(idx++);
    const uint32_t partner_x = get_arg_val<uint32_t>(idx++);
    const uint32_t partner_y = get_arg_val<uint32_t>(idx++);
    const RouteArgs route = RouteArgs::read(idx);

    // A dead line direction: no connection was set up and nothing is sent.
    if (I == 0) {
        return;
    }

    tt::tt_fabric::RoutingPlaneConnectionManager conns;
    size_t fabric_idx = idx;
    open_connections(conns, C, fabric_idx);
    IncSender inc;
    inc.init<false>(conns.get(0).sender, route, 0);

    // Tell the downstream core that sends into our output that it may start; then wait to be told the same.
    // The builder keeps filling slots meanwhile, so launch skew hides behind its prefetch.
    inc.send(partner_x, partner_y, init_sem, 1);
    sem_wait_consume(init_sem, 1);

    // data_valid goes to our mirror core downstream, every signal_units units and at each iteration end (the
    // downstream relays iteration by iteration), and whenever the builder makes us wait, so a downstream gate
    // never starves behind an unsent count.
    const uint32_t total = (I - 1) * iter_units + last_units;
    uint32_t sent = 0;
    uint32_t pending = 0;
    uint32_t iter_end = iter_units < total ? iter_units : total;
    auto signal = [&]() {
        if (pending != 0) {
            inc.send(self_x, self_y, dv_sem, pending);
            pending = 0;
        }
    };

    Pump<cb, slots, slot_bytes, payload_off, hdr_bytes, C, A> pump(conns);
    pump.run(
        total,
        [&](uint32_t slot, uint32_t units) {
            if (local_copy && sent < iter_units) {
                pump.local_copy(slot);
            }
            sent += units;
            pending += units;
            if (sent == iter_end) {
                signal();
                iter_end = iter_end + iter_units < total ? iter_end + iter_units : total;
            } else if (pending >= signal_units) {
                signal();
            }
        },
        signal);
    signal();
    pump.finish();
    close_connections(conns);
    noc_async_write_barrier();
}
