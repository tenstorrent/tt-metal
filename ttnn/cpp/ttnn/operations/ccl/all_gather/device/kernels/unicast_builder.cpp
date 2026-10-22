// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

#include <cstdint>

#include "pump.hpp"
#include "route.hpp"
#include "slot_io.hpp"

// Store-and-forward builder: fills slots with complete one-hop packets for its pump. Iteration 0 reads this
// device's input; iteration i relays the stripe upstream sent in its iteration i - 1, reading it back from our
// output once data_valid (units delivered by upstream, cumulative) says it has landed. Finally waits for
// everything upstream delivers, which is when our output is complete for this direction.
void kernel_main() {
    using Ct = BuilderCt;
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    static_assert(Ct::C == 1, "unicast sends on one connection");

    uint32_t idx = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t dev = get_arg_val<uint32_t>(idx++);
    const uint32_t stripe_step = get_arg_val<uint32_t>(idx++);
    const uint32_t I = get_arg_val<uint32_t>(idx++);
    const uint32_t e0 = get_arg_val<uint32_t>(idx++);
    const uint32_t ne = get_arg_val<uint32_t>(idx++);
    const uint32_t last_skip = get_arg_val<uint32_t>(idx++);
    const uint32_t last_take = get_arg_val<uint32_t>(idx++);
    const uint32_t iter_units = get_arg_val<uint32_t>(idx++);
    const uint32_t recv_units = get_arg_val<uint32_t>(idx++);
    const uint32_t dv_sem = get_arg_val<uint32_t>(idx++);
    const RouteArgs route = RouteArgs::read(idx);

    // A line endpoint's dead direction sends nothing; it only waits for what arrives.
    if (I > 0) {
        const auto in = TensorAccessor(in_args, in_addr);
        const auto out = TensorAccessor(out_args, out_addr);
        route_slots<false, false>(get_write_ptr(Ct::cb), Ct::slots, Ct::slot_bytes, Ct::hdr_bytes, &route, Ct::C);

        CbSlotRing ring(Ct::cb, Ct::slots, Ct::slot_bytes, Ct::inflight);
        NocIo<PACKET_HEADER_TYPE> io{Ct::C, Ct::hdr_bytes};
        Packetizer<CbSlotRing, NocIo<PACKET_HEADER_TYPE>> pk(ring, io, Ct::A, Ct::P, Ct::payload_off);
        Walk walk(out, WalkGeo{Ct::A, Ct::Uo, Ct::S, Ct::N, Ct::E, Ct::P, Ct::dst_cross, Ct::cross_row});
        InputPieces input(in, Ct::A, Ct::Ui, Ct::E, Ct::src_cross);
        RelayPieces relay{Ct::A, Ct::E};
        auto* dv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dv_sem);

        // Waits until upstream has delivered `target` units. Push our committed slots first: if every device
        // blocked holding finished packets of the previous stripe, a ring would deadlock. With the push, a
        // wait only ever points at the upstream's previous iteration, and iteration 0 never waits.
        auto gate = [&](uint32_t target) {
            invalidate_l1_cache();
            if (*dv < target) {
                ring.drain();
                noc_semaphore_wait_min(dv, target);
            }
        };

        for (uint32_t iter = 0; iter < I; ++iter) {
            const bool last = iter == I - 1;
            const uint32_t skip = last ? last_skip : 0;
            walk.init((dev + iter * stripe_step) % Ct::N, e0, ne, skip, last ? last_take : ne);
            // Units upstream delivered before the stripe we relay now, and our position within it.
            const uint32_t base = iter > 0 ? (iter - 1) * iter_units : 0;
            uint32_t pos = skip * Ct::E;
            DstRun run;
            while (walk.next(run)) {
                pos += run.n * Ct::E;
                if (iter == 0) {
                    pk.add(run, input);
                } else {
                    gate(base + pos);
                    pk.add(run, relay);
                }
            }
            // No packet spans stripes, so the pump can find iteration boundaries by count.
            pk.close();
        }
        ring.drain();
        noc_async_read_set_trid(0);
    }

    sem_wait_consume(dv_sem, recv_units);
}
