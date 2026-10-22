// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

#include <cstdint>

#include "slot_io.hpp"

// Multicast builder: one pass over this device's slice of the input, into slots whose C headers each carry one
// connection's route. The fabric does the replication, so this is all the reading the op does. Each landed slot
// is also written into our own output here, on our NOC, so the pump only sends.
void kernel_main() {
    using Ct = BuilderCt;
    constexpr bool alternate = get_named_compile_time_arg_val("alternate") != 0;
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    static_assert(Ct::C >= 1 && Ct::C <= 4, "one header per fabric connection");

    uint32_t idx = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t dev = get_arg_val<uint32_t>(idx++);
    const uint32_t e0 = get_arg_val<uint32_t>(idx++);
    const uint32_t ne = get_arg_val<uint32_t>(idx++);
    RouteArgs routes[Ct::C];
    for (uint32_t c = 0; c < Ct::C; ++c) {
        routes[c] = RouteArgs::read(idx);
    }

    const auto in = TensorAccessor(in_args, in_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    route_slots<true, alternate>(
        get_write_ptr(Ct::cb), Ct::slots, Ct::slot_bytes, Ct::hdr_bytes, Ct::payload_off, routes, Ct::C);

    SlotRing ring(/*local_copy=*/true);
    SlotIo io;
    SlotPacketizer pk(ring, io, Ct::P, Ct::payload_off);
    Walk walk(out, WalkGeo{Ct::A, Ct::Uo, Ct::S, Ct::N, Ct::E, Ct::P, Ct::dst_cross});
    InputPieces input(in, Ct::A, Ct::Ui, Ct::E, Ct::src_cross);

    walk.init(dev, e0, ne, 0, ne);
    pk.add_all(walk, input, [](const DstRun&) {});
    pk.close();
    ring.drain();
    io.finish();
    noc_async_write_barrier();
}
