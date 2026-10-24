// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// all_gather reader, shared by unicast and multicast: the slot producer. It never touches the fabric, and it starts at
// once: the init handshake gates only the writer's remote sends, so the first slots and the local copy overlap it.
//
// Per slot: [relay: wait for the units this slot may read, holding no slot] -> reserve -> build up to K packets
// (records + payload reads) -> one read barrier -> [local copy, flushed] -> SlotInfo -> push.
// Iteration 0 reads this device's stripe from the input; iteration i >= 1 (unicast) re-reads the stripe upstream
// relayed into our output in its iteration i - 1. See ag_sync.hpp for the sync protocol and the address rule.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

#include "ag_walk.hpp"
#include "ag_packet.hpp"
#include "ag_sync.hpp"

void kernel_main() {
    constexpr uint32_t A = get_named_compile_time_arg_val("A");
    constexpr uint32_t P_A = get_named_compile_time_arg_val("P_A");
    constexpr uint32_t CAP = get_named_compile_time_arg_val("CAP");
    constexpr uint32_t K = get_named_compile_time_arg_val("K");
    constexpr uint32_t SLOT_BYTES = get_named_compile_time_arg_val("SLOT_BYTES");
    constexpr uint32_t CB_ID = get_named_compile_time_arg_val("CB_ID");
    constexpr uint32_t UI = get_named_compile_time_arg_val("UI");
    constexpr uint32_t UO = get_named_compile_time_arg_val("UO");
    constexpr uint32_t S = get_named_compile_time_arg_val("S");
    constexpr uint32_t N = get_named_compile_time_arg_val("N");
    constexpr uint32_t E = get_named_compile_time_arg_val("E");
    constexpr bool DST_CROSS = get_named_compile_time_arg_val("DST_CROSS") != 0;
    constexpr bool SRC_CROSS = get_named_compile_time_arg_val("SRC_CROSS") != 0;
    constexpr uint32_t RDELAY = get_named_compile_time_arg_val("RDELAY");
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    static_assert(CAP % A == 0 && CAP >= A && CAP <= P_A, "the packet cap is a multiple of A within the payload");
    constexpr ag::SlotLayout lay = ag::slot_offsets(K, A, sizeof(PACKET_HEADER_TYPE), CAP);
    static_assert(lay.slot_bytes == SLOT_BYTES, "host and kernel disagree on the packet header size");

    size_t arg_idx = 0;
    const uint32_t in_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t e0 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ne = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t iters = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stripe0 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stripe_step = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t last_skip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t last_take = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t iter_units = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t end_wait = get_arg_val<uint32_t>(arg_idx++);
    const bool local_copy = get_arg_val<uint32_t>(arg_idx++) != 0;
    const uint32_t data_sem = get_arg_val<uint32_t>(arg_idx++);

    const auto in = TensorAccessor(in_args, in_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    Noc noc;
    CircularBuffer cb(CB_ID);

    constexpr ag::WalkGeo geo{.A = A, .Uo = UO, .S = S, .N = N, .E = E, .P_A = P_A, .dst_cross = DST_CROSS};
    ag::Walk<decltype(out), ag::FabricFrameAddr> walk(out, geo);
    ag::InputPieces<decltype(in)> input_pieces(in, A, UI, E, SRC_CROSS);
    ag::RelayPieces relay_pieces(A, E);
    ag::PacketBuilder<decltype(walk), decltype(input_pieces)> input_builder(walk, input_pieces, CAP);
    ag::PacketBuilder<decltype(walk), decltype(relay_pieces)> relay_builder(walk, relay_pieces, CAP);

    auto read = [&](uint64_t src, uint32_t dst, uint32_t bytes) {
        noc.async_read<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            PrecomposedUnicastEndpoint{}, CoreLocalMem<uint32_t>(dst), bytes, {.noc_addr = src}, {});
    };
    auto* data_valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_sem);
    // Own-NOC segment starts of a slot's packet, for the local copy (records hold fabric-frame addresses).
    auto local = [&](uint32_t slot, uint32_t j) { return reinterpret_cast<uint64_t*>(lay.local(slot, j)); };
    for (uint32_t i = 0; i < iters; ++i) {
        const bool last = i + 1 == iters;
        const uint32_t skip = last ? last_skip : 0;
        const uint32_t take = last ? last_take : ne;
        const uint32_t take_units = take * E;
        const bool relay = i > 0;
        const bool copy = i == 0 && local_copy;
        // Upstream sent its iteration i - 1 in walk order and signals cumulative units, so the first `skip`
        // elements of that pass precede ours.
        const uint32_t relay_base = relay ? (i - 1) * iter_units + skip * E : 0;
        walk.init((stripe0 + i * stripe_step) % N, e0, ne, skip, take);

        for (uint32_t consumed = 0; consumed < take_units;) {
            if (relay) {
                // Wait before reserving: a slot holds at most max_slot_units, all of this iteration, so the target
                // covers every read below and is reached without our slot (a ring would deadlock otherwise).
                const uint32_t ahead = std::min(consumed + ag::max_slot_units(K, A, CAP), take_units);
                noc_semaphore_wait_min(data_valid, relay_base + ahead);
            }
            cb.reserve_back(1);
            const uint32_t slot = cb.get_write_ptr();
            uint32_t packets = 0, units = 0;
            for (; packets < K; ++packets) {
                auto* rec = reinterpret_cast<ag::PacketRecord*>(lay.record(slot, packets));
                const uint32_t payload = lay.payload(slot, packets);
                auto into_payload = [&](uint64_t src, uint32_t off, uint32_t n) { read(src, payload + off, n); };
                const uint32_t bytes = relay ? relay_builder.next(rec, into_payload)
                                             : input_builder.next(rec, into_payload, local(slot, packets));
                if (bytes == 0) {
                    break;
                }
                units += bytes / A;
            }
            // The walk ending before the schedule's units (a plan / walk mismatch) would push empty slots forever. A
            // kernel has no release-mode way to fail the op, so this is a watcher assert; the host sim checks that the
            // walk and the schedule agree.
            ASSERT(units != 0);
            noc.async_read_barrier();
            if (copy) {
                // Our own stripe into our own output, on a second VC so the writes do not queue behind the fabric
                // traffic on this NOC; flushed before the slot can be overwritten, acked by the barrier at the end.
                for (uint32_t j = 0; j < packets; ++j) {
                    const auto& rec = *reinterpret_cast<const ag::PacketRecord*>(lay.record(slot, j));
                    ag::for_each_segment(rec, local(slot, j), [&](uint32_t off, uint64_t dst, uint32_t bytes) {
                        noc.async_write<NocOptions::CUSTOM_VC, NOC_MAX_BURST_SIZE>(
                            CoreLocalMem<uint32_t>(lay.payload(slot, j) + off),
                            PrecomposedUnicastEndpoint{},
                            bytes,
                            {},
                            {.noc_addr = dst},
                            {.vc = NOC_UNICAST_WRITE_VC + 1});
                    });
                }
                noc.async_writes_flushed();
            }
            consumed += units;
            *reinterpret_cast<ag::SlotInfo*>(slot) = ag::SlotInfo{
                static_cast<uint16_t>(packets), static_cast<uint16_t>(consumed == take_units ? 1 : 0), units};
            ag::slot_publish();
            cb.push_back(1);
            ag::test_delay<RDELAY>();
        }
    }

    // Every unit (unicast) or done credit (multicast) addressed to us has landed; take only this launch's.
    ag::wait_and_consume(data_sem, end_wait);
    noc.async_atomic_barrier();
    // The local copy has landed in our output before the op completes.
    noc.async_write_barrier();
}
