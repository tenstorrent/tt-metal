// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// all_gather reader, shared by unicast and multicast: the slot producer. It never touches the fabric, and it starts at
// once: the init handshake gates only the writer's remote sends, so the first slots and the local copy overlap it.
//
// Per slot: [relay: wait for the units this slot may read, holding no slot] -> reserve -> build up to
// packets_per_slot packets (records + payload reads) -> one read barrier -> [local copy, flushed] -> SlotInfo -> push.
// Iteration 0 reads this device's stripe from the input; iteration i >= 1 (unicast) re-reads the stripe upstream
// relayed into our output in its iteration i - 1. See all_gather_sync.hpp for the sync protocol and the address rule.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

#include "ttnn/operations/ccl/all_gather/shared_with_host/all_gather_walk.hpp"
#include "all_gather_packet.hpp"
#include "all_gather_sync.hpp"

void kernel_main() {
    constexpr uint32_t unit_bytes = get_named_compile_time_arg_val("unit_bytes");
    constexpr uint32_t max_packet_bytes = get_named_compile_time_arg_val("max_packet_bytes");
    constexpr uint32_t packet_cap = get_named_compile_time_arg_val("packet_cap");
    constexpr uint32_t packets_per_slot = get_named_compile_time_arg_val("packets_per_slot");
    constexpr uint32_t slot_bytes = get_named_compile_time_arg_val("slot_bytes");
    constexpr uint32_t slot_cb = get_named_compile_time_arg_val("slot_cb");
    constexpr uint32_t in_page_units = get_named_compile_time_arg_val("in_page_units");
    constexpr uint32_t out_page_units = get_named_compile_time_arg_val("out_page_units");
    constexpr uint32_t stripe_units = get_named_compile_time_arg_val("stripe_units");
    constexpr uint32_t num_devices = get_named_compile_time_arg_val("num_devices");
    constexpr uint32_t element_units = get_named_compile_time_arg_val("element_units");
    constexpr bool dst_cross = get_named_compile_time_arg_val("dst_cross") != 0;
    constexpr bool src_cross = get_named_compile_time_arg_val("src_cross") != 0;
    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    static_assert(
        packet_cap % unit_bytes == 0 && packet_cap >= unit_bytes && packet_cap <= max_packet_bytes,
        "the packet cap is a whole number of units within the payload");
    constexpr ag::SlotLayout lay =
        ag::slot_offsets(packets_per_slot, unit_bytes, sizeof(PACKET_HEADER_TYPE), packet_cap);
    static_assert(lay.slot_bytes == slot_bytes, "host and kernel disagree on the packet header size");

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
    CircularBuffer cb(slot_cb);

    constexpr ag::WalkGeo geo{
        .unit_bytes = unit_bytes,
        .out_page_units = out_page_units,
        .stripe_units = stripe_units,
        .num_devices = num_devices,
        .element_units = element_units,
        .max_packet_bytes = max_packet_bytes,
        .dst_cross = dst_cross};
    ag::Walk<decltype(out), ag::FabricFrameAddr> walk(out, geo);
    ag::InputPieces<decltype(in)> input_pieces(in, unit_bytes, in_page_units, element_units, src_cross);
    ag::RelayPieces relay_pieces(unit_bytes, element_units);
    ag::PacketBuilder<decltype(walk), decltype(input_pieces)> input_builder(walk, input_pieces, packet_cap);
    ag::PacketBuilder<decltype(walk), decltype(relay_pieces)> relay_builder(walk, relay_pieces, packet_cap);

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
        const uint32_t take_units = take * element_units;
        const bool relay = i > 0;
        const bool copy = i == 0 && local_copy;
        // Upstream sent its iteration i - 1 in walk order and signals cumulative units, so the first `skip`
        // elements of that pass precede ours.
        const uint32_t relay_base = relay ? (i - 1) * iter_units + skip * element_units : 0;
        walk.init((stripe0 + i * stripe_step) % num_devices, e0, ne, skip, take);

        for (uint32_t consumed = 0; consumed < take_units;) {
            if (relay) {
                // Wait before reserving: a slot holds at most max_slot_units, all of this iteration, so the target
                // covers every read below and is reached without our slot (a ring would deadlock otherwise).
                const uint32_t ahead =
                    std::min(consumed + ag::max_slot_units(packets_per_slot, unit_bytes, packet_cap), take_units);
                noc_semaphore_wait_min(data_valid, relay_base + ahead);
            }
            cb.reserve_back(1);
            const uint32_t slot = cb.get_write_ptr();
            uint32_t packets = 0, units = 0;
            for (; packets < packets_per_slot; ++packets) {
                auto* rec = reinterpret_cast<ag::PacketRecord*>(lay.record(slot, packets));
                const uint32_t payload = lay.payload(slot, packets);
                auto into_payload = [&](uint64_t src, uint32_t off, uint32_t n) { read(src, payload + off, n); };
                const uint32_t bytes = relay ? relay_builder.next(rec, into_payload)
                                             : input_builder.next(rec, into_payload, local(slot, packets));
                if (bytes == 0) {
                    break;
                }
                units += bytes / unit_bytes;
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
        }
    }

    // Every unit (unicast) or done credit (multicast) addressed to us has landed; take only this launch's.
    ag::wait_and_consume(data_sem, end_wait);
    noc.async_atomic_barrier();
    // The local copy has landed in our output before the op completes.
    noc.async_write_barrier();
}
