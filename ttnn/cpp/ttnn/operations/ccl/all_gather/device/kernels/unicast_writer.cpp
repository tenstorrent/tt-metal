// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// all_gather unicast writer: the slot consumer, one Mux V2 channel. Its one semaphore wait is init, before its first
// send, after it has sent its own (ag_sync.hpp: deadlock freedom); otherwise its progress depends only on the CB, mux
// credits and the mux handshakes.
//
// Per slot: wait -> encode each record into the slot's own header (the official setters) -> send header + payload
// through the mux -> [data_valid at the cadence] -> flush -> pop. Headers live in the slot, so flush-before-pop is
// what lets both the payload and its header be reused.

#include <cstdint>
#include <type_traits>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"

#include "ag_walk.hpp"
#include "ag_packet.hpp"
#include "ag_sync.hpp"

namespace fabric_api = tt::tt_fabric::linear::experimental;

namespace {

// Store-and-forward: every packet goes to the immediate neighbor.
constexpr uint8_t kHops = 1;

// The 2D route rides in the header; on 1D the hop count set by set_state is the whole route. Keyed on the header type
// because a mux client never gets the FABRIC_2D define.
FORCE_INLINE void set_route(volatile PACKET_HEADER_TYPE* hdr, uint16_t chip, uint16_t mesh) {
    if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>) {
        fabric_set_unicast_route(reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(hdr), chip, mesh);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t A = get_named_compile_time_arg_val("A");
    constexpr uint32_t CAP = get_named_compile_time_arg_val("CAP");
    constexpr uint32_t K = get_named_compile_time_arg_val("K");
    constexpr uint32_t SLOT_BYTES = get_named_compile_time_arg_val("SLOT_BYTES");
    constexpr uint32_t CB_ID = get_named_compile_time_arg_val("CB_ID");
    constexpr uint32_t CB_SLOTS = get_named_compile_time_arg_val("CB_SLOTS");
    constexpr uint32_t MUX_BUFS = get_named_compile_time_arg_val("MUX_BUFS");
    constexpr uint32_t WDELAY = get_named_compile_time_arg_val("WDELAY");

    constexpr ag::SlotLayout lay = ag::slot_offsets(K, A, sizeof(PACKET_HEADER_TYPE), CAP);
    static_assert(lay.slot_bytes == SLOT_BYTES, "host and kernel disagree on the packet header size");

    size_t arg_idx = 0;
    const bool active = get_arg_val<uint32_t>(arg_idx++) != 0;
    const uint32_t iters = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ne = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t last_take = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cadence_slots = get_arg_val<uint32_t>(arg_idx++);
    const bool init_send = get_arg_val<uint32_t>(arg_idx++) != 0;
    const uint32_t init_wait = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t init_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t data_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t partner_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t partner_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mirror_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mirror_y = get_arg_val<uint32_t>(arg_idx++);
    const uint16_t dst_chip = get_arg_val<uint32_t>(arg_idx++);
    const uint16_t dst_mesh = get_arg_val<uint32_t>(arg_idx++);

    // An idle direction (a line end) has no mux. Everything below sits inside this block with no early return, so
    // an active writer always reaches close(): a mux hangs if any declared channel is never closed.
    if (active) {
        Noc noc;
        CircularBuffer cb(CB_ID);
        using Sender = tt::tt_fabric::FabricMuxV2Sender<false, MUX_BUFS>;
        Sender sender = Sender::build_from_args(arg_idx);
        sender.open();

        // One signal header, re-encoded per signal; a flush always separates two uses.
        auto* signal = PacketHeaderPool::allocate_header(1);
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            signal, kHops, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
        set_route(signal, dst_chip, dst_mesh);
        if (init_send) {
            // Tell the neighbor's opposite-direction reader we are alive (our output exists).
            fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
                UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
                &sender,
                signal,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    ag::fabric_frame_addr(partner_x, partner_y, init_sem), 1});
            noc.async_writes_flushed();
        }
        // Our data goes into the neighbor's output: wait until it is alive (it sent its init the same way). Our reader
        // fills the slots meanwhile.
        ag::wait_and_consume(init_sem, init_wait);
        const uint64_t data_valid = ag::fabric_frame_addr(mirror_x, mirror_y, data_sem);

        ag::SignalCadence cadence(cadence_slots);
        // The header region is ours for the whole op: route each slot's headers the first time we see that slot.
        uint32_t routed[CB_SLOTS] = {};
        uint32_t num_routed = 0;
        for (uint32_t i = 0; i < iters; ++i) {
            const uint32_t take = (i + 1 == iters) ? last_take : ne;
            // An empty pass (empty slice, or an even-ring half of nothing) pushes no slot.
            for (bool iter_end = take == 0; !iter_end;) {
                cb.wait_front(1);
                const uint32_t slot = cb.get_read_ptr();
                ag::slot_acquire();
                bool seen = false;
                for (uint32_t r = 0; r < num_routed; ++r) {
                    seen = seen || routed[r] == slot;
                }
                ASSERT(seen || num_routed < CB_SLOTS);  // a CB of CB_SLOTS pages has CB_SLOTS slot addresses
                if (!seen && num_routed < CB_SLOTS) {
                    for (uint32_t j = 0; j < K; ++j) {
                        auto* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(lay.header(slot, j));
                        fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
                            hdr, kHops);
                        set_route(hdr, dst_chip, dst_mesh);
                    }
                    routed[num_routed++] = slot;
                }
                const ag::SlotInfo info = *reinterpret_cast<const ag::SlotInfo*>(slot);
                for (uint32_t j = 0; j < info.packets; ++j) {
                    const auto& rec = *reinterpret_cast<const ag::PacketRecord*>(lay.record(slot, j));
                    auto* hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(lay.header(slot, j));
                    const uint32_t payload = lay.payload(slot, j);
                    if (rec.cmd.chunk_count == 1) {
                        hdr->to_noc_unicast_write(
                            tt::tt_fabric::NocUnicastCommandHeader{rec.cmd.noc_address[0]}, rec.payload_bytes);
                        fabric_api::fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::None>(
                            &sender, hdr, payload);
                    } else {
                        hdr->to_noc_unicast_scatter_write(rec.cmd, rec.payload_bytes);
                        fabric_api::fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::None>(
                            &sender, hdr, payload);
                    }
                }
                // Same sender as the data, so the channel's FIFO lands the signal after the units it counts.
                if (const uint32_t units = cadence.on_slot(info)) {
                    fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
                        UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
                        &sender, signal, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{data_valid, units});
                }
                // Payload, headers and the signal have left L1 before the reader may refill the slot.
                noc.async_writes_flushed();
                cb.pop_front(1);
                iter_end = info.iter_end != 0;
                ag::test_delay<WDELAY>();
            }
        }
        noc.async_write_barrier();
        noc.async_atomic_barrier();
        sender.close();
    }
}
