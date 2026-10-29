// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"

#include <cstdint>
#include <type_traits>

#include "all_gather_common.hpp"

using address_t = uint32_t;
namespace fabric_api = tt::tt_fabric::linear::experimental;

namespace {

// Store-and-forward: every packet goes to the immediate neighbor.
constexpr uint8_t num_hops = 1;

// The 2D route rides in the header; on 1D the hop count set by set_state is the whole route. Keyed on the header type
// because a mux client never gets the FABRIC_2D define.
FORCE_INLINE void set_route(volatile PACKET_HEADER_TYPE* hdr, uint16_t chip, uint16_t mesh) {
    if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>) {
        fabric_set_unicast_route(reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(hdr), chip, mesh);
    }
}

// When to signal data_valid: after every `every_slots` slots, and always after the last slot of an iteration, so a
// downstream wait target (capped at the iteration's units) is always reached.
class SignalCadence {
public:
    explicit SignalCadence(uint32_t every_slots) : every_(every_slots) {}

    // Accounts one popped slot; returns the units to signal now (0: not yet).
    uint32_t on_slot(const ag::SlotInfo& info) {
        pending_ += info.units;
        if (info.iter_end == 0 && ++slots_ < every_) {
            return 0;
        }
        slots_ = 0;
        const uint32_t units = pending_;
        pending_ = 0;
        return units;
    }

private:
    uint32_t every_, slots_ = 0, pending_ = 0;
};

}  // namespace

// Store-and-forward writer: CB consumer, one channel of its direction's fabric mux. Its one semaphore wait is the init
// handshake, before its first send, after it has sent its own (see the protocol note in all_gather_common.hpp);
// otherwise its progress depends only on the CB and the mux. Per slot: encode each packet record into the slot's own
// header with the fabric setters, send header + payload, signal data_valid at the cadence, flush, pop. Headers live in
// the slot, so flushing before the pop is what lets both the payload and its header be reused.
void kernel_main() {
    ///////////////////////////////////////////////////
    // COMPILE TIME ARGS
    ///////////////////////////////////////////////////
    constexpr uint32_t unit_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t packet_cap = get_compile_time_arg_val(1);
    constexpr uint32_t packets_per_slot = get_compile_time_arg_val(2);
    constexpr uint32_t packet_header_size = get_compile_time_arg_val(3);
    constexpr uint32_t cb0_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t mux_num_buffers_per_channel = get_compile_time_arg_val(6);

    constexpr ag::SlotLayout lay = ag::slot_offsets(packets_per_slot, unit_bytes, packet_header_size, packet_cap);
    static_assert(lay.slot_bytes == cb_page_size, "host and kernel disagree on the slot layout");
    static_assert(sizeof(PACKET_HEADER_TYPE) <= lay.hdr_stride, "the slot has no room for the fabric packet headers");
    static_assert(ag::kSegmentsPerPacket <= NOC_SCATTER_WRITE_MAX_CHUNKS, "a packet has more segments than a scatter");

    ///////////////////////////////////////////////////
    // RUNTIME ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    const uint32_t num_iters = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slice_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_count = get_arg_val<uint32_t>(arg_idx++);
    const address_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const address_t data_valid_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t barrier_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);  // neighbor opposite-direction core
    const uint32_t barrier_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t data_valid_sem_noc_x = get_arg_val<uint32_t>(arg_idx++);  // mirror core (data_valid_sem target)
    const uint32_t data_valid_sem_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t data_valid_cadence = get_arg_val<uint32_t>(arg_idx++);  // signal data_valid every this many slots
    const uint16_t neighbor_chip_id = get_arg_val<uint32_t>(arg_idx++);
    const uint16_t neighbor_mesh_id = get_arg_val<uint32_t>(arg_idx++);

    // A direction with no neighbor (a line endpoint) relays nothing; no mux channel was declared for it. Every other
    // writer runs to close(), even with nothing to send: a mux hangs if a declared channel is never closed.
    if (num_iters == 0) {
        return;
    }

    ///////////////////////////////////////////////////
    // FABRIC INIT
    ///////////////////////////////////////////////////

    Noc noc;
    CircularBuffer cb(cb0_id);
    using Sender = tt::tt_fabric::FabricMuxV2Sender<false, mux_num_buffers_per_channel>;
    Sender sender = Sender::build_from_args(arg_idx);
    sender.open();

    // One signal header, re-encoded per signal; a flush always separates two uses.
    auto* signal = PacketHeaderPool::allocate_header(1);
    fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        signal, num_hops, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, 1});
    set_route(signal, neighbor_chip_id, neighbor_mesh_id);
    // Tell the neighbor's opposite-direction worker we are alive: our output exists, and we are done reading the
    // previous launch's.
#ifndef AG_TEST_NO_INIT  // TEST ONLY
    fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
        UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
        &sender,
        signal,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            ag::fabric_frame_addr(barrier_sem_noc_x, barrier_sem_noc_y, barrier_sem), 1});
    noc.async_writes_flushed();
#endif

    // The headers of every slot are ours for the whole op (the reader never writes them), so each gets its command
    // type and route once: a unicast-write and a scatter-write header per packet. The CB starts at its base, and its
    // pages follow one another.
    for (uint32_t s = 0, slot = cb.get_read_ptr(); s < ag::kNumSlots; ++s, slot += cb_page_size) {
        for (uint32_t j = 0; j < packets_per_slot; ++j) {
            auto* unicast_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(lay.unicast_header(slot, j));
            fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(unicast_hdr, num_hops);
            set_route(unicast_hdr, neighbor_chip_id, neighbor_mesh_id);
            auto* scatter_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(lay.scatter_header(slot, j));
            fabric_api::fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
                scatter_hdr, num_hops);
            set_route(scatter_hdr, neighbor_chip_id, neighbor_mesh_id);
        }
    }
    // Our data goes into the neighbor's output: wait until it is alive (it sent its init the same way). Our reader
    // fills the slots meanwhile.
#ifndef AG_TEST_NO_INIT  // TEST ONLY
    {
        AG_TEST_ZONE("AG-WR-INIT");
        ag::wait_and_consume(barrier_sem, 1);
    }
#endif
    const uint64_t data_valid_addr = ag::fabric_frame_addr(data_valid_sem_noc_x, data_valid_sem_noc_y, data_valid_sem);

    ///////////////////////////////////////////////////
    // MAIN
    ///////////////////////////////////////////////////

    SignalCadence cadence(data_valid_cadence);
    {
        AG_TEST_ZONE("AG-WR-LOOP");  // TEST ONLY
        for (uint32_t iter = 0; iter < num_iters; ++iter) {
            const uint32_t count = (iter + 1 == num_iters) ? final_count : slice_count;
            // An empty pass (empty slice, or an even-ring half of nothing) pushes no slot.
            for (bool iter_end = count == 0; !iter_end;) {
                cb.wait_front(1);
                const uint32_t slot = cb.get_read_ptr();
                ag::slot_acquire();
                const ag::SlotInfo info = *reinterpret_cast<const ag::SlotInfo*>(slot);
                for (uint32_t j = 0; j < info.packets; ++j) {
                    const auto& rec = *reinterpret_cast<const ag::PacketRecord*>(lay.record(slot, j));
                    const uint32_t payload = lay.payload(slot, j);
                    if (rec.segments == 1) {
                        fabric_api::fabric_unicast_noc_unicast_write_with_state<
                            UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                            &sender,
                            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(lay.unicast_header(slot, j)),
                            payload,
                            tt::tt_fabric::NocUnicastCommandHeader{rec.dst[0]},
                            rec.payload_bytes);
                    } else {
                        fabric_api::fabric_unicast_noc_scatter_write_with_state<
                            UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                            UnicastScatterWriteUpdateMask::PayloadSize>(
                            &sender,
                            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(lay.scatter_header(slot, j)),
                            payload,
                            tt::tt_fabric::NocUnicastScatterCommandHeader(
                                rec.dst, rec.bytes, static_cast<uint8_t>(rec.segments)),
                            rec.payload_bytes);
                    }
                }
                // Same sender as the data, so the channel's FIFO lands the signal after the units it counts.
                if (const uint32_t units = cadence.on_slot(info)) {
                    fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
                        UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
                        &sender, signal, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{data_valid_addr, units});
                }
                // Payload, headers and the signal have left L1 before the reader may refill the slot.
                noc.async_writes_flushed();
                cb.pop_front(1);
                iter_end = info.iter_end != 0;
            }
        }
    }

    ///////////////////////////////////////////////////
    // CLEANUP
    ///////////////////////////////////////////////////

    AG_TEST_ZONE("AG-WR-TAIL");  // TEST ONLY
    noc.async_write_barrier();
    noc.async_atomic_barrier();
    sender.close();
}
