// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#include <array>
#include <cstdint>
#include <utility>

// Store-and-forward AllGather is Fabric_1D only: every fabric send is a single 1-hop unicast to the neighbor.
namespace fabric_api = tt::tt_fabric::linear::experimental;

////////////////////////////////////////////////////////////////
// data_valid semaphore protocol
//
// data_valid counts the chunks upstream has relayed into our output -- cumulative over the op, reset at
// completion. A chunk's absolute position is base_chunk + within-slice offset, with base_chunk = (iter-1) *
// slice_count. The writer maintains the count (atomic-inc per chunks delivered); the reader waits on it with
// noc_semaphore_wait_min at the last chunk of each batch it reads, then a final wait for total_chunks before
// reset.
//
// Waiting on an absolute position (not a signal count) lets one reader path cover every case with no alignment
// or per-topology special-casing:
//   - full relay, and even-ring split prefix half (offset 0) / suffix half (offset = half): same per-batch
//     wait, differing only in base_chunk/count;
//   - sink stripe (a line endpoint's incoming, or a ring antipode): no relay wait, covered by the final
//     total_chunks wait;
//   - sink direction (num_iters == 0): only the total_chunks wait runs.
// So data_valid_granularity is a pure writer-side perf knob: the reader auto-paces to the writer's cadence.
////////////////////////////////////////////////////////////////

// Walks the output-tensor chunks of one stripe. Templated on the geometry so per-stripe starts are computed
// here and the matched/split fast path (output_chunks_per_page == 1) folds away. Re-pointed to any stripe each
// relay iteration via init(). A stripe's src address on this device equals its dst address on the neighbor, so
// reader and writer share this iterator unchanged.
template <
    uint32_t output_chunks_per_stripe,
    uint32_t output_chunks_per_page,
    uint32_t output_chunk_size,
    uint32_t num_devices>
class OutputStripeIterator {
    static constexpr uint32_t output_page_size = output_chunks_per_page * output_chunk_size;
    static constexpr uint32_t stripe_distance_chunks = num_devices * output_chunks_per_stripe;
    static constexpr uint32_t output_pages_per_row = stripe_distance_chunks / output_chunks_per_page;

public:
    // Point at `stripe` for the chunk range [start, start + count).
    FORCE_INLINE void init(uint32_t stripe, uint32_t start, uint32_t count) {
        const uint32_t s_start = (start / output_chunks_per_stripe) * stripe_distance_chunks +
                                 (start % output_chunks_per_stripe) + stripe * output_chunks_per_stripe;
        page_id_ = s_start / output_chunks_per_page;
        byte_off_ = (s_start % output_chunks_per_page) * output_chunk_size;
        if constexpr (output_chunks_per_page == 1) {
            phase_ = 0;
            stripe_jump_ = output_pages_per_row - (output_chunks_per_stripe - 1);
        } else {
            // In concat mode the page phase (and hence the stripe jump) depends on the stripe.
            const uint32_t off = (stripe * output_chunks_per_stripe) % output_chunks_per_page;
            phase_ = off * output_chunk_size;
            stripe_jump_ = output_pages_per_row - (off + output_chunks_per_stripe - 1) / output_chunks_per_page;
        }
        chunk_in_stripe_ = start % output_chunks_per_stripe;
        sent_ = 0;
        count_ = count;
    }

    FORCE_INLINE bool valid() const { return sent_ < count_; }

    // Return {output_page_id, byte_offset} of the current chunk, then advance.
    FORCE_INLINE std::pair<uint32_t, uint32_t> next() {
        std::pair<uint32_t, uint32_t> loc{page_id_, byte_off_};
        sent_++;
        if (++chunk_in_stripe_ == output_chunks_per_stripe) {
            chunk_in_stripe_ = 0;
            page_id_ += stripe_jump_;
            byte_off_ = phase_;
        } else {
            byte_off_ += output_chunk_size;
            if (byte_off_ == output_page_size) {
                byte_off_ = 0;
                page_id_++;
            }
        }
        return loc;
    }

private:
    uint32_t page_id_, byte_off_, chunk_in_stripe_, sent_, count_, phase_, stripe_jump_;
};

// Unicasts pages one hop to the neighbor over the V2 stateful send lane: pack several pages per scatter packet,
// or split a page too big for one packet. Drives either sender type (direct EDM or mux); both share the lane.
// setup_stateful_send_cmd_bufs() programs the cmd bufs once so each send just patches address/size.
// Requires DM_DEDICATED_NOC: the stateful data/sync cmd bufs must not alias the caller's local-copy write buf.
// eager_staging (mux, no init barrier): drain a full staging ring before waiting, else the slot wait deadlocks.
template <uint32_t page_size, uint32_t packet_size, bool eager_staging, typename SenderT>
class FabricWriter {
public:
    explicit FabricWriter(SenderT* sender) :
        sender{sender},
        scatter_packet_header{PacketHeaderPool::allocate_header(1)},
        unicast_packet_header{PacketHeaderPool::allocate_header(1)},
        sem_packet_header{PacketHeaderPool::allocate_header(1)},
        scatter_header({}, {}),
        chunk_count{0} {
        constexpr uint8_t num_hops = 1;  // store-and-forward: always the immediate neighbor
        scatter_packet_header->to_chip_unicast(num_hops);
        unicast_packet_header->to_chip_unicast(num_hops);
        sem_packet_header->to_chip_unicast(num_hops);
        if constexpr (use_scatter_write) {
            // Uniform, fixed chunk sizes; only dst addresses + count change per packet.
            for (uint32_t i = 0; i < max_pages_per_packet - 1; ++i) {
                scatter_header.chunk_size[i] = page_size;
            }
        }
        sender->setup_stateful_send_cmd_bufs();
    }

    ~FabricWriter() {
        ASSERT(chunk_count == 0);  // outstanding chunks! flush() not called correctly
    }

    void async_write(uint32_t l1_addr, uint64_t remote_noc_addr) {
        if constexpr (use_scatter_write) {
            // Queue up multiple pages (contiguous in L1) to send in a single scatter packet.
            if (chunk_count == 0) {
                start_l1_addr = l1_addr;
            }
            scatter_header.noc_address[chunk_count++] = remote_noc_addr;
            if (chunk_count == pages_per_packet) {
                scatter_header.chunk_count = chunk_count;
                send_scatter(start_l1_addr, payload_size);
                chunk_count = 0;
            }
        } else {
            // Page larger than a packet: split across packets.
            for (uint32_t packet = 0; packet < packets_per_page; ++packet) {
                send_unicast(
                    l1_addr, remote_noc_addr, (packet < packets_per_page - 1) ? payload_size : last_payload_size);
                l1_addr += payload_size;
                remote_noc_addr += payload_size;
            }
        }
    }

    // Barrier + data_valid incs, on the stateful lane too (a non-stateful inline write would clobber the SYNC
    // cmd buf). flush (default on) keeps a data_valid inc ordered after its payload.
    void atomic_inc(uint64_t noc_addr, uint32_t val) {
        noc_async_writes_flushed();
        sem_packet_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{noc_addr, val});
        wait_for_slot();
        sender->send_current_slot_stateful_non_blocking_from_address(
            (uint32_t)sem_packet_header, sizeof(PACKET_HEADER_TYPE));
    }

    // Call this before popping CB entry
    void async_writes_flushed() {
        if constexpr (use_scatter_write) {
            static_assert(min_pages_per_packet == 2, "hardcoded to assume scatter_write min_pages_per_packet == 2");
            if (chunk_count > 0) {
                if (chunk_count == 1) {
                    // scatter_write needs chunk_count >= 2, so send a lone trailing chunk as a unicast write.
                    send_unicast(start_l1_addr, scatter_header.noc_address[0], page_size);
                } else {
                    scatter_header.chunk_count = chunk_count;
                    send_scatter(start_l1_addr, chunk_count * page_size);
                }
                chunk_count = 0;
            }
        }
        // Wait for Fabric writes to be sent out before popping CB entry.
        noc_async_writes_flushed();
    }

private:
    // Eager: drain a full staging ring first, else the slot wait spins on a ring nothing frees.
    FORCE_INLINE void wait_for_slot() {
        if constexpr (eager_staging) {
            if (sender->is_staging_ring_full()) {
                sender->template flush<true>();
            }
        }
        sender->wait_for_empty_write_slot();
    }

    // Flush first: the previous send's header write must land before we overwrite the reusable header.
    FORCE_INLINE void send_scatter(uint32_t l1_addr, uint32_t size) {
        noc_async_writes_flushed();
        scatter_packet_header->to_noc_unicast_scatter_write(scatter_header, size);
        wait_for_slot();
        sender->send_current_slot_stateful_non_blocking(l1_addr, size, (uint32_t)scatter_packet_header);
    }

    FORCE_INLINE void send_unicast(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t size) {
        noc_async_writes_flushed();
        unicast_packet_header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{remote_noc_addr}, size);
        wait_for_slot();
        sender->send_current_slot_stateful_non_blocking(l1_addr, size, (uint32_t)unicast_packet_header);
    }

    // Fabric limits
    static constexpr uint32_t max_pages_per_packet = NOC_SCATTER_WRITE_MAX_CHUNKS;
    static constexpr uint32_t min_pages_per_packet = NOC_SCATTER_WRITE_MIN_CHUNKS;
    // When page_size < packet_size
    static constexpr uint32_t pages_per_packet = std::min(packet_size / page_size, max_pages_per_packet);  // div_down
    // When page_size > packet_size
    static constexpr uint32_t packets_per_page = (page_size + packet_size - 1) / packet_size;  // div_up
    // Use scatter_write or unicast_write (currently scatter_write imposes a min chunk_count)
    static constexpr bool use_scatter_write = pages_per_packet >= min_pages_per_packet;
    // Steady-state payload size. Note (pages_per_packet * page_size) may not equal packet_size.
    static constexpr uint32_t payload_size = use_scatter_write ? (pages_per_packet * page_size) : packet_size;
    // Last payload for the page_size >= packet_size case (a page sent as multiple packets).
    static constexpr uint32_t last_payload_size = page_size - ((packets_per_page - 1) * packet_size);

    SenderT* sender;  // direct or mux sender
    volatile tt_l1_ptr PACKET_HEADER_TYPE* scatter_packet_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* unicast_packet_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* sem_packet_header;
    NocUnicastScatterCommandHeader scatter_header;
    uint8_t chunk_count;     // accumulated chunks not yet sent in a packet
    uint32_t start_l1_addr;  // start address of the accumulated contiguous chunks
};
