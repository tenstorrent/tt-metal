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

// Unicasts pages one hop to the single neighbor. Handles packetization (pack several pages into one
// scatter-write packet when they fit, else split a big page across packets).
//
// Templated on the sender type (SenderT*) so the same writer drives either a direct WorkerToFabricEdmSender
// (one worker per direction) or a WorkerToFabricMuxSender (workers sharing a fabric mux). The send calls are
// base sender-pointer overloads that accept either; for 1D-linear all routing is to_chip_unicast(1) set via
// set_state, so no route-manager is needed. FabricWriter owns its two packet headers directly.
template <uint32_t page_size, uint32_t packet_size, typename SenderT>
class FabricWriter {
public:
    FabricWriter(const Noc& noc, SenderT* sender) :
        noc{noc},
        sender{sender},
        scatter_packet_header{PacketHeaderPool::allocate_header(1)},
        unicast_packet_header{PacketHeaderPool::allocate_header(1)},
        scatter_header({}, {}),
        chunk_count{0} {
        std::array<uint64_t, max_pages_per_packet> dummy_addrs{};  // init to 0s
        std::array<uint16_t, max_pages_per_packet - 1> chunk_sizes{};
        chunk_sizes.fill(page_size);
        constexpr uint8_t num_hops = 1;  // store-and-forward: always the immediate neighbor

        fabric_api::fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            scatter_packet_header,
            num_hops,
            NocUnicastScatterCommandHeader(dummy_addrs.data(), chunk_sizes.data(), pages_per_packet),
            payload_size);

        fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
            unicast_packet_header, num_hops);
    }

    ~FabricWriter() {
        ASSERT(chunk_count == 0);  // outstanding chunks! flush() not called correctly
    }

    void async_write(uint32_t l1_addr, uint64_t remote_noc_addr) {
        if constexpr (use_scatter_write) {
            // Queue up multiple pages to send in a single packet.
            // Assumption: pages are contiguous in local memory (L1).
            // Note: currently, scatter_write necessitates chunk_count >= 2.
            if (chunk_count == 0) {
                start_l1_addr = l1_addr;
            }
            scatter_header.noc_address[chunk_count++] = remote_noc_addr;
            if (chunk_count == pages_per_packet) {
                noc.async_writes_flushed();
                scatter_header.chunk_count = chunk_count;
                fabric_api::fabric_unicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::PayloadSize>(
                    sender, scatter_packet_header, start_l1_addr, scatter_header, payload_size);
                chunk_count = 0;
            }
        } else {
            // Page larger than a packet: split across packets.
            for (uint32_t packet = 0; packet < packets_per_page; ++packet) {
                noc.async_writes_flushed();
                fabric_api::fabric_unicast_noc_unicast_write_with_state<
                    UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    sender,
                    unicast_packet_header,
                    l1_addr,
                    tt::tt_fabric::NocUnicastCommandHeader{remote_noc_addr},
                    (packet < packets_per_page - 1) ? payload_size : last_payload_size);
                l1_addr += payload_size;
                remote_noc_addr += payload_size;
            }
        }
    }

    // Call this before popping CB entry
    void async_writes_flushed() {
        if constexpr (use_scatter_write) {
            static_assert(min_pages_per_packet == 2, "hardcoded to assume scatter_write min_pages_per_packet == 2");
            if (chunk_count > 0) {
                noc.async_writes_flushed();
                if (chunk_count == 1) {
                    // Note: currently, scatter_write necessitates chunk_count >= 2, so we use unicast_write
                    // for chunk_count == 1.
                    // Note: this is hardcoded assuming NOC_SCATTER_WRITE_MIN_CHUNKS == 2. Else need to put
                    // the below unicast_write in a loop.
                    fabric_api::fabric_unicast_noc_unicast_write_with_state<
                        UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                        sender,
                        unicast_packet_header,
                        start_l1_addr,
                        tt::tt_fabric::NocUnicastCommandHeader{scatter_header.noc_address[0]},
                        page_size);
                } else {
                    scatter_header.chunk_count = chunk_count;
                    fabric_api::fabric_unicast_noc_scatter_write_with_state<
                        UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::PayloadSize>(
                        sender, scatter_packet_header, start_l1_addr, scatter_header, chunk_count * page_size);
                }
                chunk_count = 0;
            }
        }
        // Wait for Fabric writes to be sent out before popping CB entry
        noc.async_writes_flushed();
    }

private:
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

    const Noc& noc;
    SenderT* sender;  // direct or mux sender
    volatile tt_l1_ptr PACKET_HEADER_TYPE* scatter_packet_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* unicast_packet_header;
    NocUnicastScatterCommandHeader scatter_header;
    uint8_t chunk_count;     // accumulated chunks not yet sent in a packet
    uint32_t start_l1_addr;  // start address of the accumulated contiguous chunks
};
