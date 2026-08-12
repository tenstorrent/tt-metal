// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

// Store-and-forward AllGather: every fabric send is a single 1-hop unicast to the neighbor.
// Runs on any effectively-1D topology (both Fabric 1D and 2D).
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

////////////////////////////////////////////////////////////////
// FabricWriter tunables
//
// Kernel-local on purpose: both are pure perf knobs with no correctness or API implications, so they belong
// here to be swept and then baked in, not plumbed through the op.
////////////////////////////////////////////////////////////////

// Depth of the packet-header ring. A header cannot be re-pointed while its copy into the fabric buffer is
// still in flight, so the ring depth is exactly how many packets can be outstanding: at depth 1 every packet
// costs a flush and nothing pipelines. Depths past the sender's slot count are harmless but pointless --
// reserve_slot() blocks first.
// Swept 1/2/4/8, jointly with the mux's num_buffers_per_channel and the posted-write flag below, at 256 KB /
// 5 MB / 48 MB: every combination lands within noise of every other. Depth cannot help much by construction
// anyway -- async_writes_flushed() runs once per CB page before the page is popped, so a CB page's worth of
// packets is the real ceiling on how many can be outstanding. The op is transaction-rate bound downstream of
// the worker, not short of packets in flight.
constexpr uint32_t fabric_header_ring_size = 2;

// Post the payload/header writes into the fabric buffer, i.e. no ack. Drops the ack return traffic from the
// worker's NOC, which the mux forwarder shares. Ordering against the credit doorbell is preserved: the
// doorbell is issued after the payload, to the same destination on the same VC.
// Swept: no measurable effect either way, same reason as the ring depth above. Left off, since off is the
// conservative choice -- posted writes drop the ack the flush would otherwise count.
constexpr bool fabric_use_posted_writes = false;

// Unicasts pages one hop to the single neighbor. Handles packetization (pack several pages into one
// scatter-write packet when they fit, else split a big page across packets), and the semaphore increments
// that announce them.
//
// Templated on the sender type (SenderT*) so the same writer drives either a direct WorkerToFabricEdmSender
// (one worker per direction) or a FabricMuxV2Sender (workers sharing a fabric mux). Both expose the same
// stateful send lane, so no route-manager is needed -- which is also why this class routes its own headers.
//
// Sends use the stateful lane: the NOC command buffers are programmed once in the constructor and each send
// only rewrites src/dst/len, instead of reprogramming coordinates, VC and command fields per packet. The
// headers are therefore populated here (populate_unicast_*_fields) rather than by the linear fabric API,
// which only drives the non-stateful lane.
template <uint32_t page_size, uint32_t packet_size, typename SenderT>
class FabricWriter {
public:
    FabricWriter(const Noc& noc, SenderT* sender, uint16_t neighbor_chip_id, uint16_t neighbor_mesh_id) :
        noc{noc}, sender{sender}, scatter_header({}, {}), chunk_count{0} {
        constexpr uint8_t num_hops = 1;  // store-and-forward: always the immediate neighbor

        for (uint32_t i = 0; i < fabric_header_ring_size; ++i) {
            data_headers[i] = PacketHeaderPool::allocate_header(1);
            if constexpr (use_scatter_write) {
                std::array<uint64_t, max_pages_per_packet> dummy_addrs{};  // init to 0s
                std::array<uint16_t, max_pages_per_packet - 1> chunk_sizes{};
                chunk_sizes.fill(page_size);
                fabric_api::fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::ChunkSizes>(
                    data_headers[i],
                    num_hops,
                    NocUnicastScatterCommandHeader(dummy_addrs.data(), chunk_sizes.data(), pages_per_packet));
            } else {
                fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
                    data_headers[i], num_hops);
            }
            set_route(data_headers[i], neighbor_chip_id, neighbor_mesh_id);
        }

        if constexpr (use_scatter_write) {
            // scatter_write imposes a min chunk count, so a lone trailing chunk goes out as a plain unicast
            // write. Kept out of the ring: it is rare, and sharing the ring would cost a header slot per entry.
            tail_header = PacketHeaderPool::allocate_header(1);
            fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(tail_header, num_hops);
            set_route(tail_header, neighbor_chip_id, neighbor_mesh_id);
        }

        // One atomic-inc header for the "alive" barrier inc + data_valid signals; Flush orders it after the
        // payload it announces.
        sem_header = PacketHeaderPool::allocate_header(1);
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            sem_header, num_hops, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});
        set_route(sem_header, neighbor_chip_id, neighbor_mesh_id);

        // Program the send command buffers once. From here until teardown nothing else on this RISC may use
        // write_reg_cmd_buf or write_at_cmd_buf: the CB ops touch no NOC at all, and the local output copy uses
        // write_cmd_buf, so the writer kernel is clear -- but note that the sender's own close() does clobber
        // the credit state, which is why it only runs after the last send.
        sender->template setup_stateful_send_cmd_bufs<fabric_use_posted_writes>();
    }

    ~FabricWriter() {
        ASSERT(chunk_count == 0);  // outstanding chunks! flush() not called correctly
    }

    // Increment a semaphore on the neighbor.
    void atomic_inc(uint64_t addr, uint32_t val) {
        // sem_header is not part of the ring, so drain before re-pointing it.
        flush_if_pending();
        populate_unicast_atomic_inc_fields<UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
            sem_header, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{addr, val});
        send_header_only(sem_header);
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
                send_scatter_packet(payload_size);
            }
        } else {
            // Page larger than a packet: split across packets.
            for (uint32_t packet = 0; packet < packets_per_page; ++packet) {
                const uint16_t size = (packet < packets_per_page - 1) ? payload_size : last_payload_size;
                populate_unicast_write_fields<UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                    current_data_header(), size, tt::tt_fabric::NocUnicastCommandHeader{remote_noc_addr});
                send_data_packet(l1_addr, size);
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
                if (chunk_count == 1) {
                    // Note: currently, scatter_write necessitates chunk_count >= 2, so we use unicast_write
                    // for chunk_count == 1.
                    // Note: this is hardcoded assuming NOC_SCATTER_WRITE_MIN_CHUNKS == 2. Else need to put
                    // the below unicast_write in a loop.
                    // tail_header is not part of the ring, so drain before re-pointing it.
                    flush_if_pending();
                    populate_unicast_write_fields<
                        UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                        tail_header, page_size, tt::tt_fabric::NocUnicastCommandHeader{scatter_header.noc_address[0]});
                    send_with_payload(tail_header, start_l1_addr, page_size);
                } else {
                    send_scatter_packet(chunk_count * page_size);
                }
                chunk_count = 0;
            }
        }
        // The CB page is about to be reused: every payload read out of it must have left L1. Already satisfied
        // if the last send happened to land on a ring wrap.
        flush_if_pending();
    }

private:
    // Fabric limits
    //
    // These two caps set the op's ceiling, because it is transaction-rate bound: measured time tracks the
    // packet count, not the byte count (halving the payload roughly doubles the time). Payload per packet is
    // min(packet_size/page, 4) * page, so a tile page of 2 KB is stuck at 8 KB per packet however big the
    // fabric packet is, and page size ends up worth 16x across the range measured.
    // TODO(perf): two openings. Chunk count is not free either -- at an identical 8 KB payload, one chunk
    // beats four by ~10%, which points at the receiving ERISC's per-chunk NOC write issue rate. And a
    // destination-contiguous run could go out as one plain unicast write instead of a scatter, lifting both
    // caps at once; interleaved DRAM never gives that, but a sharded output would.
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

    // The stateful send lane owns write_reg_cmd_buf (payload/header) and write_at_cmd_buf (credit doorbell).
    // Under dynamic NOC those alias the general-purpose write_cmd_buf that the local output copy uses, and the
    // programmed state would be clobbered mid-loop.
    static_assert(noc_mode == DM_DEDICATED_NOC, "FabricWriter's stateful send lane requires dedicated NOC mode");

    // Ring + optional scatter tail + semaphore. The pool hangs at runtime when exhausted, so bound it here.
    static constexpr uint32_t num_headers = fabric_header_ring_size + (use_scatter_write ? 2u : 1u);
    static_assert(
        num_headers <= NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType,
        "fabric_header_ring_size does not fit this RISC's share of the packet header pool");

    static constexpr NocOptions flush_opts = fabric_use_posted_writes ? NocOptions::POSTED : NocOptions::DEFAULT;

    // For Fabric_2D, set_state() sets routes only in its RoutingPlaneConnectionManager overloads, so set them
    // here. Keyed on the header type since the FABRIC_2D define is absent on the mux path.
    static void set_route(
        volatile tt_l1_ptr PACKET_HEADER_TYPE* header, uint16_t neighbor_chip_id, uint16_t neighbor_mesh_id) {
        if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>) {
            using MeshHeader = volatile tt::tt_fabric::HybridMeshPacketHeader*;
            fabric_set_unicast_route(reinterpret_cast<MeshHeader>(header), neighbor_chip_id, neighbor_mesh_id);
        }
    }

    FORCE_INLINE void flush() {
        noc.async_writes_flushed<flush_opts>();
        pending = 0;
    }

    FORCE_INLINE void flush_if_pending() {
        if (pending != 0) {
            flush();
        }
    }

    // The sender does not gate on its own flow control, so reserve a slot first. The count is cached because
    // get_num_free_write_slots() invalidates the whole L1 data cache (a `fence` on Blackhole); this way we only
    // pay it when we actually run dry.
    FORCE_INLINE void reserve_slot() {
        while (free_slots == 0) {
            free_slots = sender->get_num_free_write_slots();
        }
        --free_slots;
    }

    FORCE_INLINE void send_header_only(volatile tt_l1_ptr PACKET_HEADER_TYPE* header) {
        reserve_slot();
        sender->template send_current_slot_stateful_non_blocking_from_address<fabric_use_posted_writes>(
            (uint32_t)header, sizeof(PACKET_HEADER_TYPE));
        ++pending;
    }

    FORCE_INLINE void send_with_payload(
        volatile tt_l1_ptr PACKET_HEADER_TYPE* header, uint32_t l1_addr, uint32_t size) {
        reserve_slot();
        sender->template send_current_slot_stateful_non_blocking<fabric_use_posted_writes>(
            l1_addr, size, (uint32_t)header);
        ++pending;
    }

    FORCE_INLINE volatile tt_l1_ptr PACKET_HEADER_TYPE* current_data_header() const { return data_headers[header_idx]; }

    // Send the packet built in the current ring header, then rotate. Wrapping flushes, so the header we come
    // back to is guaranteed to have left L1 before it is re-pointed.
    FORCE_INLINE void send_data_packet(uint32_t l1_addr, uint32_t size) {
        send_with_payload(current_data_header(), l1_addr, size);
        if (++header_idx == fabric_header_ring_size) {
            header_idx = 0;
            flush();
        }
    }

    FORCE_INLINE void send_scatter_packet(uint16_t size) {
        scatter_header.chunk_count = chunk_count;
        populate_unicast_scatter_write_fields<
            UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::PayloadSize>(
            current_data_header(), size, scatter_header);
        send_data_packet(start_l1_addr, size);
        chunk_count = 0;
    }

    const Noc& noc;
    SenderT* sender;  // direct or mux sender
    volatile tt_l1_ptr PACKET_HEADER_TYPE* data_headers[fabric_header_ring_size];
    volatile tt_l1_ptr PACKET_HEADER_TYPE* tail_header = nullptr;  // scatter straggler only
    volatile tt_l1_ptr PACKET_HEADER_TYPE* sem_header = nullptr;
    NocUnicastScatterCommandHeader scatter_header;
    uint32_t header_idx = 0;  // next ring header to build into
    uint32_t pending = 0;     // packets issued since the last flush (their headers may still be in flight)
    uint32_t free_slots = 0;  // cached sender slot credit
    uint8_t chunk_count;      // accumulated chunks not yet sent in a packet
    uint32_t start_l1_addr;   // start address of the accumulated contiguous chunks
};
