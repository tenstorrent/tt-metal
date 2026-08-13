// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#include "run_coalescer.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

// Store-and-forward AllGather: every fabric send is a single 1-hop unicast to the neighbor.
// Runs on any effectively-1D topology (both Fabric 1D and 2D).
namespace fabric_api = tt::tt_fabric::linear::experimental;

////////////////////////////////////////////////////////////////
// Runs, not pages
//
// A packet used to carry at most NOC_SCATTER_WRITE_MAX_CHUNKS pages, one scatter chunk each, which
// left most of the payload empty. Instead we send *runs*: a scatter chunk covers as many chunks as
// happen to be adjacent at the destination, so a packet is "up to 4 runs" and a run that fills the
// payload becomes a plain unicast write (one NoC command on the receiving ERISC instead of one per
// page). RunCoalescer discovers the runs; nothing here knows anything about tensor layout.
//
// The only layout input is `walk_stride`, the chunk-index step the host believes lands on adjacent
// destinations (num_banks for interleaved, 1 for sharded, and so on). It is a hint: a bad value
// costs merge opportunities, never correctness.
//
// Filling the packet moves the bottleneck onto the workers, so the host also scales worker count
// with the chunk rate -- the two changes only pay off together.
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
// data_valid semaphore protocol
//
// data_valid counts the chunks upstream has relayed into our output -- cumulative over the op, reset at
// completion. A chunk's absolute position is base_chunk + its position in the walk, with base_chunk =
// (iter-1) * slice_count. Note "position in the walk", not chunk index: they differ once walk_stride > 1,
// which is why the even-ring split is handed to the iterator as a walk range (skip/take) rather than a
// chunk-index interval. The writer maintains the count (atomic-inc per chunks delivered); the reader waits on it with
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

// Walks the output-tensor chunks of one stripe, in strides of `walk_stride`: first the chunks at
// offsets 0, w, 2w..., then those at 1, 1+w..., until every chunk of the range is covered. Templated
// on the geometry so per-stripe starts are computed here and the matched/split fast path
// (output_chunks_per_page == 1) folds away. Re-pointed to any stripe each relay iteration via init().
// A stripe's src address on this device equals its dst address on the neighbor, so reader and writer
// share this iterator unchanged.
template <
    uint32_t output_chunks_per_stripe,
    uint32_t output_chunks_per_page,
    uint32_t output_chunk_size,
    uint32_t num_devices,
    uint32_t walk_stride>
class OutputStripeIterator {
    static constexpr uint32_t output_page_size = output_chunks_per_page * output_chunk_size;
    static constexpr uint32_t stripe_distance_chunks = num_devices * output_chunks_per_stripe;
    static constexpr uint32_t output_pages_per_row = stripe_distance_chunks / output_chunks_per_page;
    // Wrapping into the next row costs the rest of the row plus the step itself. At most one wrap
    // per step, because the host caps walk_stride at the stripe length.
    static constexpr uint32_t stripe_wrap_step = (num_devices - 1) * output_chunks_per_stripe + walk_stride;

    static_assert(walk_stride >= 1 && walk_stride <= output_chunks_per_stripe, "one wrap per step at most");
    // Concat already has adjacent chunks side by side inside an output page; striding would skip them.
    static_assert(output_chunks_per_page == 1 || walk_stride == 1);

public:
    // `index` is the chunk's position in this device's chunk space; the reader maps it back to an
    // input page on iteration 0.
    struct Chunk {
        uint32_t index;
        uint32_t page_id;
        uint32_t byte_off;
    };

    // Walk `stripe` over the chunk range [start, start + count), emitting only walk positions
    // [skip, skip + take). The order is always defined over the whole range even when we emit part
    // of it, because a walk position has to mean the same thing on the device that sent these
    // chunks -- that is what data_valid counts.
    FORCE_INLINE void init(uint32_t stripe, uint32_t start, uint32_t count, uint32_t skip, uint32_t take) {
        if constexpr (output_chunks_per_page > 1) {
            // In concat mode the page phase (and hence the stripe jump) depends on the stripe.
            const uint32_t off = (stripe * output_chunks_per_stripe) % output_chunks_per_page;
            phase_ = off * output_chunk_size;
            stripe_jump_ = output_pages_per_row - (off + output_chunks_per_stripe - 1) / output_chunks_per_page;
        }
        stripe_ = stripe;
        start_ = start;
        end_ = start + count;
        count_ = take;
        sent_ = 0;
        seek_position(skip, count);
    }

    FORCE_INLINE bool valid() const { return sent_ < count_; }

    // Return the current chunk, then advance.
    FORCE_INLINE Chunk next() {
        const Chunk chunk{index_, page_id_, byte_off_};
        if (++sent_ == count_) {
            return chunk;  // done -- skip the advance so we can never step outside the range
        }
        index_ += walk_stride;
        if (index_ < end_) {
            advance();
        } else {
            seek(start_ + ++residue_);  // this residue class is done, restart on the next
        }
        return chunk;
    }

private:
    // Jump to a walk position. Residue r holds ceil((count - r) / walk_stride) chunks, so finding
    // the one containing `pos` is a walk_stride-long scan -- once per iteration, never per chunk.
    FORCE_INLINE void seek_position(uint32_t pos, uint32_t count) {
        uint32_t residue = 0;
        uint32_t before = 0;
        if constexpr (walk_stride > 1) {
            while (true) {
                const uint32_t in_residue =
                    (count > residue) ? (count - residue + walk_stride - 1) / walk_stride : 0;
                if (before + in_residue > pos) {
                    break;
                }
                before += in_residue;
                ++residue;
            }
        }
        residue_ = residue;
        seek(start_ + residue + (pos - before) * walk_stride);
    }

    // Locate an arbitrary chunk. Runs once per init and once per residue class -- never per chunk.
    FORCE_INLINE void seek(uint32_t index) {
        const uint32_t s_start = (index / output_chunks_per_stripe) * stripe_distance_chunks +
                                 (index % output_chunks_per_stripe) + stripe_ * output_chunks_per_stripe;
        index_ = index;
        page_id_ = s_start / output_chunks_per_page;
        byte_off_ = (s_start % output_chunks_per_page) * output_chunk_size;
        chunk_in_stripe_ = index % output_chunks_per_stripe;
    }

    // Step walk_stride chunks forward from the current position.
    FORCE_INLINE void advance() {
        if constexpr (output_chunks_per_page == 1) {
            chunk_in_stripe_ += walk_stride;
            if (chunk_in_stripe_ >= output_chunks_per_stripe) {
                chunk_in_stripe_ -= output_chunks_per_stripe;
                page_id_ += stripe_wrap_step;
            } else {
                page_id_ += walk_stride;
            }
        } else {
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
        }
    }

    uint32_t index_, page_id_, byte_off_, chunk_in_stripe_;
    uint32_t stripe_, start_, end_, count_, sent_, residue_;
    uint32_t phase_ = 0, stripe_jump_ = 0;  // concat only
};

// Unicasts runs one hop to the single neighbor. Packs runs into packets until either the payload or
// the scatter chunk count is exhausted, and sends the one-chunk case as a unicast write.
//
// Templated on the sender type (SenderT*) so the same writer drives either a direct WorkerToFabricEdmSender
// (one worker per direction) or a FabricMuxV2Sender (workers sharing a fabric mux). The send calls accept
// either (see CheckFabricSenderType in api_common.h), so no route-manager is needed -- which is also why this
// class routes its own headers.
template <uint32_t packet_size, typename SenderT>
class FabricWriter {
public:
    FabricWriter(const Noc& noc, SenderT* sender, uint16_t neighbor_chip_id, uint16_t neighbor_mesh_id) :
        noc{noc},
        sender{sender},
        scatter_packet_header{PacketHeaderPool::allocate_header(1)},
        unicast_packet_header{PacketHeaderPool::allocate_header(1)},
        sem_packet_header{PacketHeaderPool::allocate_header(1)},
        scatter_header({}, {}),
        chunk_count{0},
        payload{0} {
        // Addresses and sizes both vary per packet now, so set_state only fixes the route; the dummy
        // sizes exist because the header rejects zero-length chunks.
        std::array<uint64_t, max_chunks> dummy_addrs{};
        std::array<uint16_t, max_chunks - 1> dummy_sizes{};
        dummy_sizes.fill(1);
        constexpr uint8_t num_hops = 1;  // store-and-forward: always the immediate neighbor

        fabric_api::fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
            scatter_packet_header,
            num_hops,
            NocUnicastScatterCommandHeader(dummy_addrs.data(), dummy_sizes.data(), max_chunks));

        fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(
            unicast_packet_header, num_hops);

        // One atomic-inc header for the "alive" barrier inc + data_valid signals; Flush orders it after the
        // payload it announces.
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            sem_packet_header, num_hops, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});

        // For Fabric_2D, set_state() sets routes only in its RoutingPlaneConnectionManager overloads, so set
        // them here. Keyed on the header type since the FABRIC_2D define is absent on the mux path.
        if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>) {
            using MeshHeader = volatile tt::tt_fabric::HybridMeshPacketHeader*;
            fabric_set_unicast_route(
                reinterpret_cast<MeshHeader>(scatter_packet_header), neighbor_chip_id, neighbor_mesh_id);
            fabric_set_unicast_route(
                reinterpret_cast<MeshHeader>(unicast_packet_header), neighbor_chip_id, neighbor_mesh_id);
            fabric_set_unicast_route(
                reinterpret_cast<MeshHeader>(sem_packet_header), neighbor_chip_id, neighbor_mesh_id);
        }
    }

    ~FabricWriter() {
        ASSERT(chunk_count == 0);  // outstanding chunks! flush() not called correctly
    }

    // Increment a semaphore on the neighbor.
    void atomic_inc(uint64_t addr, uint32_t val) {
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
            UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
            sender, sem_packet_header, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{addr, val});
    }

    // `bytes` is one destination-contiguous run; the caller caps it at packet_size.
    //
    // A run that does not fit starts a new packet rather than spilling into this one. Splitting it
    // would fill the tail but cost an extra scatter chunk, i.e. an extra NoC write at the receiver,
    // and that measures more expensive than the bytes it recovers.
    FORCE_INLINE void async_write(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        ASSERT(bytes <= packet_size);
        if (chunk_count == max_chunks || payload + bytes > packet_size) {
            send();
        }
        if (chunk_count == 0) {
            start_l1_addr = l1_addr;
        }
        // Only the first max_chunks-1 sizes travel; the last chunk's is implied by the payload size.
        if (chunk_count < max_chunks - 1) {
            scatter_header.chunk_size[chunk_count] = static_cast<uint16_t>(bytes);
        }
        scatter_header.noc_address[chunk_count++] = remote_noc_addr;
        payload += bytes;
    }

    // Call this before popping CB entry
    void async_writes_flushed() {
        send();
        // Wait for Fabric writes to be sent out before popping CB entry
        noc.async_writes_flushed();
    }

private:
    static constexpr uint32_t max_chunks = NOC_SCATTER_WRITE_MAX_CHUNKS;
    static_assert(packet_size <= 0xFFFF, "NocUnicastScatterCommandHeader::chunk_size is uint16_t");

    void send() {
        if (chunk_count == 0) {
            return;
        }
        noc.async_writes_flushed();
        if (chunk_count == 1) {
            // One run filling the packet: the receiving ERISC issues a single NoC write for it.
            fabric_api::fabric_unicast_noc_unicast_write_with_state<
                UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                sender,
                unicast_packet_header,
                start_l1_addr,
                tt::tt_fabric::NocUnicastCommandHeader{scatter_header.noc_address[0]},
                payload);
        } else {
            scatter_header.chunk_count = chunk_count;
            fabric_api::fabric_unicast_noc_scatter_write_with_state<
                UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                UnicastScatterWriteUpdateMask::PayloadSize>(
                sender, scatter_packet_header, start_l1_addr, scatter_header, payload);
        }
        chunk_count = 0;
        payload = 0;
    }

    const Noc& noc;
    SenderT* sender;  // direct or mux sender
    volatile tt_l1_ptr PACKET_HEADER_TYPE* scatter_packet_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* unicast_packet_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* sem_packet_header;
    NocUnicastScatterCommandHeader scatter_header;
    uint8_t chunk_count;     // runs queued for the current packet
    uint32_t payload;        // bytes queued for the current packet
    uint32_t start_l1_addr;  // start address of the queued runs, contiguous in L1
};
