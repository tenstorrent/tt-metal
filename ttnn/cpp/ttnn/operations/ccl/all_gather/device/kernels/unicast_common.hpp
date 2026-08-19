// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#include <cstdint>
#include <type_traits>

// Store-and-forward AllGather: every fabric send is a single 1-hop unicast to the neighbor.
// Runs on any effectively-1D topology (both Fabric 1D and 2D).
namespace fabric_api = tt::tt_fabric::linear::experimental;

////////////////////////////////////////////////////////////////
// Runs
//
// Glossary (chunk, chunk id, global, seqno, lane, run, segment, stripe) is in
// all_gather_unicast_factory.cpp.
//
// Runs come from TensorAccessor::num_contiguous_pages, so no layout is special-cased here or on the
// host. They must be stepped by the accessor's own contiguous_page_stride(); any other step lands in
// a different bank or shard. Reading the stride from the accessor rather than a compile arg also
// makes every device agree on the seqno order for free.
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
// data_valid semaphore protocol
//
// data_valid counts the chunks upstream has relayed into our output -- cumulative over the op, reset at
// completion. A chunk's absolute position is base_seqno + its seqno in the batch, with base_seqno =
// (iter-1) * slice_chunks + skip. The writer maintains the count (atomic-inc per chunks delivered); the
// reader waits on it with noc_semaphore_wait_min at the last chunk of each batch it reads, then a final
// wait for total_chunks before reset.
//
// Waiting on an absolute position (not a signal count) lets one reader path cover every case with no
// alignment or per-topology special-casing:
//   - full relay, and even-ring split prefix half (skip 0) / suffix half (skip = half): same per-batch
//     wait, differing only in base_seqno/take;
//   - sink stripe (a line endpoint's incoming, or a ring antipode): no relay wait, covered by the final
//     total_chunks wait;
//   - sink direction (num_iters == 0): only the total_chunks wait runs.
// So data_valid_granularity is a pure writer-side perf knob: the reader auto-paces to the writer's cadence.
////////////////////////////////////////////////////////////////

// Walks the chunks of one stripe in stride order, in runs. A stripe's src address on this device equals
// its dst address on the neighbor, so reader and writer share this walk unchanged.
//
// Position is held once, as a chunk id; page and byte offset are derived from it. Only the row bias is
// carried forward, and only across a row boundary, so there is no incremental copy of the mapping that
// could drift from the closed form in seat().
template <
    uint32_t output_chunks_per_stripe,
    uint32_t output_chunks_per_page,
    uint32_t output_chunk_size,
    uint32_t num_devices>
class StripeWalk {
    static constexpr uint32_t row_bias_step = (num_devices - 1) * output_chunks_per_stripe;

public:
    // Walk `stripe` over chunk ids [start, start + count), emitting seqnos [skip, skip + take).
    // The order is defined over the whole range even when we emit part of it, because a seqno has to
    // mean the same thing on the device that sent these chunks -- that is what data_valid counts.
    FORCE_INLINE void init(
        uint32_t stripe, uint32_t start, uint32_t count, uint32_t skip, uint32_t take, uint32_t stride) {
        stripe_ = stripe;
        start_ = start;
        end_ = start + count;
        take_ = take;
        stride_ = stride;
        emitted_ = 0;
        // Lane r holds ceil((count - r) / stride) chunks, so finding the one holding `skip` is a
        // stride-long scan. Runs once per iteration, never per chunk.
        uint32_t lane = 0;
        uint32_t before = 0;
        for (; lane + 1 < stride; ++lane) {
            const uint32_t in_lane = (count > lane) ? (count - lane + stride - 1) / stride : 0;
            if (before + in_lane > skip) {
                break;
            }
            before += in_lane;
        }
        lane_ = lane;
        seat(start + lane + (skip - before) * stride);
    }

    FORCE_INLINE bool valid() const { return emitted_ < take_; }

    FORCE_INLINE uint32_t chunk_id() const { return c_; }
    FORCE_INLINE uint32_t page_id() const { return global() / output_chunks_per_page; }
    FORCE_INLINE uint32_t byte_off() const { return (global() % output_chunks_per_page) * output_chunk_size; }
    FORCE_INLINE uint32_t seqnos_left() const { return take_ - emitted_; }

    // Chunks left in this stripe. A run clipped here keeps advance() inside one row.
    FORCE_INLINE uint32_t chunks_to_stripe_end() const { return (stripe_end() - c_ + stride_ - 1) / stride_; }

    // One past the stripe's last page, for num_contiguous_pages to clip against.
    FORCE_INLINE uint32_t end_page_id() const {
        if constexpr (output_chunks_per_page == 1) {
            return stripe_end() + bias_;
        } else {
            return (stripe_end() + bias_ + output_chunks_per_page - 1) / output_chunks_per_page;
        }
    }

    // `n` must not exceed chunks_to_stripe_end().
    FORCE_INLINE void advance(uint32_t n) {
        emitted_ += n;
        if (emitted_ == take_) {
            return;  // done -- do not step outside the range
        }
        c_ += n * stride_;
        if (c_ >= end_) {
            seat(start_ + ++lane_);  // this lane is done, restart on the next
        } else if (c_ >= row_end_) {
            row_end_ += output_chunks_per_stripe;
            bias_ += row_bias_step;
        }
    }

private:
    FORCE_INLINE uint32_t global() const { return c_ + bias_; }
    FORCE_INLINE uint32_t stripe_end() const { return row_end_ < end_ ? row_end_ : end_; }

    // The only closed-form site. Runs once per init and once per lane, never per chunk.
    FORCE_INLINE void seat(uint32_t c) {
        const uint32_t row = c / output_chunks_per_stripe;
        c_ = c;
        row_end_ = (row + 1) * output_chunks_per_stripe;
        bias_ = row * row_bias_step + stripe_ * output_chunks_per_stripe;
    }

    uint32_t c_, bias_, row_end_;
    uint32_t stripe_, start_, end_, take_, emitted_, lane_, stride_;
};

// Unicasts segments one hop to the single neighbor. Packs segments into a packet until either the
// payload or the scatter-chunk count runs out, and sends the one-segment case as a unicast write --
// which costs the receiving ERISC a single NoC command instead of one per segment.
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
        constexpr uint8_t num_hops = 1;  // store-and-forward: always the immediate neighbor

        // Addresses and sizes both vary per packet, so set_state only fixes the route.
        fabric_api::fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
            scatter_packet_header, num_hops);

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
        ASSERT(chunk_count == 0);  // outstanding segments! flush_packet_and_wait() not called correctly
    }

    // Increment a semaphore on the neighbor.
    void atomic_inc(uint64_t addr, uint32_t val) {
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
            UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
            sender, sem_packet_header, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{addr, val});
    }

    // A segment that does not fit starts a new packet rather than spilling into this one: splitting it
    // would fill the tail but cost an extra scatter chunk, i.e. an extra NoC write at the receiver.
    FORCE_INLINE void queue_segment(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        // Only a chunk larger than a packet gets here; the caller caps runs at packet_size.
        while (bytes > packet_size) {
            send();
            push(l1_addr, remote_noc_addr, packet_size);
            send();
            l1_addr += packet_size;
            remote_noc_addr += packet_size;
            bytes -= packet_size;
        }
        if (chunk_count == max_chunks || payload + bytes > packet_size) {
            send();
        }
        push(l1_addr, remote_noc_addr, bytes);
    }

    // Call this before popping a CB entry: a queued packet still points into it.
    void flush_packet_and_wait() {
        send();
        noc.async_writes_flushed();
    }

private:
    static constexpr uint32_t max_chunks = NOC_SCATTER_WRITE_MAX_CHUNKS;
    static_assert(packet_size <= 0xFFFF, "NocUnicastScatterCommandHeader::chunk_size is uint16_t");

    FORCE_INLINE void push(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        if (chunk_count == 0) {
            start_l1_addr = l1_addr;
        }
        // Only the first max_chunks-1 sizes travel; the last one is implied by the payload size.
        if (chunk_count < max_chunks - 1) {
            scatter_header.chunk_size[chunk_count] = static_cast<uint16_t>(bytes);
        }
        scatter_header.noc_address[chunk_count++] = remote_noc_addr;
        payload += bytes;
    }

    void send() {
        if (chunk_count == 0) {
            return;
        }
        noc.async_writes_flushed();
        if (chunk_count == 1) {
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
    uint8_t chunk_count;     // segments queued for the current packet
    uint32_t payload;        // bytes queued for the current packet
    uint32_t start_l1_addr;  // start of the queued segments, contiguous in L1
};
