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
// Glossary (chunk, chunk id, global, seqno, lane, tile, run, segment, stripe) is in
// all_gather_unicast_factory.cpp.
//
// A run is chunks contiguous in memory, sent as one transfer. Where they are comes from
// TensorAccessor, so no layout is special-cased here or on the host: `stride` is the accessor's own
// chunk step between neighbours, and `xfer` is how many chunks fit one transfer.
//
// Long runs alone are not enough. Stepping by `stride` parks a worker in one DRAM bank, which costs
// more bandwidth than the runs win, so the walk is a tiled transpose: `xfer` chunks from one lane,
// then the next lane. Every fallback then falls out of `xfer` -- a padded page or a page-sized
// packet gives xfer == 1, which is plain ascending order.
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

// Chunks in one transfer: the most that fits both a packet and a single NOC command.
constexpr uint32_t chunks_per_transfer(uint32_t packet_size, uint32_t chunk_size) {
    const uint32_t cap = packet_size < NOC_MAX_BURST_SIZE ? packet_size : NOC_MAX_BURST_SIZE;
    return cap < chunk_size ? 1u : cap / chunk_size;
}

// Whether a tensor's contiguity lines up with the walk, so a run can hold more than one chunk.
struct RunSource {
    bool sub_page;    // chunks inside one page are consecutive seqnos
    bool cross_page;  // a run may carry on into the next page
};

// `packed` has to be false for a padded page: a run steps by the aligned page size while the CB is
// packed, so the pad would land in the payload.
FORCE_INLINE RunSource run_source(bool packed, uint32_t chunks_per_page, uint32_t page_stride, uint32_t stride) {
    if (!packed) {
        return {false, false};
    }
    const bool sub_page = chunks_per_page > 1 && stride == 1;
    return {sub_page, chunks_per_page > 1 ? (sub_page && page_stride == 1) : (page_stride == stride)};
}

// Chunks contiguous in memory from `chunk` on, stepping the way the walk steps, capped at
// `end_chunk`. Can reach past the current run, so the caller has to clip it with run_limit().
template <uint32_t chunks_per_page, typename Accessor>
FORCE_INLINE uint32_t contiguous_chunks(const Accessor& acc, RunSource src, uint32_t chunk, uint32_t end_chunk) {
    if constexpr (chunks_per_page == 1) {
        return src.cross_page ? acc.num_contiguous_pages(chunk, end_chunk) : 1u;
    } else {
        uint32_t n = src.sub_page ? chunks_per_page - chunk % chunks_per_page : 1u;
        if (src.cross_page) {
            const uint32_t end_page = (end_chunk + chunks_per_page - 1) / chunks_per_page;
            n += (acc.num_contiguous_pages(chunk / chunks_per_page, end_page) - 1) * chunks_per_page;
        }
        // end_page rounds up, so clip in chunks too. stride is 1 whenever a page holds several
        // chunks, so chunks left and seqnos left are the same number here.
        const uint32_t room = end_chunk - chunk;
        return n < room ? n : room;
    }
}

// The walk order plus the output's runs. Reader and writer build this the same way, so their walks
// cannot diverge.
struct WalkPlan {
    uint32_t stride;
    uint32_t xfer;
    RunSource out;
};

template <uint32_t chunks_per_page, uint32_t chunk_size, uint32_t xfer_max, typename Accessor>
FORCE_INLINE WalkPlan walk_plan(const Accessor& out) {
    const uint32_t page_stride = out.contiguous_page_stride();
    // Concat packs chunks inside one output page, so there neighbours are one chunk apart.
    const uint32_t stride = chunks_per_page > 1 ? 1u : page_stride;
    const bool packed = out.get_aligned_page_size() == chunks_per_page * chunk_size;
    return {stride, packed ? xfer_max : 1u, run_source(packed, chunks_per_page, page_stride, stride)};
}

// Walks a chunk id range as tiles of `xfer * stride`, reading each tile column-major: `xfer` chunks
// `stride` apart, then the next lane. One column is one run. `xfer == 1` is plain ascending order.
//
// The last tile is ragged -- its columns are `hb_` or `hb_ + 1` tall -- which is the only reason this
// holds state beyond the position.
class TiledWalk {
public:
    // Walk [first, first + count) from seqno `skip`. `stride` and `xfer` are at least 1.
    FORCE_INLINE void init(uint32_t first, uint32_t count, uint32_t skip, uint32_t stride, uint32_t xfer) {
        stride_ = stride;
        xfer_ = xfer;
        tile_ = xfer * stride;
        const uint32_t full = count / tile_;
        const uint32_t rem = count - full * tile_;
        tail_first_ = first + full * tile_;
        hb_ = rem / stride;
        he_ = rem - hb_ * stride;
        lane_ = 0;
        k_ = 0;
        if (count == 0) {
            tile_first_ = c_ = first;
            return;
        }
        if (skip < full * tile_) {
            const uint32_t tile = skip / tile_;
            const uint32_t in_tile = skip - tile * tile_;
            tile_first_ = first + tile * tile_;
            lane_ = in_tile / xfer;
            k_ = in_tile - lane_ * xfer;
        } else {
            // Lanes below he_ hold one extra chunk, so seqnos before lane r are r*hb_ + min(r, he_).
            tile_first_ = tail_first_;
            const uint32_t s = skip - full * tile_;
            const uint32_t tall = he_ * (hb_ + 1);
            if (s < tall) {
                lane_ = s / (hb_ + 1);
                k_ = s - lane_ * (hb_ + 1);
            } else {
                // hb_ > 0 here: at hb_ == 0 the tile holds only its taller lanes, so `s < tall`.
                const uint32_t rest = s - tall;
                lane_ = he_ + rest / hb_;
                k_ = rest - (lane_ - he_) * hb_;
            }
        }
        c_ = tile_first_ + lane_ + k_ * stride_;
    }

    FORCE_INLINE uint32_t chunk() const { return c_; }

    // Chunks left in this column, i.e. the longest run still allowed here.
    FORCE_INLINE uint32_t run_limit() const { return col() - k_; }

    FORCE_INLINE void advance(uint32_t n) {
        ASSERT(n != 0 && n <= run_limit());
        k_ += n;
        if (k_ < col()) {
            c_ += n * stride_;
            return;
        }
        k_ = 0;
        if (++lane_ == lanes()) {
            lane_ = 0;
            tile_first_ += tile_;
        }
        c_ = tile_first_ + lane_;
    }

private:
    FORCE_INLINE bool in_tail() const { return tile_first_ == tail_first_; }
    FORCE_INLINE uint32_t col() const { return in_tail() ? hb_ + (lane_ < he_ ? 1u : 0u) : xfer_; }
    // Empty columns can only be a tail of lanes, so a lane step never has to skip one.
    FORCE_INLINE uint32_t lanes() const { return (in_tail() && hb_ == 0) ? he_ : stride_; }

    uint32_t c_, stride_, xfer_, tile_, lane_, k_, tile_first_, tail_first_, hb_, he_;
};

// A chunk's index in this device's contribution -> its index in the output tensor. Between rows the
// output holds the other devices' stripes, so a run stops at the row edge.
template <uint32_t chunks_per_stripe, uint32_t num_devices>
class StripeMap {
public:
    struct Pos {
        uint32_t global;   // chunk index in the output tensor
        uint32_t row_end;  // one past this row's last chunk of our stripe
    };

    FORCE_INLINE void init(uint32_t stripe) { offset_ = stripe * chunks_per_stripe; }

    FORCE_INLINE Pos at(uint32_t local) const {
        const uint32_t row = local / chunks_per_stripe;
        const uint32_t base = row * (num_devices * chunks_per_stripe) + offset_;
        return {base + local - row * chunks_per_stripe, base + chunks_per_stripe};
    }

private:
    uint32_t offset_;
};

// A chunk index split into the page holding it and the byte offset inside that page.
template <uint32_t chunks_per_page>
FORCE_INLINE uint32_t page_of(uint32_t chunk) {
    return chunk / chunks_per_page;
}

template <uint32_t chunks_per_page, uint32_t chunk_size>
FORCE_INLINE uint32_t byte_off_of(uint32_t chunk) {
    return (chunk % chunks_per_page) * chunk_size;
}

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
