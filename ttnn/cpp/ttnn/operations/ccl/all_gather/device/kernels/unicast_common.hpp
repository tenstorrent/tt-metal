// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#include <cstdint>
#include <type_traits>

#include "gather_walk.hpp"

// Store-and-forward AllGather: every fabric send is a single 1-hop unicast to the neighbor.
// Runs on any effectively-1D topology (both Fabric 1D and 2D).
namespace fabric_api = tt::tt_fabric::linear::experimental;

////////////////////////////////////////////////////////////////
// data_valid semaphore protocol
//
// data_valid counts the chunks upstream has relayed into our output -- cumulative over the op. A chunk's
// absolute position is base_seqno + its seqno in the batch, with base_seqno = (iter-1) * slice_chunks +
// skip. The writer maintains the count (atomic-inc per chunks delivered); the reader waits on it with
// noc_semaphore_wait_min at the last chunk of each batch it reads, then a final wait for total_chunks.
//
// This semaphore is reused across cached invocations, so the reader subtracts total_chunks instead of
// clearing, which would drop credits posted by an upstream which is an invocation ahead. total_chunks
// must be exact: too many and a later reader passes early.
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
    //
    // Precondition: a packet has one payload from start_l1_addr, so its segments must be contiguous
    // in L1. The packed CB gives that; a CB with gaps would need a send() here.
    FORCE_INLINE void queue_segment(uint32_t l1_addr, uint64_t remote_noc_addr, uint32_t bytes) {
        ASSERT(chunk_count == 0 || l1_addr == start_l1_addr + payload);
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
    static_assert(NOC_SCATTER_WRITE_MIN_CHUNKS == 2, "send() covers the too-few-chunks case with one unicast write");

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
