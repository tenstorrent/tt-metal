// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

#include <cstdint>
#include <type_traits>

#include "chunk_packets.hpp"

// Store-and-forward all_gather: every fabric send is a single 1-hop unicast to the neighbor.
// Runs on any effectively-1D topology (both Fabric 1D and 2D).
namespace fabric_api = tt::tt_fabric::linear::experimental;

////////////////////////////////////////////////////////////////
// Relaying, and the data_valid semaphore
//
// Glossary:
//   hop          -- one relay step. Iteration i sends the stripe we got on iteration i-1.
//   sink         -- a stripe consumed here, not relayed on (a line endpoint's incoming, or a ring
//                   antipode). A sink *direction* relays nothing at all.
//   signal_every -- entries per data_valid signal. Smaller pipelines finer, larger syncs less.
//   relayed      -- how many of our leading sends the downstream will itself relay.
//
// data_valid counts the chunks upstream has relayed into our output -- cumulative over the op. A
// chunk's absolute position is base + its place in the batch, with base = (hop-1) * slice_chunks +
// skip. The writer keeps the count (atomic-inc per chunks delivered); the reader waits on it with
// noc_semaphore_wait_min at the last chunk of each batch it reads, then a final wait for
// total_chunks.
//
// This semaphore is reused across cached invocations, so the reader subtracts total_chunks instead
// of clearing, which would drop credits posted by an upstream which is an invocation ahead.
// total_chunks must be exact: too many and a later reader passes early.
//
// Waiting on an absolute position (not a signal count) lets one reader path cover every case with
// no alignment or per-topology special-casing:
//   - full relay, and even-ring split prefix half (skip 0) / suffix half (skip = half): same
//     per-batch wait, differing only in base/take;
//   - sink stripe: no relay wait, covered by the final total_chunks wait;
//   - sink direction (num_hops == 0): only the total_chunks wait runs.
// So signal_every is a pure writer-side perf knob: the reader auto-paces to the writer's cadence.
////////////////////////////////////////////////////////////////

// Sends one hop to the single neighbor. Gives Packer its two writes, and owns the routes and
// headers behind them.
//
// Templated on the sender type (SenderT*) so the same code drives either a direct
// WorkerToFabricEdmSender (one worker per direction) or a FabricMuxV2Sender (workers sharing a
// fabric mux). The send calls accept either (see CheckFabricSenderType in api_common.h), so no
// route-manager is needed -- which is also why this routes its own headers.
template <typename SenderT>
class UnicastSender {
public:
    UnicastSender(SenderT* sender, uint16_t neighbor_chip_id, uint16_t neighbor_mesh_id) :
        sender{sender},
        scatter_header{PacketHeaderPool::allocate_header(1)},
        unicast_header{PacketHeaderPool::allocate_header(1)},
        sem_header{PacketHeaderPool::allocate_header(1)} {
        constexpr uint8_t num_hops = 1;  // store-and-forward: always the immediate neighbor

        // Addresses and sizes both vary per packet, so set_state only fixes the route.
        fabric_api::fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
            scatter_header, num_hops);

        fabric_api::fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::None>(unicast_header, num_hops);

        // One atomic-inc header for the "alive" barrier inc + data_valid signals; Flush orders it
        // after the payload it announces.
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            sem_header, num_hops, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});

        // For Fabric_2D, set_state() sets routes only in its RoutingPlaneConnectionManager
        // overloads, so set them here. Keyed on the header type since the FABRIC_2D define is
        // absent on the mux path.
        if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>) {
            using MeshHeader = volatile tt::tt_fabric::HybridMeshPacketHeader*;
            fabric_set_unicast_route(reinterpret_cast<MeshHeader>(scatter_header), neighbor_chip_id, neighbor_mesh_id);
            fabric_set_unicast_route(reinterpret_cast<MeshHeader>(unicast_header), neighbor_chip_id, neighbor_mesh_id);
            fabric_set_unicast_route(reinterpret_cast<MeshHeader>(sem_header), neighbor_chip_id, neighbor_mesh_id);
        }
    }

    FORCE_INLINE void write_one(uint32_t l1_addr, uint64_t dst, uint32_t bytes) {
        fabric_api::fabric_unicast_noc_unicast_write_with_state<
            UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
            sender, unicast_header, l1_addr, tt::tt_fabric::NocUnicastCommandHeader{dst}, bytes);
    }

    FORCE_INLINE void write_scatter(uint32_t l1_addr, NocUnicastScatterCommandHeader& header, uint32_t payload) {
        fabric_api::fabric_unicast_noc_scatter_write_with_state<
            UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
            UnicastScatterWriteUpdateMask::PayloadSize>(sender, scatter_header, l1_addr, header, payload);
    }

    // Increment a semaphore on the neighbor.
    void atomic_inc(uint64_t addr, uint32_t val) {
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_with_state<
            UnicastAtomicIncUpdateMask::DstAddr | UnicastAtomicIncUpdateMask::Val>(
            sender, sem_header, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{addr, val});
    }

private:
    SenderT* sender;  // direct or mux sender
    volatile tt_l1_ptr PACKET_HEADER_TYPE* scatter_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* unicast_header;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* sem_header;
};
