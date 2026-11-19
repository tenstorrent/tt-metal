// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/distributed/types.hpp"

#include "relay_chunk_scan_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// A fabric packet of whole FP32 tiles, up to three.
inline uint32_t relay_chunk_scan_packet_bytes() {
    constexpr uint32_t state_tile = tt::constants::TILE_HW * sizeof(float);
    const uint32_t payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    return std::min(3 * state_tile, payload / state_tile * state_tile);
}

// The packets of one tile column of a head's state.
inline uint32_t relay_chunk_scan_column_packets(uint32_t key_tiles) {
    const uint32_t column_bytes = key_tiles * tt::constants::TILE_HW * sizeof(float);
    const uint32_t packet_bytes = relay_chunk_scan_packet_bytes();
    return (column_bytes + packet_bytes - 1) / packet_bytes;
}

// Packet headers the fabric's pool gives each data-movement RISC (NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType in
// the architecture's dev_mem_map.h); the sender kernel pins the count with a static_assert.
inline uint32_t fabric_packet_headers_per_risc(tt::ARCH arch) { return arch == tt::ARCH::BLACKHOLE ? 12 : 8; }

// Tile-column buffers per sender, each holding one packet header per packet of its column: later columns are read out
// of their head cores while earlier ones are on the link. Up to four, as many as the header pool holds.
inline uint32_t relay_chunk_scan_sender_slots(uint32_t key_tiles, tt::ARCH arch) {
    return std::min(4u, fabric_packet_headers_per_risc(arch) / relay_chunk_scan_column_packets(key_tiles));
}

// Heads one sender serves: each needs its own count semaphore on the sender core, and the fabric connection takes
// two more of the core's 16 (tt::tt_metal::NUM_SEMAPHORES, which no public header exposes).
inline constexpr uint32_t relay_chunk_scan_sender_semaphores = 16;
inline constexpr uint32_t relay_chunk_scan_connection_semaphores = 2;
inline constexpr uint32_t relay_chunk_scan_max_heads_per_link =
    relay_chunk_scan_sender_semaphores - relay_chunk_scan_connection_semaphores;
// The fabric connections a sender may open (the fewest RoutingPlaneConnectionManager::MaxConnections allows).
inline constexpr uint32_t relay_chunk_scan_max_connections = 4;

// The cores of each head: the op's cores are the heads' cores, head by head, then two senders per link.
// The cores are each head's cores, then two senders per link, then a helper per head.
inline uint32_t relay_chunk_scan_cores_per_head(size_t cores, uint32_t heads, uint32_t num_links) {
    return (cores - 2 * num_links - heads) / heads;
}

// Legacy program descriptors: the metal2 ProgramSpec cannot yet open fabric connections.
struct RelayChunkScanProgramFactory {
    // One program per coordinate: each sends to its own successor along the sequence-parallel axis. They differ only
    // in runtime args (sp_rank, the successor, its route, the carry's ranges and fabric connections), so a range runs
    // them as one program with per-device runtime args.
    static constexpr bool merge_per_device_programs = true;
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const RelayChunkScanParams&,
        const RelayChunkScanInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
