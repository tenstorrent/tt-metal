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

// A fabric packet: the fabric's whole payload (16 B aligned), up to three FP32 tiles. A relay unit is a contiguous
// tile column, so its packets need not hold whole tiles: the fewer the packets, the faster the relay's stream.
inline uint32_t relay_chunk_scan_packet_bytes() {
    constexpr uint32_t state_tile = tt::constants::TILE_HW * sizeof(float);
    const uint32_t payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    return std::min(3 * state_tile, payload / 16 * 16);
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
// of their relay cores while earlier ones are on the link. Up to four, as many as the header pool holds.
inline uint32_t relay_chunk_scan_sender_slots(uint32_t key_tiles, tt::ARCH arch) {
    return std::min(4u, fabric_packet_headers_per_risc(arch) / relay_chunk_scan_column_packets(key_tiles));
}

// Heads one sender serves: each needs its own count semaphore on the sender core, and the fabric connection takes two
// more of the core's 16 (tt::tt_metal::NUM_SEMAPHORES, which no public header exposes).
inline constexpr uint32_t relay_chunk_scan_sender_semaphores = 16;
inline constexpr uint32_t relay_chunk_scan_connection_semaphores = 2;
inline constexpr uint32_t relay_chunk_scan_max_heads_per_link =
    relay_chunk_scan_sender_semaphores - relay_chunk_scan_connection_semaphores;
// The fabric connections a sender may open (the fewest RoutingPlaneConnectionManager::MaxConnections allows).
inline constexpr uint32_t relay_chunk_scan_max_connections = 4;

// The relay's hop runs on relay cores, each serving up to this many heads of one link, as far apart in the link's
// order as the link's relay cores allow: a relay core's heads then arrive one at a time.
inline constexpr uint32_t relay_chunk_scan_heads_per_relay_core = 2;

// Head h's relay core and its slot there (its landing and next-state buffers): the relay cores are link 0's, then
// link 1's, ...; the i-th head of a link (h = link + i * num_links) goes to its link's relay core i % cores, slot
// i / cores.
struct RelaySlot {
    uint32_t core;
    uint32_t slot;
};
// The relay cores of a link, and of links [0, links).
inline uint32_t relay_chunk_scan_link_relay_cores(uint32_t heads, uint32_t num_links, uint32_t link) {
    constexpr uint32_t per_core = relay_chunk_scan_heads_per_relay_core;
    return link < heads ? ((heads - link + num_links - 1) / num_links + per_core - 1) / per_core : 0;
}
inline uint32_t relay_chunk_scan_relay_cores(uint32_t heads, uint32_t num_links, uint32_t links) {
    uint32_t cores = 0;
    for (uint32_t l = 0; l < links; ++l) {
        cores += relay_chunk_scan_link_relay_cores(heads, num_links, l);
    }
    return cores;
}
inline uint32_t relay_chunk_scan_relay_cores(uint32_t heads, uint32_t num_links) {
    return relay_chunk_scan_relay_cores(heads, num_links, num_links);
}
inline RelaySlot relay_chunk_scan_relay_slot(uint32_t head, uint32_t heads, uint32_t num_links) {
    const uint32_t link = head % num_links;
    const uint32_t index = head / num_links;
    const uint32_t cores = relay_chunk_scan_link_relay_cores(heads, num_links, link);
    return {relay_chunk_scan_relay_cores(heads, num_links, link) + index % cores, index / cores};
}

// Heads whose correction cores share a grid column (kernels/correction_core.hpp).
inline uint32_t relay_chunk_scan_heads_per_column(uint32_t grid_y, uint32_t first_row, uint32_t cores_per_head) {
    return (grid_y - first_row) / cores_per_head;
}

// The cores of each head: the op's cores are the heads' correction cores, head by head, then two senders per link,
// then the relay cores.
inline uint32_t relay_chunk_scan_cores_per_head(size_t cores, uint32_t heads, uint32_t num_links) {
    return (cores - 2 * num_links - relay_chunk_scan_relay_cores(heads, num_links)) / heads;
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
