// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/distributed/types.hpp"

#include "relay_affine_prefix_device_operation_types.hpp"

namespace ttnn::experimental::prim {

// Tile-column buffers per sender: later columns are read out of their head cores while earlier ones are on the link.
inline constexpr uint32_t relay_affine_prefix_sender_slots = 4;

// A fabric packet of whole FP32 tiles, up to three.
inline uint32_t relay_affine_prefix_packet_bytes() {
    constexpr uint32_t state_tile = tt::constants::TILE_HW * sizeof(float);
    const uint32_t payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    return std::min(3 * state_tile, payload / state_tile * state_tile);
}

// The packet headers a sender holds: one per packet of each slot's tile column.
inline uint32_t relay_affine_prefix_packet_headers(uint32_t key_tiles) {
    const uint32_t column_bytes = key_tiles * tt::constants::TILE_HW * sizeof(float);
    const uint32_t packet_bytes = relay_affine_prefix_packet_bytes();
    return relay_affine_prefix_sender_slots * ((column_bytes + packet_bytes - 1) / packet_bytes);
}

// Packet headers the fabric's pool gives each data-movement RISC (NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType in
// the architecture's dev_mem_map.h); the sender kernel pins the count with a static_assert.
inline uint32_t fabric_packet_headers_per_risc(tt::ARCH arch) { return arch == tt::ARCH::BLACKHOLE ? 12 : 8; }

// Legacy program descriptors, as select_final_carry's: the metal2 ProgramSpec cannot yet open fabric connections.
struct RelayAffinePrefixProgramFactory {
    // One program per coordinate: each sends to its own successor on the sequence-parallel ring.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const RelayAffinePrefixParams&,
        const RelayAffinePrefixInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
