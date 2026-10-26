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

// The fabric packet a tile column of a head's state is sent in: whole FP32 tiles, the column split as evenly as the
// fewest packets the payload allows (a 4-tile column as 2 + 2 tiles, not 3 + 1: equal packets pipeline through the
// routers' store-and-forward stages, so the column's last byte lands sooner). The device operation checks that one tile
// fits the payload.
inline uint32_t relay_affine_prefix_packet_bytes(uint32_t key_tiles) {
    constexpr uint32_t state_tile = tt::constants::TILE_HW * sizeof(float);
    const uint32_t max_tiles = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / state_tile;
    const uint32_t packets = (key_tiles + max_tiles - 1) / max_tiles;
    return (key_tiles + packets - 1) / packets * state_tile;
}

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
