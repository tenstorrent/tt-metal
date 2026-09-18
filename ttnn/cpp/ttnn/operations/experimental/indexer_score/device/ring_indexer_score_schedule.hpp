// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <cstdint>
#include <vector>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_id_sequencer.hpp"

namespace ttnn::operations::experimental::indexer_score::program::ring_schedule {

using ArrivalWaves = std::vector<std::vector<uint32_t>>;

struct RingWrites {
    uint32_t forward_writes_expected;
    uint32_t backward_writes_expected;
};

// Forward/backward all-gather writes expected for one device (mirrors ring_joint's build_ring_write_plan).
inline RingWrites ring_writes_for(uint32_t ring_size, uint32_t ring_index, ttnn::ccl::Topology topology) {
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ttnn::ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    (void)dynamic_alternate;
    if (topology == ttnn::ccl::Topology::Ring && (ring_index % 2 == 0)) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    if (topology == ttnn::ccl::Topology::Linear) {
        return {static_cast<uint32_t>(num_targets_backward), static_cast<uint32_t>(num_targets_forward)};
    }
    return {static_cast<uint32_t>(num_targets_forward), static_cast<uint32_t>(num_targets_backward)};
}

// Group physical shards by the order in which this rank can consume them. Wave 0 is local; true Rings with
// more than two ranks then have paired forward/backward waves and, for even ring sizes, one final opposite shard.
inline ArrivalWaves arrival_waves(uint32_t ring_size, uint32_t ring_index, RingWrites writes) {
    ArrivalWaves waves(ring_size / 2 + 1);
    RingIdSequencer seq(ring_index, ring_size, writes.backward_writes_expected, writes.forward_writes_expected);
    for (uint32_t iteration = 0; iteration < ring_size; ++iteration) {
        const uint32_t shard = seq.get_next_ring_id([](uint32_t, uint32_t) {});
        waves[(iteration + 1) / 2].push_back(shard);
    }
    return waves;
}

// Linear and Ring-2 do not have paired bidirectional arrival waves; retain their existing lane assignment.
inline bool rotation_enabled(ttnn::ccl::Topology topology, uint32_t ring_size) {
    return topology == ttnn::ccl::Topology::Ring && ring_size > 2;
}

}  // namespace ttnn::operations::experimental::indexer_score::program::ring_schedule
