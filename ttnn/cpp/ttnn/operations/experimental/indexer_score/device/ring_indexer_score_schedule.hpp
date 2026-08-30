// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_id_sequencer.hpp"

namespace ttnn::operations::experimental::indexer_score::program::ring_schedule {

using ArrivalWaves = std::vector<std::vector<uint32_t>>;
using WorkList = std::vector<std::vector<std::vector<uint32_t>>>;

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

inline ArrivalWaves arrival_waves(uint32_t ring_size, uint32_t ring_index, ttnn::ccl::Topology topology) {
    return arrival_waves(ring_size, ring_index, ring_writes_for(ring_size, ring_index, topology));
}

// Linear and Ring-2 do not have paired bidirectional arrival waves; retain their existing lane assignment.
inline bool rotation_enabled(ttnn::ccl::Topology topology, uint32_t ring_size) {
    return topology == ttnn::ccl::Topology::Ring && ring_size > 2;
}

inline uint32_t wave_column_shift(uint32_t wave, uint32_t wave_count, uint32_t column_count) {
    if (wave == 0 || wave + 1 == wave_count) {
        return 0;
    }
    // One virtual wave keeps the remote-wave span below a full column circle when waves fit, limiting disruption
    // to long DRAM streams. More waves than columns use the smallest possible nonzero stride and wrap naturally.
    const uint32_t column_stride = std::max(1u, column_count / (wave_count + 1));
    return (wave * column_stride) % column_count;
}

// Build each (row-block, column) lane's shard-major list of physical K-tile starts. Rotation changes only which
// column residue a lane owns in each arrival wave. It is a cyclic column permutation within each row block, so
// every KC unit remains in exactly one lane, adjacent row blocks retain their paired DRAM access pattern, and
// paired shards retain matching, monotonically increasing offsets.
inline WorkList make_work_list(
    const ArrivalWaves& waves,
    uint32_t units_per_shard,
    uint32_t tiles_per_shard,
    uint32_t k_tiles_per_unit,
    uint32_t num_blocks,
    uint32_t cols_used,
    bool rotate_waves) {
    WorkList work_list(num_blocks, std::vector<std::vector<uint32_t>>(cols_used));
    const uint32_t lane_count = num_blocks * cols_used;
    const uint32_t wave_count = static_cast<uint32_t>(waves.size());

    for (uint32_t block = 0; block < num_blocks; ++block) {
        for (uint32_t wave = 0; wave < wave_count; ++wave) {
            const uint32_t column_shift = rotate_waves ? wave_column_shift(wave, wave_count, cols_used) : 0;
            for (uint32_t col = 0; col < cols_used; ++col) {
                const uint32_t source_col = (col + cols_used - column_shift) % cols_used;
                const uint32_t source_lane = block + source_col * num_blocks;
                for (uint32_t unit = source_lane; unit < units_per_shard; unit += lane_count) {
                    for (uint32_t shard : waves[wave]) {
                        work_list[block][col].push_back(shard * tiles_per_shard + unit * k_tiles_per_unit);
                    }
                }
            }
        }
    }
    return work_list;
}

}  // namespace ttnn::operations::experimental::indexer_score::program::ring_schedule
