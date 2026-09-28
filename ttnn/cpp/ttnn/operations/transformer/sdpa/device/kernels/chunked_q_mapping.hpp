// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Absolute positions of the two contiguous segments packed into one device's Q slab.
struct ChunkedQMapping {
    uint32_t q_pre_wrap_start_tile = 0;
    uint32_t q_pre_wrap_tile_count = 0;
    uint32_t q_post_wrap_start_tile = 0;
    uint32_t q_valid_tile_count = 0;
};

// Whether any device's packed Q contains segments from two groups.
constexpr bool chunked_q_wraps(uint32_t start_tile, uint32_t end_tile, uint32_t q_local_tile_rows, uint32_t ring_size) {
    const uint32_t second_slab_start = (start_tile / q_local_tile_rows + ring_size) * q_local_tile_rows;
    return start_tile % q_local_tile_rows != 0 && end_tile > second_slab_start;
}

// Map [start_tile, end_tile) into a device's Q slab. The range length must not
// exceed q_local_tile_rows * ring_size; both dimensions must be nonzero.
constexpr ChunkedQMapping build_chunked_q_mapping(
    uint32_t start_tile, uint32_t end_tile, uint32_t q_local_tile_rows, uint32_t ring_size, uint32_t device_index) {
    ChunkedQMapping mapping;
    const uint32_t group = start_tile / (q_local_tile_rows * ring_size);
    for (uint32_t part = 0; part < 2; ++part) {
        const uint32_t slab_start = (group + part) * q_local_tile_rows * ring_size + device_index * q_local_tile_rows;
        const uint32_t begin = start_tile > slab_start ? start_tile : slab_start;
        const uint32_t finish = end_tile < slab_start + q_local_tile_rows ? end_tile : slab_start + q_local_tile_rows;
        const uint32_t count = finish > begin ? finish - begin : 0;
        if (part == 0) {
            mapping.q_pre_wrap_start_tile = count ? begin : 0;
            mapping.q_pre_wrap_tile_count = count;
        } else {
            mapping.q_post_wrap_start_tile = count ? begin : 0;
        }
        mapping.q_valid_tile_count += count;
    }
    return mapping;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
