// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::transformer::sdpa::ring_joint {

// Absolute positions of the two contiguous segments packed into one device's Q slab.
// Explicit stripes reuse first-start and valid-count: every row is mapped through
// striped_q_global_tile instead of the legacy pre-/post-wrap segments.
struct ChunkedQMapping {
    uint32_t q_pre_wrap_start_tile = 0;
    uint32_t q_pre_wrap_tile_count = 0;
    uint32_t q_post_wrap_start_tile = 0;
    uint32_t q_valid_tile_count = 0;
};

// Number of rows owned by this rank before an absolute global tile position.
constexpr uint32_t striped_local_prefix(uint32_t position, uint32_t stripe, uint32_t ring_size, uint32_t rank) {
    const uint32_t group = stripe * ring_size;
    const uint32_t remainder = position % group;
    const uint32_t rank_start = rank * stripe;
    const uint32_t tail = remainder > rank_start ? remainder - rank_start : 0;
    return (position / group) * stripe + (tail < stripe ? tail : stripe);
}

// Advance in this rank's packed Q rows, jumping over the other ranks' stripes.
constexpr uint32_t striped_q_global_tile(uint32_t first_global, uint32_t row, uint32_t stripe, uint32_t ring_size) {
    const uint32_t within_stripe = first_global % stripe;
    return first_global - within_stripe + ((within_stripe + row) / stripe) * stripe * ring_size +
           (within_stripe + row) % stripe;
}

// Whether any device's packed Q contains segments from two groups.
constexpr bool chunked_q_wraps(uint32_t start_tile, uint32_t end_tile, uint32_t q_local_tile_rows, uint32_t ring_size) {
    const uint32_t second_slab_start = (start_tile / q_local_tile_rows + ring_size) * q_local_tile_rows;
    return start_tile % q_local_tile_rows != 0 && end_tile > second_slab_start;
}

// Map [start_tile, end_tile) into a device's Q slab. The range length must not
// exceed q_local_tile_rows * ring_size; both dimensions must be nonzero.
constexpr ChunkedQMapping build_chunked_q_mapping(
    uint32_t start_tile,
    uint32_t end_tile,
    uint32_t q_local_tile_rows,
    uint32_t ring_size,
    uint32_t device_index,
    uint32_t stripe_tile_rows = 0) {
    ChunkedQMapping mapping;
    if (stripe_tile_rows != 0) {
        const uint32_t first = striped_local_prefix(start_tile, stripe_tile_rows, ring_size, device_index);
        const uint32_t last = striped_local_prefix(end_tile, stripe_tile_rows, ring_size, device_index);
        mapping.q_pre_wrap_start_tile = (first / stripe_tile_rows) * stripe_tile_rows * ring_size +
                                        device_index * stripe_tile_rows + first % stripe_tile_rows;
        mapping.q_valid_tile_count = last - first;
        return mapping;
    }
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
