// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

namespace ttnn::operations::transformer::sdpa::ring_joint {

struct SlidingKVSourceRange {
    uint32_t source_ring_id = 0;
    uint32_t first_k_chunk = 0;
    uint32_t last_k_chunk = 0;
    uint32_t first_compact_k_chunk = 0;

    constexpr uint32_t k_chunk_count() const { return last_k_chunk - first_k_chunk; }
};

struct SlidingKChunkRef {
    uint32_t source_ring_id = 0;
    uint32_t source_k_chunk = 0;
    uint32_t compact_k_chunk = 0;
};

constexpr uint32_t chunked_sliding_halo_tile_rows(
    uint32_t sliding_window_tokens, uint32_t tile_height, uint32_t k_chunk_tile_rows) {
    if (tile_height == 0 || k_chunk_tile_rows == 0) {
        return 0;
    }
    const uint32_t left_window_tokens = sliding_window_tokens > 0 ? sliding_window_tokens - 1 : 0;
    const uint32_t k_chunk_tokens = k_chunk_tile_rows * tile_height;
    return ((left_window_tokens + k_chunk_tokens - 1) / k_chunk_tokens) * k_chunk_tile_rows;
}

constexpr uint32_t chunked_sliding_halo_source_start_tile(
    uint32_t source_device,
    uint32_t q_local_tile_rows,
    uint32_t ring_size,
    uint32_t logical_k_tile_rows,
    uint32_t halo_tile_rows) {
    const uint32_t q_group_tile_rows = q_local_tile_rows * ring_size;
    if (q_group_tile_rows == 0 || logical_k_tile_rows < 2 * q_group_tile_rows || halo_tile_rows > q_local_tile_rows) {
        return 0;
    }
    const uint32_t current_group = logical_k_tile_rows / q_group_tile_rows - 1;
    const uint32_t source_group = source_device + 1 == ring_size ? current_group - 1 : current_group;
    return source_group * q_local_tile_rows + q_local_tile_rows - halo_tile_rows;
}

// Device-compatible work plan for one Q chunk. The supported 128-token window
// touches at most the Q-owned region and its predecessor, so two fixed ranges
// cover the chunked layout without a dynamic container.
struct SlidingQWorkPlan {
    static constexpr uint32_t max_source_ranges = 2;

    std::array<SlidingKVSourceRange, max_source_ranges> source_ranges{};
    uint32_t source_range_count = 0;
    uint32_t total_k_chunk_count = 0;

    constexpr SlidingKChunkRef k_chunk_at(uint32_t work_index) const {
        for (uint32_t range_index = 0; range_index < source_range_count; ++range_index) {
            const auto& range = source_ranges[range_index];
            if (work_index < range.k_chunk_count()) {
                return SlidingKChunkRef{
                    .source_ring_id = range.source_ring_id,
                    .source_k_chunk = range.first_k_chunk + work_index,
                    .compact_k_chunk = range.first_compact_k_chunk + work_index,
                };
            }
            work_index -= range.k_chunk_count();
        }
        return {};
    }
};

// Chunked prefill stores each global Q-sized group as one local slab per ring
// device. A window can need only the local slab and the cyclic predecessor's
// tail; device 0 consumes the final device's tail from the preceding group.
constexpr SlidingQWorkPlan build_sliding_q_work_plan(
    uint32_t q_local_start_tile,
    uint32_t q_chunk_tile_rows,
    uint32_t q_device_index,
    uint32_t q_local_tile_rows,
    uint32_t ring_size,
    uint32_t sliding_window_tokens,
    uint32_t tile_height,
    uint32_t k_local_tile_rows,
    uint32_t k_chunk_tile_rows,
    uint32_t logical_k_tile_rows) {
    SlidingQWorkPlan plan;
    if (q_local_tile_rows == 0 || ring_size == 0 || k_chunk_tile_rows == 0 ||
        q_local_tile_rows % k_chunk_tile_rows != 0) {
        return plan;
    }

    const uint32_t q_group_tile_rows = ring_size * q_local_tile_rows;
    if (logical_k_tile_rows < 2 * q_group_tile_rows || q_local_start_tile + q_chunk_tile_rows > q_local_tile_rows) {
        return plan;
    }

    const uint32_t current_q_group_start = logical_k_tile_rows - q_group_tile_rows;
    const uint32_t global_q_start_tile =
        current_q_group_start + q_device_index * q_local_tile_rows + q_local_start_tile;
    const uint32_t left_window_tokens = sliding_window_tokens > 0 ? sliding_window_tokens - 1 : 0;
    const uint32_t left_window_tile_rows = tile_height == 0 ? 0 : (left_window_tokens + tile_height - 1) / tile_height;
    const uint32_t window_start_tile =
        global_q_start_tile > left_window_tile_rows ? global_q_start_tile - left_window_tile_rows : 0;
    const uint32_t window_end_tile = global_q_start_tile + q_chunk_tile_rows;

    const uint32_t clipped_window_start =
        window_start_tile < logical_k_tile_rows ? window_start_tile : logical_k_tile_rows;
    const uint32_t clipped_window_end = window_end_tile < logical_k_tile_rows ? window_end_tile : logical_k_tile_rows;
    if (clipped_window_start >= clipped_window_end) {
        return plan;
    }

    const uint32_t first_slab = clipped_window_start / q_local_tile_rows;
    const uint32_t last_slab = (clipped_window_end - 1) / q_local_tile_rows;
    for (uint32_t slab = first_slab; slab <= last_slab; ++slab) {
        const uint32_t source_ring_id = slab % ring_size;
        const uint32_t source_group = slab / ring_size;
        const uint32_t slab_global_start = slab * q_local_tile_rows;
        const uint32_t range_global_start =
            clipped_window_start > slab_global_start ? clipped_window_start : slab_global_start;
        const uint32_t slab_global_end = slab_global_start + q_local_tile_rows;
        const uint32_t range_global_end = clipped_window_end < slab_global_end ? clipped_window_end : slab_global_end;
        const uint32_t source_local_base = source_group * q_local_tile_rows;
        const uint32_t range_local_start = source_local_base + range_global_start - slab_global_start;
        const uint32_t range_local_end = source_local_base + range_global_end - slab_global_start;
        if (range_local_start >= k_local_tile_rows) {
            continue;
        }
        const uint32_t clipped_range_local_end =
            range_local_end < k_local_tile_rows ? range_local_end : k_local_tile_rows;
        if (range_local_start >= clipped_range_local_end) {
            continue;
        }

        const uint32_t first_k_chunk = range_local_start / k_chunk_tile_rows;
        const uint32_t last_k_chunk = (clipped_range_local_end + k_chunk_tile_rows - 1) / k_chunk_tile_rows;
        uint32_t first_compact_k_chunk = 0;
        if (source_ring_id != q_device_index) {
            const uint32_t halo_tile_rows =
                chunked_sliding_halo_tile_rows(sliding_window_tokens, tile_height, k_chunk_tile_rows);
            const uint32_t halo_source_start = chunked_sliding_halo_source_start_tile(
                source_ring_id, q_local_tile_rows, ring_size, logical_k_tile_rows, halo_tile_rows);
            first_compact_k_chunk = (first_k_chunk * k_chunk_tile_rows - halo_source_start) / k_chunk_tile_rows;
        }
        auto& range = plan.source_ranges[plan.source_range_count++];
        range = SlidingKVSourceRange{
            .source_ring_id = source_ring_id,
            .first_k_chunk = first_k_chunk,
            .last_k_chunk = last_k_chunk,
            .first_compact_k_chunk = first_compact_k_chunk,
        };
        plan.total_k_chunk_count += range.k_chunk_count();
    }
    return plan;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
