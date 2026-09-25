// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "chunked_q_mapping.hpp"

namespace ttnn::operations::transformer::sdpa::ring_joint {

struct SlidingHaloSources {
    uint32_t first_start_tile = 0;
    uint32_t second_start_tile = 0;
    uint32_t count = 1;
};

struct SlidingKVSourceRange {
    uint32_t source_ring_id = 0;
    uint32_t first_k_chunk = 0;
    uint32_t last_k_chunk = 0;
    uint32_t first_compact_k_chunk = 0;
    // Global (absolute-sequence) K-chunk index of first_k_chunk. Carried in the plan so compute
    // masking never has to invert local cache rows back to global positions — an inversion that is
    // ambiguous once a bounded circular cache aliases multiple chunk groups onto one local slab.
    uint32_t first_global_k_chunk = 0;

    constexpr uint32_t k_chunk_count() const { return last_k_chunk - first_k_chunk; }
};

struct SlidingKChunkRef {
    uint32_t source_ring_id = 0;
    uint32_t source_k_chunk = 0;
    uint32_t compact_k_chunk = 0;
};

// Circular caches wrap local slabs; global token positions remain absolute.
// Zero or one slab denotes an unbounded cache.
constexpr uint32_t circular_kv_local_slab(uint32_t source_group, uint32_t circular_kv_slab_count) {
    return circular_kv_slab_count > 1 ? source_group % circular_kv_slab_count : source_group;
}

constexpr uint32_t chunked_sliding_halo_tile_rows(
    uint32_t sliding_window_tokens, uint32_t tile_height, uint32_t k_chunk_tile_rows) {
    if (tile_height == 0 || k_chunk_tile_rows == 0) {
        return 0;
    }
    const uint32_t left_window_tokens = sliding_window_tokens > 0 ? sliding_window_tokens - 1 : 0;
    const uint32_t k_chunk_tokens = k_chunk_tile_rows * tile_height;
    return ((left_window_tokens + k_chunk_tokens - 1) / k_chunk_tokens) * k_chunk_tile_rows;
}

// The receiver's Q mapping determines which predecessor slabs must be sent.
constexpr SlidingHaloSources sliding_halo_sources(
    const ChunkedQMapping& mapping, uint32_t local, uint32_t ring_size, uint32_t halo, uint32_t circular_slabs = 0) {
    SlidingHaloSources sources;
    const auto origin = [=](uint32_t query_start) {
        const uint32_t query_slab = query_start / local;
        return query_slab == 0
                   ? 0
                   : circular_kv_local_slab((query_slab - 1) / ring_size, circular_slabs) * local + local - halo;
    };
    if (mapping.q_valid_tile_count == 0) {
        return sources;
    }
    sources.first_start_tile =
        origin(mapping.q_pre_wrap_tile_count ? mapping.q_pre_wrap_start_tile : mapping.q_post_wrap_start_tile);
    if (mapping.q_pre_wrap_tile_count && mapping.q_valid_tile_count > mapping.q_pre_wrap_tile_count) {
        sources.second_start_tile = origin(mapping.q_post_wrap_start_tile);
        sources.count = 2;
    }
    return sources;
}

// A Q compute block can straddle its packed slab's wrap: two source ranges per segment.
struct SlidingQWorkPlan {
    static constexpr uint32_t max_source_ranges = 4;

    std::array<SlidingKVSourceRange, max_source_ranges> source_ranges{};
    uint32_t source_range_count = 0;
    uint32_t total_k_chunk_count = 0;
    bool is_valid = false;

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

    // Absolute (global-sequence) K-chunk index of work item work_index, for masks that must not
    // invert local cache rows (ambiguous once a circular cache aliases several chunk groups onto one
    // local slab). Kept off SlidingKChunkRef and only called by circular kernels.
    constexpr uint32_t global_k_chunk_at(uint32_t work_index) const {
        for (uint32_t range_index = 0; range_index < source_range_count; ++range_index) {
            const auto& range = source_ranges[range_index];
            if (work_index < range.k_chunk_count()) {
                return range.first_global_k_chunk + work_index;
            }
            work_index -= range.k_chunk_count();
        }
        return 0;
    }
};

// Chunked prefill stores each global Q-sized group as one local slab per ring
// device. A window can need only the local slab and the cyclic predecessor's
// tail; device 0 consumes the final device's tail from the preceding group.
// circular_kv_slab_count (0/1 = unbounded): the local K/V cache is a circular buffer of that many
// Q-sized slabs, so chunk group g lives in local slab (g % n_slabs) instead of slab g. Only the
// local-row derivation wraps; every range_global_* / first_global_k_chunk value stays absolute.
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
    uint32_t logical_k_tile_rows,
    uint32_t circular_kv_slab_count = 0,
    const ChunkedQMapping* rotated_q = nullptr) {
    SlidingQWorkPlan plan;
    if (q_chunk_tile_rows == 0 || q_local_tile_rows == 0 || ring_size == 0 || sliding_window_tokens == 0 ||
        tile_height == 0 || k_chunk_tile_rows == 0 || q_local_tile_rows % k_chunk_tile_rows != 0 ||
        q_local_start_tile + q_chunk_tile_rows > q_local_tile_rows) {
        return plan;
    }
    const uint32_t group_rows = ring_size * q_local_tile_rows;
    if (!rotated_q && logical_k_tile_rows < group_rows) {
        return plan;
    }
    const uint32_t halo = chunked_sliding_halo_tile_rows(sliding_window_tokens, tile_height, k_chunk_tile_rows);
    if (halo > q_local_tile_rows) {
        return plan;
    }
    const ChunkedQMapping mapping = rotated_q
                                        ? *rotated_q
                                        : ChunkedQMapping{
                                              logical_k_tile_rows - group_rows + q_device_index * q_local_tile_rows,
                                              q_local_tile_rows,
                                              0,
                                              q_local_tile_rows};
    const uint32_t first_query_slab =
        (mapping.q_pre_wrap_tile_count ? mapping.q_pre_wrap_start_tile : mapping.q_post_wrap_start_tile) /
        q_local_tile_rows;
    const uint32_t left_tiles = (sliding_window_tokens - 1 + tile_height - 1) / tile_height;
    for (uint32_t part = 0; part < 2; ++part) {
        const uint32_t segment_begin = part == 0 ? 0 : mapping.q_pre_wrap_tile_count;
        const uint32_t segment_end = part == 0 ? mapping.q_pre_wrap_tile_count : mapping.q_valid_tile_count;
        const uint32_t begin = q_local_start_tile > segment_begin ? q_local_start_tile : segment_begin;
        const uint32_t chunk_end = q_local_start_tile + q_chunk_tile_rows;
        const uint32_t end = chunk_end < segment_end ? chunk_end : segment_end;
        if (begin >= end) {
            continue;
        }
        const uint32_t segment_origin = part == 0 ? mapping.q_pre_wrap_start_tile : mapping.q_post_wrap_start_tile;
        const uint32_t q_begin = segment_origin + begin - segment_begin;
        const uint32_t q_end = segment_origin + end - segment_begin;
        const uint32_t window_begin = q_begin > left_tiles ? q_begin - left_tiles : 0;
        const uint32_t window_end = q_end < logical_k_tile_rows ? q_end : logical_k_tile_rows;
        for (uint32_t slab = window_begin / q_local_tile_rows; slab * q_local_tile_rows < window_end; ++slab) {
            const uint32_t source = slab % ring_size;
            const uint32_t slab_begin = slab * q_local_tile_rows;
            const uint32_t range_begin = window_begin > slab_begin ? window_begin : slab_begin;
            const uint32_t range_end =
                window_end < slab_begin + q_local_tile_rows ? window_end : slab_begin + q_local_tile_rows;
            const uint32_t local_base =
                circular_kv_local_slab(slab / ring_size, circular_kv_slab_count) * q_local_tile_rows;
            const uint32_t local_begin = local_base + range_begin - slab_begin;
            const uint32_t local_end = local_base + range_end - slab_begin;
            const uint32_t clipped_end = local_end < k_local_tile_rows ? local_end : k_local_tile_rows;
            if (local_begin >= clipped_end) {
                continue;
            }
            const uint32_t first_k = local_begin / k_chunk_tile_rows;
            const uint32_t last_k = (clipped_end + k_chunk_tile_rows - 1) / k_chunk_tile_rows;
            uint32_t compact_k = 0;
            if (source != q_device_index) {
                const uint32_t halo_origin = local_base + q_local_tile_rows - halo;
                if (first_k * k_chunk_tile_rows < halo_origin) {
                    return SlidingQWorkPlan{};
                }
                const uint32_t halo_slot = slab + 1 == first_query_slab ? 0 : 1;
                compact_k = (halo_slot * halo + first_k * k_chunk_tile_rows - halo_origin) / k_chunk_tile_rows;
            }
            if (plan.source_range_count == SlidingQWorkPlan::max_source_ranges) {
                return SlidingQWorkPlan{};
            }
            plan.source_ranges[plan.source_range_count++] = SlidingKVSourceRange{
                source,
                first_k,
                last_k,
                compact_k,
                slab_begin / k_chunk_tile_rows + first_k - local_base / k_chunk_tile_rows};
            plan.total_k_chunk_count += last_k - first_k;
        }
    }
    if (rotated_q && plan.total_k_chunk_count == 0) {
        // Padded Q blocks still consume one fully masked K block to preserve the output cadence.
        plan.source_ranges[0] =
            SlidingKVSourceRange{q_device_index, 0, 1, 0, q_device_index * q_local_tile_rows / k_chunk_tile_rows};
        plan.source_range_count = plan.total_k_chunk_count = 1;
    }
    plan.is_valid = plan.total_k_chunk_count != 0;
    return plan;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
