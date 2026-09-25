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

// Multi-hop halo geometry. The halo is the `halo_tile_rows` of history immediately preceding the
// receiver's first Q slab. When it is wider than one per-device Q slab it cannot come from a single
// neighbour, so it is split across `hop_count` cyclic predecessors: hop d (1-based) carries the TAIL
// of the slab d positions back, and the hops land in disjoint, oldest-first row ranges of the same
// compact buffer. At hop_count == 1 every formula below reduces to the single-neighbour halo.
// A multi-hop halo needs a single Q segment: block-cyclic Q that wraps (two segments) is supported
// only when the halo fits in one slab.

constexpr uint32_t chunked_sliding_halo_hop_count(uint32_t halo_tile_rows, uint32_t q_local_tile_rows) {
    return q_local_tile_rows == 0 ? 0 : (halo_tile_rows + q_local_tile_rows - 1) / q_local_tile_rows;
}

// Rows carried by hop d: a full slab, except the farthest hop which carries only the remainder.
constexpr uint32_t chunked_sliding_halo_hop_rows(uint32_t halo_tile_rows, uint32_t q_local_tile_rows, uint32_t hop) {
    if (hop == 0 || q_local_tile_rows == 0) {
        return 0;
    }
    const uint32_t consumed = (hop - 1) * q_local_tile_rows;
    if (consumed >= halo_tile_rows) {
        return 0;
    }
    const uint32_t remaining = halo_tile_rows - consumed;
    return remaining < q_local_tile_rows ? remaining : q_local_tile_rows;
}

// Hops that actually cross the fabric. A halo deep enough to reach `ring_size` slabs back wraps
// onto this device's OWN earlier slab, which the work plan reads locally, so no exchange is built
// for it. Everything downstream (kernel indices, signal counts) counts remote hops.
constexpr uint32_t chunked_sliding_halo_remote_hop_count(
    uint32_t halo_tile_rows, uint32_t q_local_tile_rows, uint32_t ring_size) {
    const uint32_t hops = chunked_sliding_halo_hop_count(halo_tile_rows, q_local_tile_rows);
    if (ring_size == 0) {
        return 0;
    }
    return hops < ring_size - 1 ? hops : ring_size - 1;
}

// First compact-buffer row written by hop d. The buffer holds the halo oldest-first, so the
// farthest hop starts at row 0 and hop 1 is the last block, ending at row halo_tile_rows.
constexpr uint32_t chunked_sliding_halo_hop_dest_row(
    uint32_t halo_tile_rows, uint32_t q_local_tile_rows, uint32_t hop) {
    const uint32_t consumed = hop * q_local_tile_rows;
    return consumed >= halo_tile_rows ? 0 : halo_tile_rows - consumed;
}

// The receiver's Q mapping determines which predecessor slabs must be sent: hop d ships the tail of
// the slab d positions before each Q segment's first slab. A one-hop halo can serve two segments
// (block-cyclic Q that wraps); a multi-hop halo serves one.
constexpr SlidingHaloSources sliding_halo_sources(
    const ChunkedQMapping& mapping,
    uint32_t local,
    uint32_t ring_size,
    uint32_t halo,
    uint32_t circular_slabs = 0,
    uint32_t hop = 1) {
    SlidingHaloSources sources;
    const uint32_t tail = chunked_sliding_halo_hop_rows(halo, local, hop);
    const auto origin = [=](uint32_t query_start) {
        const uint32_t query_slab = query_start / local;
        // No such slab before the first group: the receiver's clipped plan never reads the payload.
        return query_slab < hop
                   ? 0
                   : circular_kv_local_slab((query_slab - hop) / ring_size, circular_slabs) * local + local - tail;
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

// A Q compute block can straddle its packed slab's wrap: two source ranges per segment. A multi-hop
// halo (one segment) needs one range per hop plus the local slab; chunked sliding attention accepts
// rings of at most 8 (SP8, validated in ring_joint_sdpa_device_operation.cpp), so a halo never needs
// more than 8 hops. build_sliding_q_work_plan returns an EMPTY plan on overflow, and
// validate_on_program_cache_miss rejects deeper halos up front.
struct SlidingQWorkPlan {
    static constexpr uint32_t max_halo_hops = 8;
    static constexpr uint32_t max_source_ranges = max_halo_hops + 1;

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
    const uint32_t hops = chunked_sliding_halo_hop_count(halo, q_local_tile_rows);
    // Beyond the whole ring a source is this device's own earlier slab, already a local read.
    if (hops > ring_size) {
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
    // A multi-hop halo holds one block per hop for a single Q segment. Wrapped Q (two segments) with a
    // multi-hop halo is rejected on the host; if it reaches here anyway it takes the masked fallback.
    const bool two_segments =
        mapping.q_pre_wrap_tile_count != 0 && mapping.q_valid_tile_count > mapping.q_pre_wrap_tile_count;
    const uint32_t segment_count = hops > 1 && two_segments ? 0 : 2;
    for (uint32_t part = 0; part < segment_count; ++part) {
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
            if (source != q_device_index && hops == 1) {
                const uint32_t halo_origin = local_base + q_local_tile_rows - halo;
                if (first_k * k_chunk_tile_rows < halo_origin) {
                    return SlidingQWorkPlan{};
                }
                const uint32_t halo_slot = slab + 1 == first_query_slab ? 0 : 1;
                compact_k = (halo_slot * halo + first_k * k_chunk_tile_rows - halo_origin) / k_chunk_tile_rows;
            } else if (source != q_device_index) {
                // Hop d's block holds the tail of the slab d before the Q slab (sliding_halo_sources).
                const uint32_t hop = first_query_slab - slab;
                const uint32_t tail = chunked_sliding_halo_hop_rows(halo, q_local_tile_rows, hop);
                const uint32_t halo_origin = local_base + q_local_tile_rows - tail;
                if (slab >= first_query_slab || tail == 0 || first_k * k_chunk_tile_rows < halo_origin) {
                    return SlidingQWorkPlan{};
                }
                compact_k = (chunked_sliding_halo_hop_dest_row(halo, q_local_tile_rows, hop) +
                             first_k * k_chunk_tile_rows - halo_origin) /
                            k_chunk_tile_rows;
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
