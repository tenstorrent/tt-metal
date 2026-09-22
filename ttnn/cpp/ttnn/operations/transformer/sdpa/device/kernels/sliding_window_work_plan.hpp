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

// Circular sliding KV cache (circular_kv_slab_count = n_slabs): each device keeps only
// n_slabs Q-sized slabs and chunk group g lives in local slab (g % n_slabs). 0 or 1 means the
// cache is unbounded (identity). This is the ONLY place local slab placement is derived; keep
// every consumer (host plan, reader, compute) on this helper so the three never desync. The
// trace-metadata halo helper (compute_halo_tail_start_Ht, ring_attention_all_gather_metadata.hpp)
// still derives the slab unwrapped, which is why circular caches are rejected on that path.
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
// receiver's own slab. When it is wider than one per-device Q slab it cannot come from a single
// neighbour, so it is split across `hop_count` cyclic predecessors: hop d (1-based) carries the
// TAIL of the slab d positions back, and the hops land in disjoint, oldest-first row ranges of the
// same compact buffer. At hop_count == 1 every formula below reduces to the single-neighbour halo.

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
    return ring_size == 0 ? 0 : (hops < ring_size - 1 ? hops : ring_size - 1);
}

// First compact-buffer row written by hop d. The buffer holds the halo oldest-first, so the
// farthest hop starts at row 0 and hop 1 is the last block, ending at row halo_tile_rows.
constexpr uint32_t chunked_sliding_halo_hop_dest_row(
    uint32_t halo_tile_rows, uint32_t q_local_tile_rows, uint32_t hop) {
    const uint32_t consumed = hop * q_local_tile_rows;
    return consumed >= halo_tile_rows ? 0 : halo_tile_rows - consumed;
}

// Local tile row at which `source_device` starts the tail it ships `hop` positions forward.
constexpr uint32_t chunked_sliding_halo_source_start_tile(
    uint32_t source_device,
    uint32_t q_local_tile_rows,
    uint32_t ring_size,
    uint32_t logical_k_tile_rows,
    uint32_t halo_tile_rows,
    uint32_t circular_kv_slab_count = 0,
    uint32_t hop = 1) {
    const uint32_t q_group_tile_rows = q_local_tile_rows * ring_size;
    if (q_group_tile_rows == 0 || logical_k_tile_rows < q_group_tile_rows || hop > ring_size) {
        return 0;
    }
    const uint32_t hop_rows = chunked_sliding_halo_hop_rows(halo_tile_rows, q_local_tile_rows, hop);
    if (hop_rows == 0) {
        return 0;
    }
    const uint32_t current_group = logical_k_tile_rows / q_group_tile_rows - 1;
    // A hop that runs off the end of the ring lands on a device in the NEXT group, so the payload
    // comes from this device's slab in the prior group. That group does not exist in group 0, but
    // the receiver's clipped work plan never reads the fixed-size payload sent on those edges.
    // Clamp the origin to an in-bounds tile range and preserve the exchange length so the halo
    // protocol remains synchronized.
    const bool wraps_group = source_device + hop >= ring_size;
    if (current_group == 0 && wraps_group) {
        return 0;
    }
    const uint32_t source_group = wraps_group ? current_group - 1 : current_group;
    return circular_kv_local_slab(source_group, circular_kv_slab_count) * q_local_tile_rows + q_local_tile_rows -
           hop_rows;
}

// Device-compatible work plan for one Q chunk. A window narrower than one Q slab touches only the
// Q-owned region and its predecessor; a wider one reaches `hop_count` slabs back (see
// chunked_sliding_halo_hop_count). Sized for a full ring of predecessors plus the local slab so the
// fixed array covers every legal chunked layout without a dynamic container.
struct SlidingQWorkPlan {
    // One range per halo hop plus this device's own slab. build_sliding_q_work_plan returns an EMPTY
    // plan on overflow, so validate_on_program_cache_miss rejects deeper halos up front.
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
    uint32_t circular_kv_slab_count = 0) {
    SlidingQWorkPlan plan;
    if (q_chunk_tile_rows == 0 || q_local_tile_rows == 0 || ring_size == 0 || sliding_window_tokens == 0 ||
        tile_height == 0 || k_chunk_tile_rows == 0 || q_local_tile_rows % k_chunk_tile_rows != 0) {
        return plan;
    }

    const uint32_t q_group_tile_rows = ring_size * q_local_tile_rows;
    // A complete first group is valid: device 0 clips at token 0 and every other
    // device can consume its predecessor within the same group.
    if (logical_k_tile_rows < q_group_tile_rows || q_local_start_tile + q_chunk_tile_rows > q_local_tile_rows) {
        return plan;
    }

    const uint32_t halo_tile_rows =
        chunked_sliding_halo_tile_rows(sliding_window_tokens, tile_height, k_chunk_tile_rows);
    // A halo wider than one Q slab is split across several cyclic predecessors, but it can never
    // need more than the whole ring: beyond that a source is this device's own earlier slab, which
    // is already a local read.
    if (chunked_sliding_halo_hop_count(halo_tile_rows, q_local_tile_rows) > ring_size) {
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
        const uint32_t source_local_base =
            circular_kv_local_slab(source_group, circular_kv_slab_count) * q_local_tile_rows;
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
            // How far back around the ring this source sits. With a one-slab halo this is always 1;
            // a wider halo reaches 2..hop_count devices back, and each hop owns its own disjoint
            // block of the compact buffer.
            const uint32_t hop = (q_device_index + ring_size - source_ring_id) % ring_size;
            const uint32_t halo_source_start = chunked_sliding_halo_source_start_tile(
                source_ring_id,
                q_local_tile_rows,
                ring_size,
                logical_k_tile_rows,
                halo_tile_rows,
                circular_kv_slab_count,
                hop);
            const uint32_t hop_dest_row = chunked_sliding_halo_hop_dest_row(halo_tile_rows, q_local_tile_rows, hop);
            // A remote range must begin inside the fixed-size payload this hop ships. Guaranteed by
            // the halo bound and first-group clipping; make it explicit so unsigned subtraction
            // cannot produce an out-of-range compact-buffer index if that contract changes.
            // (Bounded mode wraps halo_source_start and first_k_chunk through the SAME
            // source-group slab base, so the comparison stays exact across the wrap.)
            if (halo_source_start > first_k_chunk * k_chunk_tile_rows) {
                return SlidingQWorkPlan{};
            }
            first_compact_k_chunk =
                (hop_dest_row + first_k_chunk * k_chunk_tile_rows - halo_source_start) / k_chunk_tile_rows;
        }
        if (plan.source_range_count == SlidingQWorkPlan::max_source_ranges) {
            return SlidingQWorkPlan{};
        }
        // Absolute global chunk of first_k_chunk: undo the (possibly wrapped) slab base. q_local %
        // k_chunk == 0 (checked above), so both bases are whole k-chunk multiples.
        const uint32_t slab_global_k_chunk_base = slab_global_start / k_chunk_tile_rows;
        const uint32_t source_local_k_chunk_base = source_local_base / k_chunk_tile_rows;
        const uint32_t first_global_k_chunk = slab_global_k_chunk_base + first_k_chunk - source_local_k_chunk_base;
        auto& range = plan.source_ranges[plan.source_range_count++];
        range = SlidingKVSourceRange{
            .source_ring_id = source_ring_id,
            .first_k_chunk = first_k_chunk,
            .last_k_chunk = last_k_chunk,
            .first_compact_k_chunk = first_compact_k_chunk,
            .first_global_k_chunk = first_global_k_chunk,
        };
        plan.total_k_chunk_count += range.k_chunk_count();
    }
    plan.is_valid = plan.total_k_chunk_count != 0;
    return plan;
}

}  // namespace ttnn::operations::transformer::sdpa::ring_joint
