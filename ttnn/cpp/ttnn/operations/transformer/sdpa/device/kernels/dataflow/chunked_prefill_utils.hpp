// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Chunked-prefill helpers consumed by the compute kernels (compute_common.hpp,
// compute_streaming.hpp). Kept separate from ring_utils.hpp so the compute
// headers don't pull in RingIdSequencer. The experimental sibling kernel
// (exp_ring_joint_sdpa) defines its own copy in exp_ring_utils.hpp, and
// including ring_utils.hpp from the compute headers would collide with it.

#pragma once

#include <cstdint>

// One contiguous sequence interval in a packed K chunk. Column starts are implicit:
// zero for the first run, then the preceding run's exclusive column_end.
struct PackedKVMaskRun {
    uint32_t global_start_tile;
    uint32_t column_end;
};

// Even a K tile at q_start_tile needs its within-tile diagonal mask. Skip only
// when every run ends strictly before that tile; packed source order need not
// be monotonic in global sequence coordinates.
constexpr bool packed_kv_runs_need_causal_mask(const PackedKVMaskRun* runs, uint32_t count, uint32_t q_start_tile) {
    uint32_t column_start = 0;
    for (uint32_t run = 0; run < count; ++run) {
        const uint32_t length = runs[run].column_end - column_start;
        if (runs[run].global_start_tile + length > q_start_tile) {
            return true;
        }
        column_start = runs[run].column_end;
    }
    return false;
}

// A packed attention pass concatenates equally sized physical KV sources. All sizes
// are in tiles; the final chunk may be partial, but retains the full CB stride.
struct PackedKVGroupPlan {
    uint32_t source_tiles;
    uint32_t source_count;
    uint32_t chunk_tiles;

    constexpr uint32_t tile_count() const { return source_tiles * source_count; }
    constexpr uint32_t chunk_count() const { return tile_count() / chunk_tiles + (tile_count() % chunk_tiles != 0); }
    constexpr uint32_t valid_tiles(uint32_t chunk) const {
        const uint32_t start = chunk * chunk_tiles;
        if (start >= tile_count()) {
            return 0;
        }
        const uint32_t remaining = tile_count() - start;
        return remaining < chunk_tiles ? remaining : chunk_tiles;
    }
    constexpr uint32_t source_index(uint32_t stream_tile) const { return stream_tile / source_tiles; }
    constexpr uint32_t source_offset(uint32_t stream_tile) const { return stream_tile % source_tiles; }
    constexpr uint32_t last_source(uint32_t chunk) const {
        return source_index(chunk * chunk_tiles + valid_tiles(chunk) - 1);
    }
    constexpr uint32_t segment_tiles(uint32_t chunk, uint32_t destination_offset) const {
        const uint32_t remaining = valid_tiles(chunk) - destination_offset;
        const uint32_t source_remaining = source_tiles - source_offset(chunk * chunk_tiles + destination_offset);
        return remaining < source_remaining ? remaining : source_remaining;
    }
    // Emit at most chunk_tiles runs, splitting at both source and block-cyclic slab
    // boundaries. The caller supplies chunk_tiles entries, including for partial chunks.
    constexpr uint32_t mask_runs(
        uint32_t chunk,
        const uint32_t* source_ids,
        uint32_t region_tiles,
        uint32_t global_chunk_tiles,
        PackedKVMaskRun* runs) const {
        const uint32_t valid = valid_tiles(chunk);
        uint32_t count = 0;
        for (uint32_t column = 0; column < valid;) {
            const uint32_t stream_tile = chunk * chunk_tiles + column;
            const uint32_t local = source_offset(stream_tile);
            const uint32_t region_offset = local % region_tiles;
            const uint32_t source_remaining = source_tiles - local;
            const uint32_t region_remaining = region_tiles - region_offset;
            uint32_t length = valid - column;
            length = length < source_remaining ? length : source_remaining;
            length = length < region_remaining ? length : region_remaining;
            const uint32_t global = (local / region_tiles) * global_chunk_tiles +
                                    source_ids[source_index(stream_tile)] * region_tiles + region_offset;
            column += length;
            runs[count++] = {global, column};
        }
        return count;
    }
    constexpr uint32_t global_tile(
        uint32_t stream_tile, uint32_t source_id, uint32_t region_tiles, uint32_t global_chunk_tiles) const {
        const uint32_t local = source_offset(stream_tile);
        return (local / region_tiles) * global_chunk_tiles + source_id * region_tiles + local % region_tiles;
    }
};

// Runtime logical length can change on program-cache reuse. Do not apply a plan
// for a full cache to a padded/partially active cache using the same program.
constexpr uint32_t packed_kv_source_group_size(
    uint32_t configured, uint32_t ring_size, uint32_t source_tiles, uint32_t logical_tiles, uint32_t active_mask) {
    if (ring_size == 0 || ring_size > 32 || source_tiles == 0 || configured <= 1 || ring_size % configured != 0) {
        return 1;
    }
    const uint32_t all_sources = ~uint32_t{0} >> (32 - ring_size);
    return logical_tiles == source_tiles * ring_size && active_mask == all_sources ? configured : 1;
}

struct KVPadRotationContext {
    // Maps the fixed-size Q slab used by KV-pad rotation back to absolute sequence tiles.
    // Current Q rows can straddle a chunk-group boundary, so they are represented as
    // pre-wrap and post-wrap segments. K fields are filled for each masked K chunk.
    uint32_t q_pre_wrap_start_tile = 0;
    uint32_t q_pre_wrap_tile_count = 0;
    uint32_t q_post_wrap_start_tile = 0;
    uint32_t q_valid_tile_count = 0;
    uint32_t k_local_start_tile = 0;
    uint32_t ring_id = 0;
    uint32_t logical_tile_count = 0;
};

/**
 * Per-call chunked-prefill runtime state for sdpa_ring_v2. The compile-time per-chunk
 * geometry (q_local_padded_Nt / chunk_size_t) lives in template params; this struct
 * carries the per-chunk runtime offsets.
 */
struct ChunkedContext {
    uint32_t q_start_idx_t = 0;  // absolute Q-tile offset of this chunk's Q slab
    uint32_t ring_index = 0;     // logical ring rotation index for absolute-Q-tile compute
    KVPadRotationContext kv_pad_rotation = {};
};

constexpr uint32_t chunks_until_next_multiple(uint32_t processed_chunks, uint32_t alignment) {
    const uint32_t remainder = processed_chunks % alignment;
    return remainder == 0 ? 0 : alignment - remainder;
}

// Single source of truth for the in-place latent-V predicate, derived identically by the
// program factory, the reader, and the compute kernel. In-place latent-V reads V straight
// from K^T (skipping V materialization) when the latent K/V buffer is shared AND the Q chunk
// is a single tile, where the softmax@V matmul is data-movement bound.
constexpr bool kt_inplace_v_enabled(bool v_shares_k_buffer, uint32_t Sq_chunk_t) {
    return v_shares_k_buffer && (Sq_chunk_t == 1);
}

template <bool v_shares_k_buffer, bool kt_inplace_v = false>
constexpr uint32_t dummy_kv_chunks_for_phase_alignment(uint32_t processed_chunks) {
    // Reader pushes one K entry and one V entry per real chunk; compute pops the
    // same entries. The dummy count pads the iteration so the next iteration
    // starts on the same CB phase on every chained reader core.
    if constexpr (kt_inplace_v) {
        // In-place latent-V (Sq_chunk_t==1): V is never materialized; the second
        // matmul reads K^T directly. Each real chunk consumes a single K^T entry in
        // the triple-buffered K^T CB, so a depth-3 write-pointer cycle realigns —
        // pad to the next multiple of three (matches the materialized-V phase).
        constexpr uint32_t inplace_kt_cb_entries = 3;
        return chunks_until_next_multiple(processed_chunks, inplace_kt_cb_entries);
    } else if constexpr (v_shares_k_buffer) {
        // Latent-V aliases cb_v_in to cb_k_in. Each real chunk consumes two
        // entries in a three-entry CB cycle: K^T, then materialized V. Pad to
        // the next multiple of three so the next K^T lands in the K phase.
        constexpr uint32_t aliased_kv_cb_entries = 3;
        return chunks_until_next_multiple(processed_chunks, aliased_kv_cb_entries);
    }

    // Separate K and V CBs keep the legacy two-phase chain cadence. Even chunk
    // counts need one dummy K/V pair; odd counts already leave the next writer
    // on the expected phase.
    constexpr uint32_t separate_kv_phase_count = 2;
    return (processed_chunks % separate_kv_phase_count) == 0 ? 1 : 0;
}

/**
 * Map a device-local K tile index to its global attention K position. Used by the
 * logical_n skip predicate and the diagonal-stamp mask coords. Under chunked-prefill
 * the local cache packs the per-chunk K region for each chunk back-to-back; each
 * region is q_local_padded_Nt tiles (= Q's per-device extent, since one chunk's Q
 * is one such region), so adjacent local slabs have gaps in global position space.
 */
inline uint32_t chunked_kv_global_tile_for_local(
    uint32_t ring_id, uint32_t local_tile_idx, uint32_t chunk_size_t, uint32_t q_local_padded_Nt) {
    return (local_tile_idx / q_local_padded_Nt) * chunk_size_t + ring_id * q_local_padded_Nt +
           (local_tile_idx % q_local_padded_Nt);
}

template <uint32_t chunk_size_t, uint32_t q_local_padded_Nt>
inline uint32_t chunked_kv_global_tile_for_local(uint32_t ring_id, uint32_t local_tile_idx) {
    return chunked_kv_global_tile_for_local(ring_id, local_tile_idx, chunk_size_t, q_local_padded_Nt);
}

// Global K tile of one column of a K chunk that crosses cache-region boundaries. Boundaries are
// evenly spaced at straddle_col + n*straddle_period and each adds straddle_jump; period 0 means one.
inline int32_t straddled_k_tile(
    uint32_t k_start_tile, uint32_t col, uint32_t straddle_col, uint32_t straddle_jump, uint32_t straddle_period) {
    int32_t k_pos = static_cast<int32_t>(k_start_tile) + static_cast<int32_t>(col);
    if (col >= straddle_col) {
        const uint32_t crossings = straddle_period != 0 ? 1 + (col - straddle_col) / straddle_period : 1;
        k_pos += static_cast<int32_t>(crossings * straddle_jump);
    }
    return k_pos;
}

// Column span [start, end) of run `r` in a K chunk of num_cols columns. A run is the stretch between
// two region boundaries, over which global K is contiguous. Returns false once r is past the last run.
inline bool straddle_run(
    uint32_t r, uint32_t num_cols, uint32_t straddle_col, uint32_t straddle_period, uint32_t& start, uint32_t& end) {
    if (straddle_col == 0) {
        if (r != 0) {
            return false;
        }
        start = 0;
        end = num_cols;
        return true;
    }
    const uint32_t period = straddle_period != 0 ? straddle_period : num_cols;
    start = r == 0 ? 0 : straddle_col + (r - 1) * period;
    if (start >= num_cols) {
        return false;
    }
    end = r == 0 ? straddle_col : start + period;
    if (end > num_cols) {
        end = num_cols;
    }
    return true;
}

template <
    bool chunked_enabled,
    uint32_t kv_local_padded_Nt = 0,
    uint32_t chunk_size_t = 0,
    uint32_t q_local_padded_Nt = 0>
inline uint32_t kv_global_tile_for_local(uint32_t ring_id, uint32_t local_tile_idx) {
    if constexpr (chunked_enabled) {
        return chunked_kv_global_tile_for_local<chunk_size_t, q_local_padded_Nt>(ring_id, local_tile_idx);
    } else {
        return ring_id * kv_local_padded_Nt + local_tile_idx;
    }
}

template <
    bool kv_pad_rotation_enabled,
    bool chunked_enabled,
    uint32_t kv_local_padded_Nt,
    uint32_t chunk_size_t = 0,
    uint32_t q_local_padded_Nt = 0>
inline bool kv_chunk_starts_before_logical_end(
    uint32_t ring_id, uint32_t local_k_chunk_start_tile, uint32_t logical_tile_count) {
    if (local_k_chunk_start_tile >= kv_local_padded_Nt) {
        return false;
    }
    // A partial trailing K chunk still returns true here; mask logic handles invalid columns inside it.
    if constexpr (kv_pad_rotation_enabled) {
        return chunked_kv_global_tile_for_local<chunk_size_t, q_local_padded_Nt>(ring_id, local_k_chunk_start_tile) <
               logical_tile_count;
    } else {
        return kv_global_tile_for_local<chunked_enabled, kv_local_padded_Nt, chunk_size_t, q_local_padded_Nt>(
                   ring_id, local_k_chunk_start_tile) < logical_tile_count;
    }
}

constexpr uint32_t KV_PAD_ROTATION_INVALID_TILE = 0xFFFFFFFFu;

// Map a Q row used by the mask path to its absolute sequence tile. KV-pad rotation
// leaves padded Q rows in the fixed slab; those rows map to INVALID and get fully masked.
template <bool kv_pad_rotation_enabled>
inline uint32_t q_global_tile_for_mask_row(
    uint32_t q_tile, uint32_t q_start_tile, const KVPadRotationContext& kv_pad_rotation = {}) {
    if constexpr (kv_pad_rotation_enabled) {
        const uint32_t kv_pad_q_tile = q_start_tile + q_tile;
        if (kv_pad_q_tile < kv_pad_rotation.q_pre_wrap_tile_count) {
            return kv_pad_rotation.q_pre_wrap_start_tile + kv_pad_q_tile;
        }
        if (kv_pad_q_tile < kv_pad_rotation.q_valid_tile_count) {
            return kv_pad_rotation.q_post_wrap_start_tile + (kv_pad_q_tile - kv_pad_rotation.q_pre_wrap_tile_count);
        }
        return KV_PAD_ROTATION_INVALID_TILE;
    } else {
        return q_start_tile + q_tile;
    }
}
