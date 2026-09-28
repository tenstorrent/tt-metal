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
