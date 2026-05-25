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
    // Q-segment fields are fixed for the program; K fields are filled per K chunk.
    uint32_t q_old_start_nt = 0;
    uint32_t q_old_count_nt = 0;
    uint32_t q_new_start_nt = 0;
    uint32_t q_valid_nt = 0;
    uint32_t k_local_start_tile = 0;
    uint32_t ring_id = 0;
    uint32_t logical_nt = 0;
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
inline bool kv_chunk_has_valid_cols(uint32_t ring_id, uint32_t local_tile_start, uint32_t logical_nt) {
    if (local_tile_start >= kv_local_padded_Nt) {
        return false;
    }
    if constexpr (kv_pad_rotation_enabled) {
        return chunked_kv_global_tile_for_local<chunk_size_t, q_local_padded_Nt>(ring_id, local_tile_start) <
               logical_nt;
    } else {
        return kv_global_tile_for_local<chunked_enabled, kv_local_padded_Nt, chunk_size_t, q_local_padded_Nt>(
                   ring_id, local_tile_start) < logical_nt;
    }
}

constexpr uint32_t KV_PAD_ROTATION_INVALID_TILE = 0xFFFFFFFFu;

template <bool kv_pad_rotation_enabled>
inline uint32_t q_global_tile_for_local_mask(
    uint32_t q_tile, uint32_t q_start_tile, const KVPadRotationContext& kv_pad_rotation = {}) {
    if constexpr (kv_pad_rotation_enabled) {
        const uint32_t kv_pad_q_tile = q_start_tile + q_tile;
        if (kv_pad_q_tile < kv_pad_rotation.q_old_count_nt) {
            return kv_pad_rotation.q_old_start_nt + kv_pad_q_tile;
        }
        if (kv_pad_q_tile < kv_pad_rotation.q_valid_nt) {
            return kv_pad_rotation.q_new_start_nt + (kv_pad_q_tile - kv_pad_rotation.q_old_count_nt);
        }
        return KV_PAD_ROTATION_INVALID_TILE;
    } else {
        return q_start_tile + q_tile;
    }
}
