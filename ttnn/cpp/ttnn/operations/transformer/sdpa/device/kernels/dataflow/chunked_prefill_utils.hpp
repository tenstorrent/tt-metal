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

// ---------------------------------------------------------------------------
// Rotated Q split: host/device contract. The remainder ("float") Q chunks of an uneven work split
// change owner core between ring iterations, so no grid row pays the +1 K-mcast slot on every one.
// Everything here is derived identically by the program factory and all three kernels.
//
// rotated_max_slots -- the per-iteration chunk-list length -- is not here: the factory pushes it as
// each kernel's LAST compile-time arg, and each kernel reads it back from that position.
// ---------------------------------------------------------------------------

// Handoff semaphore ring depth. A slot is live only within one ring iteration, so slots are reused
// instead of allocated per iteration: program semaphores cap at 16 and are shared with the chain
// and all-gather. 3 tolerates a full iteration of inter-core skew.
constexpr uint32_t kRotatedHandoffSemDepth = 3;

// Header words each kernel's per-iteration runtime-arg block carries before its chunk-id list.
constexpr uint32_t kRotatedReaderIterHeaderWords = 2;   // [row_slot_count, my_count]
constexpr uint32_t kRotatedWriterIterHeaderWords = 3;   // [my_count, float_migrated_in, float_dest]
constexpr uint32_t kRotatedComputeIterHeaderWords = 1;  // [my_count]

// One iteration's block is header words + rotated_max_slots chunk ids, so blocks are a fixed stride
// apart and iteration `ordinal` starts here.
constexpr uint32_t rotated_iter_stride(uint32_t header_words, uint32_t max_slots) { return header_words + max_slots; }
constexpr uint32_t rotated_iter_base(uint32_t args_base, uint32_t iter_stride, uint32_t ordinal) {
    return args_base + ordinal * iter_stride;
}

// Ordinal of `ring_iter` within the ACTIVE subsequence. All three kernels index the schedule by
// this rather than by absolute ring_iter: the handoff pairs a donor signal with a receiver wait in
// the NEXT EXECUTED iteration, and consecutive ordinals cannot straddle a skipped one. Equal to
// ring_iter for a full mask, which is what lets the device-derived kv-pad mask rotate too.
constexpr uint32_t rotated_active_ordinal(uint32_t active_ring_iter_mask, uint32_t ring_iter) {
    constexpr uint32_t kRingIterMaskBits = 32;
    const uint32_t before = ring_iter >= kRingIterMaskBits ? ~0u : ((1u << ring_iter) - 1u);
    uint32_t bits = active_ring_iter_mask & before;
    uint32_t count = 0;
    while (bits) {
        bits &= bits - 1u;  // clear lowest set bit
        ++count;
    }
    return count;
}

// Handoff semaphores for a given ring size. Clamped to >= 1: a degenerate ring_size <= 1 never
// receives a float, but the kernel still declares the array and takes a modulo by this.
constexpr uint32_t rotated_handoff_sem_count(uint32_t ring_size) {
    const uint32_t receiving_iters = ring_size > 0 ? ring_size - 1 : 0;
    const uint32_t depth = receiving_iters < kRotatedHandoffSemDepth ? receiving_iters : kRotatedHandoffSemDepth;
    return depth > 0 ? depth : 1;
}

// float_dest: the float's next owner as one packed word, or kRotatedNoDest when it stays put.
// Physical NoC coordinates are small, so y takes the low byte and x the rest.
constexpr uint32_t kRotatedDestYBits = 8;
constexpr uint32_t kRotatedDestYMask = (1u << kRotatedDestYBits) - 1u;
constexpr uint32_t kRotatedNoDest = ~0u;
constexpr uint32_t rotated_pack_dest(uint32_t phys_x, uint32_t phys_y) {
    return (phys_x << kRotatedDestYBits) | phys_y;
}
constexpr uint32_t rotated_dest_x(uint32_t packed_dest) { return packed_dest >> kRotatedDestYBits; }
constexpr uint32_t rotated_dest_y(uint32_t packed_dest) { return packed_dest & kRotatedDestYMask; }

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
