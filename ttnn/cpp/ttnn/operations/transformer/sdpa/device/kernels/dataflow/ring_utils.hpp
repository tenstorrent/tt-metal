// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring SDPA topology helpers shared between compute and dataflow kernels.

#pragma once

#include "../../ring_id_sequencer.hpp"

#include <cstdint>

/**
 * Compile-time geometry for the K-chunk that straddles the causal coarse-half boundary
 * in balanced-zigzag ring SDPA.
 *
 * Each device holds `local_padded_Nt` K tiles split as two equal coarse halves
 * (early sequence / late sequence).  When `Sk_chunk_t` does not divide the
 * coarse half size (`local_padded_Nt / 2`), exactly one K chunk straddles the
 * boundary: its first part belongs to the early half (valid on rix > rid halved
 * iterations) and its remainder belongs to the late half (must be -inf-masked).
 * The K-loop must be extended by one chunk to include it, and compute must
 * stamp -inf on the trailing `straddle_num_padded_tiles` columns.
 *
 * Used by both the compute kernel (ring_joint_sdpa.cpp) and the reader
 * dataflow kernel (ring_joint_reader.cpp) to keep the K-loop bounds in sync.
 */
template <uint32_t local_padded_Nt, uint32_t Sk_chunk_t>
struct KCausalStraddleInfo {
    static constexpr uint32_t coarse_chunk_size_t = local_padded_Nt / 2;
    static constexpr bool has_straddle = (coarse_chunk_size_t % Sk_chunk_t) != 0;
    // Index of the straddling K chunk (floor(coarse_chunk_size_t / Sk_chunk_t)).
    static constexpr uint32_t straddle_chunk_id = coarse_chunk_size_t / Sk_chunk_t;
    // Trailing tiles in the straddle chunk that belong to the late half (0 if no straddle).
    static constexpr uint32_t straddle_num_padded_tiles =
        has_straddle ? (Sk_chunk_t - (coarse_chunk_size_t % Sk_chunk_t)) : 0;
};

inline bool is_last_active_ring_iter(uint32_t active_ring_iter_mask, uint32_t ring_iter) {
    constexpr uint32_t uint32_bits = 32;
    return (ring_iter + 1 >= uint32_bits) || ((active_ring_iter_mask >> (ring_iter + 1)) == 0);
}
