// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

/**
 * Shared windowed (block-diagonal) K-loop bound geometry.
 *
 * Per Q chunk, windowed attention only needs the contiguous K-chunk range covering the windows
 * that overlap the chunk's rows; everything outside it is -inf masked. The reader (which streams
 * K/V and feeds the range to the compute kernel over a control CB) and the writer (which
 * generates the mask) each call this on their own L1 copy of the same cu_window_seqlens tensor.
 * They MUST compute identical bounds or the per-Q-chunk K counts desync and the CBs deadlock —
 * this function is the single source for that math, the same contract as
 * SlidingWindowLoopGeometry.
 *
 * The window search must match windowed_mask_gen.hpp's start_window_idx search bit for bit
 * (including the first-match-wins rule and the exact overlap disjuncts): the mask generator
 * seeds its window cursor from that search, and the narrowing proof relies on the generator
 * entering k_lo with the same cursor it would have had running from chunk 0.
 *
 * Q positions are GLOBAL (q_tok_offset added — the tensor may be a sequence-parallel shard);
 * K positions and the returned chunk indices are never offset.
 *
 * Returns [k_lo, k_hi) clamped to [0, k_num_chunks], never empty: a Q chunk overlapping no
 * window (padded tail rows) gets [0, 1) — one all--inf chunk, preserving the dense path's
 * semantics (every kernel processes >= 1 K chunk per Q chunk; the NaN rows this produces are
 * never written, as the writer's output drain clamps to valid_Sqt).
 */
struct WindowedKChunkRange {
    uint32_t k_lo;
    uint32_t k_hi;
};

inline WindowedKChunkRange windowed_k_chunk_range(
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t valid_Sqt,
    uint32_t q_tok_offset,
    volatile tt_l1_ptr uint32_t* cu_ptr,
    uint32_t cu_window_seqlens_eles,
    uint32_t Sk_chunk_t,
    uint32_t k_num_chunks,
    uint32_t tile_height) {
    // Q token range: identical to windowed_mask_gen.hpp (valid_Sqt clamp, then global offset).
    const uint32_t q_row_start_tile = q_chunk * Sq_chunk_t < valid_Sqt ? q_chunk * Sq_chunk_t : valid_Sqt;
    const uint32_t q_row_end_tile =
        q_row_start_tile + Sq_chunk_t < valid_Sqt ? q_row_start_tile + Sq_chunk_t : valid_Sqt;
    const uint32_t q_low_tok = q_tok_offset + q_row_start_tile * tile_height;
    const uint32_t q_high_tok = q_tok_offset + q_row_end_tile * tile_height;

    // First overlapping window: the exact search (and overlap disjuncts) of the mask generator.
    uint32_t start_window_idx = 0;
    bool found = false;
    for (uint32_t w = 0; w + 1 < cu_window_seqlens_eles; ++w) {
        const uint32_t ws = cu_ptr[w];
        const uint32_t we = cu_ptr[w + 1];
        if ((q_low_tok >= ws && q_low_tok < we) || (q_high_tok > ws && q_high_tok <= we)) {
            start_window_idx = w;
            found = true;
            break;
        }
    }
    if (!found) {
        return {0, 1};
    }

    // Last overlapping window: walk forward while the window ends before the Q range does.
    // Empty windows (repeated cu values) are skipped by the same comparison.
    uint32_t last_window_idx = start_window_idx;
    while (last_window_idx + 2 < cu_window_seqlens_eles && cu_ptr[last_window_idx + 1] < q_high_tok) {
        ++last_window_idx;
    }

    const uint32_t chunk_toks = tile_height * Sk_chunk_t;
    uint32_t k_lo = cu_ptr[start_window_idx] / chunk_toks;
    uint32_t k_hi = (cu_ptr[last_window_idx + 1] + chunk_toks - 1) / chunk_toks;
    if (k_hi > k_num_chunks) {
        k_hi = k_num_chunks;
    }
    if (k_lo >= k_hi) {
        // Degenerate only when the windows lie entirely in K padding; keep the >= 1 chunk contract.
        k_lo = k_hi > 0 ? k_hi - 1 : 0;
        k_hi = k_lo + 1;
    }
    return {k_lo, k_hi};
}
