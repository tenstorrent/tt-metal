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

/**
 * 3D-neighborhood (NATTEN) K-chunk range, narrowed along the OUTER (T) axis only.
 *
 * The volume is flattened T-outer (k = t*H*W + h*W + w), so the T axis is the only one whose
 * neighborhood window maps to a CONTIGUOUS band of K tokens; the H/W windows are scattered inside
 * each frame and stay fully streamed (the per-element mask -inf's them). For a Q chunk, every query
 * it holds sits at some frame qt in [qt_min, qt_max], and the inward-shifted T-window start/end are
 * both non-decreasing in qt, so the union of their T-windows is [t0(qt_min), t1(qt_max)). Any K
 * token in a frame outside that band is outside EVERY query's window, so its whole mask row is -inf
 * and skipping its chunk is exact — the same "outside is -inf" narrowing proof as the block-diagonal
 * path, and identical math must run in the reader (K stream + ctrl CB) and the writer (mask loop).
 *
 * Returns [k_lo, k_hi) clamped to [0, k_num_chunks], never empty (padded-tail chunks get [0, 1)),
 * matching windowed_k_chunk_range's contract. kh/kw are irrelevant to the T band and not taken.
 */
inline WindowedKChunkRange neighborhood_t_k_chunk_range(
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t valid_Sqt,
    uint32_t q_tok_offset,
    uint32_t T,
    uint32_t H,
    uint32_t W,
    uint32_t kt,
    uint32_t Sk_chunk_t,
    uint32_t k_num_chunks,
    uint32_t tile_height) {
    const uint32_t HW = H * W;
    const uint32_t sites = T * HW;
    const uint32_t q_row_start_tile = q_chunk * Sq_chunk_t < valid_Sqt ? q_chunk * Sq_chunk_t : valid_Sqt;
    const uint32_t q_lo = q_tok_offset + q_row_start_tile * tile_height;
    // Padded-tail chunk (no real query row): keep the >= 1 all--inf chunk contract.
    if (q_lo >= sites) {
        return {0, 1};
    }
    uint32_t q_hi_excl = q_lo + Sq_chunk_t * tile_height;
    if (q_hi_excl > sites) {
        q_hi_excl = sites;
    }
    const uint32_t qt_min = q_lo / HW;
    const uint32_t qt_max = (q_hi_excl - 1) / HW;

    // Inward-shifted window along T (mirrors nbr_axis_bounds in windowed_mask_gen.hpp): start is
    // non-decreasing in the query coord, so t0 comes from qt_min and t1 from qt_max.
    const uint32_t ker = kt > T ? T : kt;
    const uint32_t half = ker / 2;
    const uint32_t last = T - ker;
    uint32_t t0 = qt_min < half ? 0u : qt_min - half;
    if (t0 > last) {
        t0 = last;
    }
    uint32_t start_max = qt_max < half ? 0u : qt_max - half;
    if (start_max > last) {
        start_max = last;
    }
    const uint32_t t1 = start_max + ker;

    const uint32_t chunk_toks = tile_height * Sk_chunk_t;
    uint32_t k_lo = (t0 * HW) / chunk_toks;
    uint32_t k_hi = (t1 * HW + chunk_toks - 1) / chunk_toks;
    if (k_hi > k_num_chunks) {
        k_hi = k_num_chunks;
    }
    if (k_lo >= k_hi) {
        k_lo = k_hi > 0 ? k_hi - 1 : 0;
        k_hi = k_lo + 1;
    }
    return {k_lo, k_hi};
}

/**
 * Step 5 (H/W-band) refinement of the T-band above. Within the T band, the H window is a
 * contiguous [h_lo, h_hi) row range per frame but the frames are HW apart, so the in-window K is a
 * SET of runs, not one range. Rather than restructure the compute's k-loop, the reader packs only
 * the active K chunks (dense) and the writer masks them by their real positions, so the compute
 * still walks a contiguous [0, N). Reader and writer must agree on the active set exactly, so this
 * box + membership test is the single source (same contract as windowed_k_chunk_range).
 *
 * The box is the union of the chunk's queries' windows, computed conservatively per axis: T from
 * [qt_min, qt_max] (Step 3), H from [qh_min, qh_max] when the chunk lies in one frame, else the
 * whole H axis (a frame-straddling chunk can't be H-narrowed cheaply — correct, just less narrow).
 * W stays full here; the per-(t,h) W band is step 5b. Supersets are always safe: the per-element
 * mask -inf's whatever the box over-includes.
 */
struct NeighborhoodBox {
    uint32_t t0;
    uint32_t t1;
    uint32_t h_lo;
    uint32_t h_hi;
    uint32_t w_lo;
    uint32_t w_hi;
};

inline uint32_t nbr_shift_start(uint32_t q, uint32_t len, uint32_t ker) {
    if (ker > len) {
        ker = len;
    }
    const uint32_t half = ker / 2;
    const uint32_t last = len - ker;
    uint32_t start = q < half ? 0u : q - half;
    if (start > last) {
        start = last;
    }
    return start;
}

inline NeighborhoodBox neighborhood_box(
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t valid_Sqt,
    uint32_t q_tok_offset,
    uint32_t T,
    uint32_t H,
    uint32_t W,
    uint32_t kt,
    uint32_t kh,
    uint32_t kw,
    uint32_t tile_height) {
    const uint32_t HW = H * W;
    const uint32_t sites = T * HW;
    const uint32_t q_row_start_tile = q_chunk * Sq_chunk_t < valid_Sqt ? q_chunk * Sq_chunk_t : valid_Sqt;
    const uint32_t q_lo = q_tok_offset + q_row_start_tile * tile_height;
    if (q_lo >= sites) {
        return {0, 1, 0, H, 0, W};  // padded-tail chunk: degenerate box (its rows are never written)
    }
    uint32_t q_hi = q_lo + Sq_chunk_t * tile_height;
    if (q_hi > sites) {
        q_hi = sites;
    }
    const uint32_t qt_min = q_lo / HW;
    const uint32_t qt_max = (q_hi - 1) / HW;
    const uint32_t ker_t = kt > T ? T : kt;
    const uint32_t t0 = nbr_shift_start(qt_min, T, kt);
    const uint32_t t1 = nbr_shift_start(qt_max, T, kt) + ker_t;

    uint32_t qh_min;
    uint32_t qh_max;
    if (qt_min == qt_max) {
        qh_min = (q_lo % HW) / W;
        qh_max = ((q_hi - 1) % HW) / W;
    } else {
        qh_min = 0;  // spans frames: H can't be cheaply bounded, keep the whole axis
        qh_max = H - 1;
    }
    const uint32_t ker_h = kh > H ? H : kh;
    const uint32_t h_lo = nbr_shift_start(qh_min, H, kh);
    const uint32_t h_hi = nbr_shift_start(qh_max, H, kh) + ker_h;

    // W band: only when the whole chunk lies in ONE (t, h) row (row index = token / W); otherwise the
    // w coordinate wraps and can't be cheaply bounded, so keep the whole W axis (falls back to 5a).
    uint32_t w_lo = 0;
    uint32_t w_hi = W;
    if (q_lo / W == (q_hi - 1) / W) {
        const uint32_t qw_min = q_lo % W;
        const uint32_t qw_max = (q_hi - 1) % W;
        const uint32_t ker_w = kw > W ? W : kw;
        w_lo = nbr_shift_start(qw_min, W, kw);
        w_hi = nbr_shift_start(qw_max, W, kw) + ker_w;
    }
    return {t0, t1, h_lo, h_hi, w_lo, w_hi};
}

// A K chunk (flattened tokens [c*chunk_toks, c*chunk_toks + chunk_toks)) is active iff it overlaps
// the box in some (t, h) row it touches: within row r = (t*H + h), the in-window K is [r*W + w_lo,
// r*W + w_hi). A chunk spans at most a couple of rows (chunk_toks << W typically), so this is O(1).
// With w_lo=0, w_hi=W this reduces to the 5a whole-row (H-band) test.
inline bool neighborhood_chunk_active(
    uint32_t c, uint32_t chunk_toks, uint32_t W, uint32_t H, uint32_t sites, const NeighborhoodBox& box) {
    const uint32_t lo = c * chunk_toks;
    if (lo >= sites) {
        return false;
    }
    uint32_t hi = lo + chunk_toks;  // exclusive token bound of this chunk
    if (hi > sites) {
        hi = sites;
    }
    const uint32_t r0 = lo / W;
    const uint32_t r1 = (hi - 1) / W;
    for (uint32_t r = r0; r <= r1; ++r) {
        const uint32_t t = r / H;
        const uint32_t h = r % H;
        if (t < box.t0 || t >= box.t1 || h < box.h_lo || h >= box.h_hi) {
            continue;
        }
        const uint32_t band_lo = r * W + box.w_lo;
        const uint32_t band_hi = r * W + box.w_hi;
        if (lo < band_hi && hi > band_lo) {
            return true;
        }
    }
    return false;
}

/**
 * Block-permuted Q (block-permute v1). Under a 3-D block token permutation a Q chunk is exactly ONE
 * (bt, bh, bw) block, so its physical extent is known directly -- no strided decode, and the box is
 * that block dilated by the kernel. K/V stay STRIDED, so the box is in physical (t, h, w) and the K-side
 * helpers above (neighborhood_chunk_active, the packed->flat map) are UNCHANGED; only the Q->box and the
 * per-query coord decode differ. Requires zero-pad blocks (bt|T, bh|H, bw|W) so q_chunk == one block and
 * the block counts are Tb=T/bt, Hb=H/bh, Wb=W/bw. ``w_origin`` shifts a per-shard W-band to global W (0
 * when replicated) -- the only spot the W-shard enters this path. Mirrors block_permute.py exactly.
 */
struct BlockCoord {
    uint32_t t, h, w;
};

// Q chunk index -> its (bt_i, bh_i, bw_i) block position (block-major order, W innermost).
inline BlockCoord block_index_of_chunk(uint32_t qc, uint32_t hb, uint32_t wb) {
    return {qc / (wb * hb), (qc / wb) % hb, qc % wb};
}

inline NeighborhoodBox neighborhood_box_block(
    uint32_t qc,
    uint32_t bt,
    uint32_t bh,
    uint32_t bw,
    uint32_t hb,
    uint32_t wb,
    uint32_t T,
    uint32_t H,
    uint32_t W,
    uint32_t kt,
    uint32_t kh,
    uint32_t kw,
    uint32_t w_origin) {
    const BlockCoord b = block_index_of_chunk(qc, hb, wb);
    const uint32_t t_lo = b.t * bt, t_hi = t_lo + bt - 1;
    const uint32_t h_lo = b.h * bh, h_hi = h_lo + bh - 1;
    const uint32_t w_lo = w_origin + b.w * bw, w_hi = w_lo + bw - 1;
    const uint32_t ker_t = kt > T ? T : kt, ker_h = kh > H ? H : kh, ker_w = kw > W ? W : kw;
    return {
        nbr_shift_start(t_lo, T, kt),
        nbr_shift_start(t_hi, T, kt) + ker_t,
        nbr_shift_start(h_lo, H, kh),
        nbr_shift_start(h_hi, H, kh) + ker_h,
        nbr_shift_start(w_lo, W, kw),
        nbr_shift_start(w_hi, W, kw) + ker_w,
    };
}

// Physical coord of the query at within-block position ``wid`` in chunk ``qc`` (for its mask window).
inline BlockCoord block_query_coord(
    uint32_t qc, uint32_t wid, uint32_t bt, uint32_t bh, uint32_t bw, uint32_t hb, uint32_t wb, uint32_t w_origin) {
    const BlockCoord b = block_index_of_chunk(qc, hb, wb);
    const uint32_t dw = wid % bw, dh = (wid / bw) % bh, dt = wid / (bw * bh);
    return {b.t * bt + dt, b.h * bh + dh, w_origin + b.w * bw + dw};
}

// K-chunk iteration range covering the block's box in the STRIDED K/V table (k = t*HW + h*W + w). The
// reader iterates [k_lo, k_hi) and neighborhood_chunk_active filters to the box's real cells; the mask
// walks the same range. Both call this so their counts agree (the CB contract). Returns >= 1 chunk.
inline WindowedKChunkRange neighborhood_box_k_chunk_range(
    const NeighborhoodBox& box, uint32_t H, uint32_t W, uint32_t chunk_toks, uint32_t k_num_chunks) {
    const uint32_t HW = H * W;
    uint32_t k_lo = (box.t0 * HW + box.h_lo * W + box.w_lo) / chunk_toks;
    const uint32_t last = (box.t1 - 1) * HW + (box.h_hi - 1) * W + (box.w_hi - 1);
    uint32_t k_hi = last / chunk_toks + 1;
    if (k_hi > k_num_chunks) {
        k_hi = k_num_chunks;
    }
    if (k_lo >= k_hi) {
        k_lo = k_hi > 0 ? k_hi - 1 : 0;
        k_hi = k_lo + 1;
    }
    return {k_lo, k_hi};
}
