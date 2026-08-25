// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Block-diagonal (windowed) attention mask generation helpers.
//
// These build the Float16_b attention mask on-the-fly from cu_window_seqlens. They are invoked by the
// SDPA writer kernel (writer_interleaved) so that the reader is left free to stream Q/K/V only. The mask
// content depends solely on tile indices and the window boundaries -- never on the Q/K/V data -- so it
// can be produced independently of the reader.

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include <tt-metalium/constants.hpp>
#include "dataflow_common.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/windowed_loop_geometry.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/neighborhood_gather.hpp"

// Zero a [row,col) sub-rectangle of a Float16_b tile that is otherwise -inf (the partial-window boundary
// tile). A Float16_b tile is 4 row-major 16x16 faces of 2-byte elements, so this is a direct per-element
// write of 0x0000. The windowed mask is always Float16_b (the streaming path does not decode block-float
// masks, and the standard path decodes Float16_b fine).
template <uint32_t tile_bytes>
inline void fill_diag_subtile_zeros(
    uint32_t cb_id,
    uint32_t tile_id,
    uint32_t row_start_idx,
    uint32_t row_end_idx,
    uint32_t col_start_idx,
    uint32_t col_end_idx) {
    constexpr uint32_t FH = tt::constants::FACE_HEIGHT;
    constexpr uint32_t FW = tt::constants::FACE_WIDTH;
    CircularBuffer cb(cb_id);
    uint32_t write_addr = cb.get_write_ptr() + tile_id * tile_bytes;
    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);
    for (uint32_t r = row_start_idx; r < row_end_idx; ++r) {
        const uint32_t face_row = (r >= FH) ? 2u : 0u;
        const uint32_t fr = r & (FH - 1);
        for (uint32_t c = col_start_idx; c < col_end_idx; ++c) {
            const uint32_t face = face_row + ((c >= FW) ? 1u : 0u);
            const uint32_t fc = c & (FW - 1);
            p[face * (FH * FW) + fr * FW + fc] = 0x0000;
        }
    }
}

// --- 3D-neighborhood (NATTEN) mask ---------------------------------------------------------------
// Inward-shifted window bounds [lo, hi) along one axis for a query coordinate (mirrors na3d.window_bounds
// and the host-validated in_axis: start = clamp(leader(q, stride) - ker/2, 0, len - ker)). Delegates the
// start to windowed_loop_geometry's nbr_shift_start so the mask and the box can never disagree.
inline void nbr_axis_bounds(uint32_t q, uint32_t len, uint32_t ker, uint32_t stride, uint32_t& lo, uint32_t& hi) {
    if (ker > len) {
        ker = len;
    }
    lo = nbr_shift_start(q, len, ker, stride);
    hi = lo + ker;
}

// Fill one Float16_b mask tile with the 3D-neighborhood pattern: element (r, c) is 0 when the K token
// k_base+c lies in the (kt,kh,kw) box of the Q token q_base+r over a (T,H,W) grid flattened T-outer
// (t = idx/(H*W), h = (idx%(H*W))/W, w = idx%W), else -inf. Rows/cols past the last real token (index
// >= T*H*W) stay -inf. Per-element because the flattened 3D window is not a contiguous tile rectangle.
//
// Spatial-SP over W: when W_full != 0 the (T, H, W) here is the LOCAL padded shard and W_full is the
// full width. A column's GLOBAL w is w_origin + (local w); the W window is computed and clamped in
// [0, W_full), and any column whose global w falls outside [0, W_full) -- the fake halo replicated at
// a true edge -- is left -inf. This makes an edge shard inward-shift at the true edge exactly as the
// whole volume would (and interior shards are unaffected, their global window == their local one).
// w_origin is signed (a left-edge shard's fake halo maps to negative global w). W_full == 0 => no
// W-sharding: global w == local w, the original single-grid behaviour.
template <uint32_t tile_bytes, uint32_t cb_mask_in>
inline void fill_neighborhood_3d_tile(
    uint32_t tile_id,
    uint32_t q_base,
    uint32_t k_base,
    uint32_t T,
    uint32_t H,
    uint32_t W,
    uint32_t kt,
    uint32_t kh,
    uint32_t kw,
    uint32_t st,
    uint32_t sh,
    uint32_t sw,
    uint32_t W_full,
    int32_t w_origin,
    // Block-permuted Q (bt==0 => strided decode): a query's physical coord comes from its block-order
    // index (qc = q/vol, within = q%vol) via block_query_coord. K stays strided, so the key decode below
    // is unchanged. hb/wb are the block counts (H/bh, W/bw).
    uint32_t bt = 0,
    uint32_t bh = 0,
    uint32_t bw = 0,
    uint32_t hb = 0,
    uint32_t wb = 0) {
    fill_neginf_tile<tile_bytes>(cb_mask_in, tile_id);
    constexpr uint32_t FH = tt::constants::FACE_HEIGHT;
    constexpr uint32_t FW = tt::constants::FACE_WIDTH;
    const uint32_t HW = H * W;
    const uint32_t sites = T * HW;
    const bool w_sharded = W_full != 0;
    const uint32_t w_span = w_sharded ? W_full : W;  // width the W window is clamped in
    CircularBuffer cb(cb_mask_in);
    volatile tt_l1_ptr uint16_t* p =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr() + tile_id * tile_bytes);
    for (uint32_t r = 0; r < tt::constants::TILE_HEIGHT; ++r) {
        const uint32_t q = q_base + r;
        if (q >= sites) {
            continue;
        }
        uint32_t qt, qh;
        int32_t qw_g;
        if (bt != 0) {  // block-permuted Q: physical coord from the block-order index
            const uint32_t vol = bt * bh * bw;
            const BlockCoord bc =
                block_query_coord(q / vol, q % vol, bt, bh, bw, hb, wb, static_cast<uint32_t>(w_origin));
            qt = bc.t;
            qh = bc.h;
            qw_g = static_cast<int32_t>(bc.w);  // block_query_coord already folds in w_origin
        } else {
            const uint32_t qrem = q % HW;
            qt = q / HW;
            qh = qrem / W;
            // Global W coordinate of this query column (local == global when not W-sharded).
            qw_g = w_origin + static_cast<int32_t>(qrem % W);
        }
        if (qw_g < 0 || qw_g >= static_cast<int32_t>(w_span)) {
            continue;  // fake-halo query row; its output is cropped, so leave it -inf
        }
        uint32_t t0, t1, h0, h1, w0, w1;
        nbr_axis_bounds(qt, T, kt, st, t0, t1);
        nbr_axis_bounds(qh, H, kh, sh, h0, h1);
        nbr_axis_bounds(static_cast<uint32_t>(qw_g), w_span, kw, sw, w0, w1);
        const uint32_t face_row = (r >= FH) ? 2u : 0u;
        const uint32_t fr = r & (FH - 1);
        for (uint32_t c = 0; c < tt::constants::TILE_WIDTH; ++c) {
            const uint32_t k = k_base + c;
            if (k >= sites) {
                continue;
            }
            const uint32_t ktt = k / HW;
            const uint32_t krem = k % HW;
            const int32_t kw_g = w_origin + static_cast<int32_t>(krem % W);
            if (kw_g < 0 || kw_g >= static_cast<int32_t>(w_span)) {
                continue;  // fake-halo key column: never in any window
            }
            if (ktt >= t0 && ktt < t1 && (krem / W) >= h0 && (krem / W) < h1 && static_cast<uint32_t>(kw_g) >= w0 &&
                static_cast<uint32_t>(kw_g) < w1) {
                const uint32_t face = face_row + ((c >= FW) ? 1u : 0u);
                p[face * (FH * FW) + fr * FW + (c & (FW - 1))] = 0x0000;
            }
        }
    }
}

// Generate and push the full block-diagonal mask (Sq_chunk_t x Sk_chunk_t tiles per K chunk, for all K
// chunks) for a single Q chunk. Self-contained: the start window is searched from cu_window_seqlens for
// this Q chunk's row range, so the result is independent of the order in which Q chunks are scheduled
// (safe under the regular SDPA factory's global-Q scheduling, incl. zigzag). cb_cu_window_in must already
// hold the cu_window_seqlens tensor (loaded + pushed once by the caller).
template <uint32_t mask_tile_bytes, uint32_t cb_mask_in, uint32_t cb_cu_window_in>
inline void generate_windowed_mask_for_q_chunk(
    Noc& noc,
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t valid_Sqt,
    uint32_t valid_Skt,
    uint32_t k_num_chunks,
    uint32_t cu_window_seqlens_eles,
    uint32_t q_tok_offset,
    uint32_t nb_T,
    uint32_t nb_H,
    uint32_t nb_W,
    uint32_t nb_kt,
    uint32_t nb_kh,
    uint32_t nb_kw,
    uint32_t nb_st,
    uint32_t nb_sh,
    uint32_t nb_sw,
    uint32_t nb_W_full,
    int32_t nb_w_origin,
    uint32_t nb_bt = 0,
    uint32_t nb_bh = 0,
    uint32_t nb_bw = 0) {
    // 3D-neighborhood mode (nb_T != 0): per-element mask over the ACTIVE K chunks only. The T band
    // bounds the outer loop; within it the H band makes the in-window K a set of runs (frames are
    // HW apart), so we pack -- masks are emitted only for chunks the box touches, by their real k
    // positions, and the reader streams exactly those chunks in the same order, so the compute walks
    // a dense [0, N). Reader and writer share neighborhood_box + neighborhood_chunk_active, so their
    // active sets (and per-Q-chunk counts) agree exactly. W stays full inside each active chunk;
    // out-of-window H/W is -inf'd per element as before.
    if (nb_T != 0) {
        const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
        const uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
        // Block-permuted Q (nb_bt != 0): the box is one (bt,bh,bw) block dilated, the k-range spans its
        // strided K cells. Reader computes these identically (shared header) so the active sets agree.
        const uint32_t hb = nb_bt != 0 ? nb_H / nb_bh : 0;
        const uint32_t wb = nb_bt != 0 ? nb_W / nb_bw : 0;
        NeighborhoodBox box;
        WindowedKChunkRange nbr_range;
        if (nb_bt != 0) {
            box = neighborhood_box_block(
                q_chunk,
                nb_bt,
                nb_bh,
                nb_bw,
                hb,
                wb,
                nb_T,
                nb_H,
                nb_W,
                nb_kt,
                nb_kh,
                nb_kw,
                nb_st,
                nb_sh,
                nb_sw,
                static_cast<uint32_t>(nb_w_origin));
            nbr_range =
                neighborhood_box_k_chunk_range(box, nb_H, nb_W, Sk_chunk_t * tt::constants::TILE_HEIGHT, k_num_chunks);
        } else {
            nbr_range = neighborhood_t_k_chunk_range(
                q_chunk,
                Sq_chunk_t,
                valid_Sqt,
                q_tok_offset,
                nb_T,
                nb_H,
                nb_W,
                nb_kt,
                nb_st,
                Sk_chunk_t,
                k_num_chunks,
                tt::constants::TILE_HEIGHT);
            box = neighborhood_box(
                q_chunk,
                Sq_chunk_t,
                valid_Sqt,
                q_tok_offset,
                nb_T,
                nb_H,
                nb_W,
                nb_kt,
                nb_kh,
                nb_kw,
                nb_st,
                nb_sh,
                nb_sw,
                tt::constants::TILE_HEIGHT);
        }
        const uint32_t sites = nb_T * nb_H * nb_W;
        const uint32_t chunk_toks = Sk_chunk_t * tt::constants::TILE_HEIGHT;
        CircularBuffer cb_mask(cb_mask_in);
        for (uint32_t k_chunk = nbr_range.k_lo; k_chunk < nbr_range.k_hi; ++k_chunk) {
            if (!neighborhood_chunk_active(k_chunk, chunk_toks, nb_W, nb_H, sites, box)) {
                continue;
            }
            const uint32_t k_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt);
            cb_mask.reserve_back(mask_chunk_tiles);
            for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                const uint32_t q_base = q_tok_offset + (q_row_start_tile + row) * tt::constants::TILE_HEIGHT;
                for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                    const uint32_t k_base = (k_row_start_tile + col) * tt::constants::TILE_HEIGHT;
                    fill_neighborhood_3d_tile<mask_tile_bytes, cb_mask_in>(
                        row * Sk_chunk_t + col,
                        q_base,
                        k_base,
                        nb_T,
                        nb_H,
                        nb_W,
                        nb_kt,
                        nb_kh,
                        nb_kw,
                        nb_st,
                        nb_sh,
                        nb_sw,
                        nb_W_full,
                        nb_w_origin,
                        nb_bt,
                        nb_bh,
                        nb_bw,
                        hb,
                        wb);
                }
            }
            noc.async_read_barrier();
            cb_mask.push_back(mask_chunk_tiles);
        }
        return;
    }
    // cu_window_seqlens is INT32/UINT32 (validated host-side); both store non-negative cumulative
    // lengths in 32-bit words, so a plain uint32 read is correct for either.
    CircularBuffer cb_cu(cb_cu_window_in);
    volatile tt_l1_ptr uint32_t* cu_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_cu.get_read_ptr());
    auto get_cu = [&](uint32_t idx) -> uint32_t { return cu_ptr[idx]; };
    auto get_window_indices = [&](uint32_t i) {
        if (i < cu_window_seqlens_eles) {
            auto low = get_cu(i);
            auto high = (i == cu_window_seqlens_eles - 1) ? low : get_cu(i + 1);
            return std::make_pair(low, high);
        }
        auto low = get_cu(cu_window_seqlens_eles - 1);
        return std::make_pair(low, low);
    };

    const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
    const uint32_t q_row_end_tile = std::min(q_row_start_tile + Sq_chunk_t, valid_Sqt);
    // Q rows are addressed LOCALLY -- the Q tensor may be a sequence-parallel shard, so `q_chunk` and
    // `valid_Sqt` count this device's rows only. Windows in cu_window_seqlens are GLOBAL, so the search
    // and every window comparison run at the global position. K/V are never sharded (Sk is the full
    // sequence), so k indices are already global and are left alone.
    const uint32_t q_low_tok = q_tok_offset + q_row_start_tile * tt::constants::TILE_HEIGHT;
    const uint32_t q_high_tok = q_tok_offset + q_row_end_tile * tt::constants::TILE_HEIGHT;

    uint32_t start_window_idx = 0;
    bool found_mask_windows = false;
    for (uint32_t w = 0; w + 1 < cu_window_seqlens_eles; ++w) {
        auto ws = get_cu(w);
        auto we = get_cu(w + 1);
        if ((q_low_tok >= ws && q_low_tok < we) || (q_high_tok > ws && q_high_tok <= we)) {
            start_window_idx = w;
            found_mask_windows = true;
            break;
        }
    }

    // Narrowed K-chunk range for this Q chunk: the same shared function the reader uses to bound its
    // K/V streaming and to feed compute — the three kernels' per-Q-chunk counts must agree exactly.
    // Skipping the out-of-range chunks cannot change the cursor walk below: their tiles all take the
    // -inf `continue` branches, which never advance `local_window_idx`.
    const auto k_range = windowed_k_chunk_range(
        q_chunk,
        Sq_chunk_t,
        valid_Sqt,
        q_tok_offset,
        cu_ptr,
        cu_window_seqlens_eles,
        Sk_chunk_t,
        k_num_chunks,
        tt::constants::TILE_HEIGHT);
    const uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    CircularBuffer cb_mask(cb_mask_in);
    uint32_t local_window_idx = start_window_idx;
    for (uint32_t k_chunk = k_range.k_lo; k_chunk < k_range.k_hi; ++k_chunk) {
        const uint32_t k_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt);

        cb_mask.reserve_back(mask_chunk_tiles);
        uint32_t mask_write_ptr_base = cb_mask.get_write_ptr();

        int zero_tile_idx = -1;
        int inf_tile_idx = -1;
        for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
            uint32_t q_start_idx = q_tok_offset + (q_row_start_tile + row) * tt::constants::TILE_HEIGHT;
            uint32_t q_end_idx = q_start_idx + tt::constants::TILE_HEIGHT;

            auto result = get_window_indices(local_window_idx);
            uint32_t window_low_idx = result.first;
            uint32_t window_high_idx = result.second;

            for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                uint32_t k_start_idx = (k_row_start_tile + col) * tt::constants::TILE_HEIGHT;
                uint32_t k_end_idx = k_start_idx + tt::constants::TILE_HEIGHT;
                uint32_t in_mask_tile_id = row * Sk_chunk_t + col;

                if (q_start_idx >= window_low_idx && q_end_idx <= window_high_idx && k_start_idx >= window_low_idx &&
                    k_end_idx <= window_high_idx) {
                    if (zero_tile_idx == -1) {
                        fill_tile_zeros<mask_tile_bytes, false>(noc, cb_mask_in, in_mask_tile_id);
                    } else {
                        copy_tile<mask_tile_bytes>(
                            noc, mask_write_ptr_base, mask_write_ptr_base, zero_tile_idx, in_mask_tile_id);
                    }
                    zero_tile_idx = in_mask_tile_id;
                    continue;
                }

                if (inf_tile_idx == -1) {
                    fill_neginf_tile<mask_tile_bytes>(cb_mask_in, in_mask_tile_id);
                } else {
                    copy_tile<mask_tile_bytes>(
                        noc, mask_write_ptr_base, mask_write_ptr_base, inf_tile_idx, in_mask_tile_id);
                }
                if (!found_mask_windows || k_end_idx <= window_low_idx || k_start_idx >= window_high_idx ||
                    window_low_idx >= window_high_idx) {
                    inf_tile_idx = in_mask_tile_id;
                    continue;
                }

                uint32_t cqs, cks, cqe, cke;
                do {
                    cqs = std::max(q_start_idx, window_low_idx);
                    cks = std::max(k_start_idx, window_low_idx);
                    cqe = std::min(q_end_idx, window_high_idx);
                    cke = std::min(k_end_idx, window_high_idx);

                    if (cqs < cqe && cks < cke) {
                        fill_diag_subtile_zeros<mask_tile_bytes>(
                            cb_mask_in,
                            in_mask_tile_id,
                            cqs - q_start_idx,
                            cqe - q_start_idx,
                            cks - k_start_idx,
                            cke - k_start_idx);
                    }

                    if (cqe >= window_high_idx && cke >= window_high_idx) {
                        local_window_idx += 1;
                        auto nxt = get_window_indices(local_window_idx);
                        window_low_idx = nxt.first;
                        window_high_idx = nxt.second;
                    }
                } while (window_low_idx < window_high_idx && cqe < q_end_idx && cke < k_end_idx);
            }
        }
        noc.async_read_barrier();
        cb_mask.push_back(mask_chunk_tiles);
    }
}

// --- Fused-gather (dense-packed) 3D-neighborhood mask ---------------------------------------------
// The key columns are the reader's DENSELY PACKED window tokens (packed index j -> real token via the
// T-outer/H-mid/W-inner box enumeration the reader packs), not contiguous flat positions. Element (r, c)
// is 0 when the real key at packed_col_base + c lies in query (q_base + r)'s (kt,kh,kw) window; padded
// columns (j >= n_box) and out-of-grid query rows stay -inf. W-shard handling mirrors
// fill_neighborhood_3d_tile (W_full == 0 => global w == local w).
//
// Structural fill (no 1024-element test): because the box is a superset of EVERY query's window, each
// query's in-window keys are a sub-rectangle of the box — under the packing, a set of contiguous W-runs
// (one per (t,h) box-row). Each query row's window is hoisted into NbrRowWindows (computed once per Q
// row-tile, reused across all packed column-tiles), then the tile's 32 columns are walked as box-row
// SEGMENTS (constant t_key/h_key, contiguous local w) and each row's in-window column run is zeroed.

// Per-query-row window bounds for one 32-row Q tile. t/h are local==global; the W window is stored as a
// LOCAL half-open range [wl0, wl1) (global [w0,w1) minus w_origin) so it compares directly to the box's
// local w. Invalid rows (padding or fake-halo query) get wl0 >= wl1 so no column is ever zeroed for them.
struct NbrRowWindows {
    uint32_t t0[tt::constants::TILE_HEIGHT];
    uint32_t t1[tt::constants::TILE_HEIGHT];
    uint32_t h0[tt::constants::TILE_HEIGHT];
    uint32_t h1[tt::constants::TILE_HEIGHT];
    int32_t wl0[tt::constants::TILE_HEIGHT];
    int32_t wl1[tt::constants::TILE_HEIGHT];
};

inline void compute_nbr_row_windows(
    uint32_t q_base,
    uint32_t T,
    uint32_t H,
    uint32_t W,
    uint32_t kt,
    uint32_t kh,
    uint32_t kw,
    uint32_t st,
    uint32_t sh,
    uint32_t sw,
    uint32_t W_full,
    int32_t w_origin,
    uint32_t bt,  // block-permuted Q (bt==0 => strided decode); hb/wb = block counts
    uint32_t bh,
    uint32_t bw,
    uint32_t hb,
    uint32_t wb,
    NbrRowWindows& rw) {
    const uint32_t HW = H * W;
    const uint32_t sites = T * HW;
    const uint32_t w_span = (W_full != 0) ? W_full : W;
    for (uint32_t r = 0; r < tt::constants::TILE_HEIGHT; ++r) {
        rw.wl0[r] = 0;
        rw.wl1[r] = 0;  // default: empty (skipped)
        const uint32_t q = q_base + r;
        if (q >= sites) {
            continue;
        }
        uint32_t qt, qh;
        int32_t qw_g;
        if (bt != 0) {  // block-permuted Q: physical coord from the block-order index
            const uint32_t vol = bt * bh * bw;
            const BlockCoord bc =
                block_query_coord(q / vol, q % vol, bt, bh, bw, hb, wb, static_cast<uint32_t>(w_origin));
            qt = bc.t;
            qh = bc.h;
            qw_g = static_cast<int32_t>(bc.w);
        } else {
            const uint32_t qrem = q % HW;
            qt = q / HW;
            qh = qrem / W;
            qw_g = w_origin + static_cast<int32_t>(qrem % W);
        }
        if (qw_g < 0 || qw_g >= static_cast<int32_t>(w_span)) {
            continue;  // fake-halo query row: its output is cropped, leave -inf
        }
        uint32_t t0, t1, h0, h1, w0, w1;
        nbr_axis_bounds(qt, T, kt, st, t0, t1);
        nbr_axis_bounds(qh, H, kh, sh, h0, h1);
        nbr_axis_bounds(static_cast<uint32_t>(qw_g), w_span, kw, sw, w0, w1);
        rw.t0[r] = t0;
        rw.t1[r] = t1;
        rw.h0[r] = h0;
        rw.h1[r] = h1;
        // Block mode keeps the box in GLOBAL W (box.w_lo folds in w_origin), so fill_packed compares
        // global key w to a GLOBAL window -- store w0/w1 directly. Strided mode's box.w_lo is LOCAL, so
        // it stores the LOCAL image w0-w_origin (fake-halo KEY test then subsumed: global w in [0,w_span)).
        if (bt != 0) {
            rw.wl0[r] = static_cast<int32_t>(w0);
            rw.wl1[r] = static_cast<int32_t>(w1);
        } else {
            rw.wl0[r] = static_cast<int32_t>(w0) - w_origin;
            rw.wl1[r] = static_cast<int32_t>(w1) - w_origin;
        }
    }
}

template <uint32_t tile_bytes, uint32_t cb_mask_in>
inline void fill_neighborhood_3d_tile_packed(
    uint32_t tile_id,
    uint32_t packed_col_base,
    uint32_t n_box,
    const NeighborhoodBox& box,
    const neighborhood_gather::BoxDims& d,
    const NbrRowWindows& rw) {
    fill_neginf_tile<tile_bytes>(cb_mask_in, tile_id);
    constexpr uint32_t FH = tt::constants::FACE_HEIGHT;
    constexpr uint32_t FW = tt::constants::FACE_WIDTH;
    constexpr uint32_t TH = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TW = tt::constants::TILE_WIDTH;
    const uint32_t bw = d.bw;  // box W extent (packed W-inner stride)
    const uint32_t bh = d.bh;  // box H extent
    CircularBuffer cb(cb_mask_in);
    volatile tt_l1_ptr uint16_t* p =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr() + tile_id * tile_bytes);

    // Walk the tile's key columns [packed_col_base, +32) as box-row segments (each: constant t_key/h_key,
    // contiguous local w). Columns with packed index >= n_box are past the box and stay -inf.
    uint32_t c = 0;
    uint32_t j = packed_col_base;
    while (c < TW && j < n_box) {
        const uint32_t r_box = j / bw;  // (t,h) box-row index = jt*bh + jh
        const uint32_t jw = j - r_box * bw;
        uint32_t seg_len = TW - c;
        if (bw - jw < seg_len) {
            seg_len = bw - jw;  // segment ends at this box-row's W boundary
        }
        if (n_box - j < seg_len) {
            seg_len = n_box - j;  // ...or at the box end
        }
        const uint32_t jt = r_box / bh;
        const uint32_t jh = r_box - jt * bh;
        const uint32_t t_key = box.t0 + jt;
        const uint32_t h_key = box.h_lo + jh;
        const int32_t w_start = static_cast<int32_t>(box.w_lo + jw);  // local w of column c
        for (uint32_t r = 0; r < TH; ++r) {
            if (rw.wl1[r] <= rw.wl0[r]) {
                continue;  // invalid/empty row
            }
            if (t_key < rw.t0[r] || t_key >= rw.t1[r] || h_key < rw.h0[r] || h_key >= rw.h1[r]) {
                continue;  // this (t,h) box-row is out of the query's window
            }
            // In-window local w in [wl0, wl1) -> segment column offset i in [i_lo, i_hi).
            int32_t i_lo = rw.wl0[r] - w_start;
            int32_t i_hi = rw.wl1[r] - w_start;
            if (i_lo < 0) {
                i_lo = 0;
            }
            if (i_hi > static_cast<int32_t>(seg_len)) {
                i_hi = static_cast<int32_t>(seg_len);
            }
            if (i_lo >= i_hi) {
                continue;
            }
            const uint32_t face_row = (r >= FH) ? 2u : 0u;
            const uint32_t fr = r & (FH - 1);
            for (int32_t i = i_lo; i < i_hi; ++i) {
                const uint32_t col = c + static_cast<uint32_t>(i);
                const uint32_t face = face_row + ((col >= FW) ? 1u : 0u);
                p[face * (FH * FW) + fr * FW + (col & (FW - 1))] = 0x0000;
            }
        }
        c += seg_len;
        j += seg_len;
    }
}

// Perfectly-block-sparse fast fill. When the box IS the window every packed key lies in every valid
// query's window, so the tile is 0 on (valid row, real column) and -inf only where there is nothing
// real: padded columns past n_box, and query rows that are grid padding or fake halo. That removes the
// per-element window test and the box-row segment walk -- the in-window columns are just the leading
// run [0, n_cols).
//
// The mask tensor itself cannot be dropped from the protocol unless n_box is a multiple of TILE_WIDTH:
// the packed tail (13 columns for an 11x11x11 kernel) has to be -inf or those padded keys would enter
// the softmax with weight exp(0).
template <uint32_t tile_bytes, uint32_t cb_mask_in>
inline void fill_neighborhood_3d_tile_single_window(
    Noc& noc, uint32_t tile_id, uint32_t packed_col_base, uint32_t n_box, const NbrRowWindows& rw) {
    constexpr uint32_t FH = tt::constants::FACE_HEIGHT;
    constexpr uint32_t FW = tt::constants::FACE_WIDTH;
    constexpr uint32_t TH = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TW = tt::constants::TILE_WIDTH;

    uint32_t n_cols = 0;
    if (packed_col_base < n_box) {
        n_cols = n_box - packed_col_base;
        if (n_cols > TW) {
            n_cols = TW;
        }
    }
    bool all_rows_valid = true;
    for (uint32_t r = 0; r < TH; ++r) {
        if (rw.wl1[r] <= rw.wl0[r]) {
            all_rows_valid = false;
            break;
        }
    }
    if (n_cols == TW && all_rows_valid) {
        fill_tile_zeros<tile_bytes, false>(noc, cb_mask_in, tile_id);
        return;
    }
    fill_neginf_tile<tile_bytes>(cb_mask_in, tile_id);
    if (n_cols == 0) {
        return;
    }
    CircularBuffer cb(cb_mask_in);
    volatile tt_l1_ptr uint16_t* p =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb.get_write_ptr() + tile_id * tile_bytes);
    for (uint32_t r = 0; r < TH; ++r) {
        if (rw.wl1[r] <= rw.wl0[r]) {
            continue;
        }
        const uint32_t face_row = (r >= FH) ? 2u : 0u;
        const uint32_t fr = r & (FH - 1);
        for (uint32_t col = 0; col < n_cols; ++col) {
            const uint32_t face = face_row + ((col >= FW) ? 1u : 0u);
            p[face * (FH * FW) + fr * FW + (col & (FW - 1))] = 0x0000;
        }
    }
}

// Generate the dense-packed mask for one Q chunk: n_packed_chunks chunks of Sq_chunk_t x Sk_chunk_t
// tiles. n_packed_chunks is computed from the same neighborhood_box the reader gathers, so the mask
// count matches the reader's K/V chunk count and the compute's ctrl-CB walk exactly (CB balance).
template <uint32_t mask_tile_bytes, uint32_t cb_mask_in>
inline void generate_neighborhood_gather_mask_for_q_chunk(
    Noc& noc,
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t valid_Sqt,
    uint32_t q_tok_offset,
    uint32_t nb_T,
    uint32_t nb_H,
    uint32_t nb_W,
    uint32_t nb_kt,
    uint32_t nb_kh,
    uint32_t nb_kw,
    uint32_t nb_st,
    uint32_t nb_sh,
    uint32_t nb_sw,
    uint32_t nb_W_full,
    int32_t nb_w_origin,
    uint32_t nb_bt = 0,
    uint32_t nb_bh = 0,
    uint32_t nb_bw = 0) {
    const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
    // Block counts over THIS shard: nb_W is full W (K gather/clamp); the shard's local W = S_local/(T*H).
    // Block mode uses a LOCAL q index and the per-device W origin (which rides q_tok_offset under W-SP).
    const uint32_t hb = nb_bt != 0 ? nb_H / nb_bh : 0;
    const uint32_t wb = nb_bt != 0 ? nb_W / nb_bw : 0;  // op-T-sharded: op-W is the full (non-sharded) axis
    const uint32_t w_origin_eff = nb_bt != 0 ? q_tok_offset : static_cast<uint32_t>(nb_w_origin);
    const auto box = nb_bt != 0 ? neighborhood_box_block(
                                      q_chunk,
                                      nb_bt,
                                      nb_bh,
                                      nb_bw,
                                      hb,
                                      wb,
                                      nb_T,
                                      nb_H,
                                      nb_W,
                                      nb_kt,
                                      nb_kh,
                                      nb_kw,
                                      nb_st,
                                      nb_sh,
                                      nb_sw,
                                      w_origin_eff)
                                : neighborhood_box(
                                      q_chunk,
                                      Sq_chunk_t,
                                      valid_Sqt,
                                      q_tok_offset,
                                      nb_T,
                                      nb_H,
                                      nb_W,
                                      nb_kt,
                                      nb_kh,
                                      nb_kw,
                                      nb_st,
                                      nb_sh,
                                      nb_sw,
                                      tt::constants::TILE_HEIGHT);
    const neighborhood_gather::BoxDims d = neighborhood_gather::box_dims(box);
    const uint32_t n_box = d.n_box;
    // Under a GNA stride that matches the Q block the box collapses to one shared window, and the mask
    // degenerates to "real key, real query" -- see fill_neighborhood_3d_tile_single_window. Restricted to
    // block mode or an unsharded W because only there is a packed key guaranteed to be a real global
    // position: strided W-SP keeps a LOCAL box whose columns still need the per-element halo test.
    const bool single_window =
        neighborhood_box_is_single_window(box, nb_T, nb_H, nb_W, nb_kt, nb_kh, nb_kw) && (nb_bt != 0 || nb_W_full == 0);
    const uint32_t n_packed_t = (n_box + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    uint32_t n_packed_chunks = (n_packed_t + Sk_chunk_t - 1) / Sk_chunk_t;
    if (n_packed_chunks == 0) {
        n_packed_chunks = 1;
    }
    const uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    CircularBuffer cb_mask(cb_mask_in);
    for (uint32_t pc = 0; pc < n_packed_chunks; ++pc) {
        cb_mask.reserve_back(mask_chunk_tiles);
        for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
            // Block mode: LOCAL q index (the block-order position is q/vol,q%vol); the per-device W origin
            // is w_origin_eff, applied via compute_nbr_row_windows' block branch.
            const uint32_t q_base =
                (nb_bt != 0 ? 0u : q_tok_offset) + (q_row_start_tile + row) * tt::constants::TILE_HEIGHT;
            NbrRowWindows rw;  // this Q row-tile's per-row windows (reused across the Sk_chunk_t col tiles)
            compute_nbr_row_windows(
                q_base,
                nb_T,
                nb_H,
                nb_W,
                nb_kt,
                nb_kh,
                nb_kw,
                nb_st,
                nb_sh,
                nb_sw,
                nb_W_full,
                static_cast<int32_t>(w_origin_eff),
                nb_bt,
                nb_bh,
                nb_bw,
                hb,
                wb,
                rw);
            for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                const uint32_t packed_col_base = (pc * Sk_chunk_t + col) * tt::constants::TILE_HEIGHT;
                if (single_window) {
                    fill_neighborhood_3d_tile_single_window<mask_tile_bytes, cb_mask_in>(
                        noc, row * Sk_chunk_t + col, packed_col_base, n_box, rw);
                } else {
                    fill_neighborhood_3d_tile_packed<mask_tile_bytes, cb_mask_in>(
                        row * Sk_chunk_t + col, packed_col_base, n_box, box, d, rw);
                }
            }
        }
        noc.async_read_barrier();
        cb_mask.push_back(mask_chunk_tiles);
    }
}

// Template wrapper: instantiate the packed generator ONLY when GATHER is true (same rationale as
// windowed_generate_if_enabled -- kernel_main is not a template, so guard get_tile_size behind this).
template <bool GATHER, uint32_t cb_mask_in>
inline void neighborhood_gather_generate_if_enabled(
    Noc& noc,
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t valid_Sqt,
    uint32_t q_tok_offset,
    uint32_t nb_T,
    uint32_t nb_H,
    uint32_t nb_W,
    uint32_t nb_kt,
    uint32_t nb_kh,
    uint32_t nb_kw,
    uint32_t nb_st,
    uint32_t nb_sh,
    uint32_t nb_sw,
    uint32_t nb_W_full,
    int32_t nb_w_origin,
    uint32_t nb_bt = 0,
    uint32_t nb_bh = 0,
    uint32_t nb_bw = 0) {
    if constexpr (GATHER) {
        constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
        generate_neighborhood_gather_mask_for_q_chunk<mask_tile_bytes, cb_mask_in>(
            noc,
            q_chunk,
            Sq_chunk_t,
            Sk_chunk_t,
            valid_Sqt,
            q_tok_offset,
            nb_T,
            nb_H,
            nb_W,
            nb_kt,
            nb_kh,
            nb_kw,
            nb_st,
            nb_sh,
            nb_sw,
            nb_W_full,
            nb_w_origin,
            nb_bt,
            nb_bh,
            nb_bw);
    }
}

// Template wrapper so the windowed generator is instantiated ONLY when use_windowed_mask is true.
// kernel_main is not a template, so an `if constexpr` there does NOT discard its body — it would still
// compile, constexpr-evaluating get_tile_size on a possibly-inactive CB id. Inside this template,
// `if constexpr (W)` discards properly, so non-windowed writer builds never touch the generator.
template <bool W, uint32_t cb_mask_in, uint32_t cb_cu_window_in>
inline void windowed_generate_if_enabled(
    Noc& noc,
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t valid_Sqt,
    uint32_t valid_Skt,
    uint32_t k_num_chunks,
    uint32_t cu_window_seqlens_eles,
    uint32_t q_tok_offset,
    uint32_t nb_T,
    uint32_t nb_H,
    uint32_t nb_W,
    uint32_t nb_kt,
    uint32_t nb_kh,
    uint32_t nb_kw,
    uint32_t nb_st,
    uint32_t nb_sh,
    uint32_t nb_sw,
    uint32_t nb_W_full,
    int32_t nb_w_origin) {
    if constexpr (W) {
        constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
        generate_windowed_mask_for_q_chunk<mask_tile_bytes, cb_mask_in, cb_cu_window_in>(
            noc,
            q_chunk,
            Sq_chunk_t,
            Sk_chunk_t,
            valid_Sqt,
            valid_Skt,
            k_num_chunks,
            cu_window_seqlens_eles,
            q_tok_offset,
            nb_T,
            nb_H,
            nb_W,
            nb_kt,
            nb_kh,
            nb_kw,
            nb_st,
            nb_sh,
            nb_sw,
            nb_W_full,
            nb_w_origin);
    }
}
