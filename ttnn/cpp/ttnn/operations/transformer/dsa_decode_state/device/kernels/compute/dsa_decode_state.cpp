// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel of the fused DSA decode state maintenance. Two core roles:
//
//   role A (one core per 32-user tile row): q_resid = rmsnorm(stem[:, qa]) * w_qa, packed bf16 tiles.
//   role B (one core per user b, row r = b % 32 of its tile row):
//     latent  = rmsnorm(stem[r, kva]) * w_kva           -> placed at row a % 32 of the user's latent page (in place)
//     k       = layernorm(stem[r, k]) * w_k + b_k        -> placed at ring slot a % P of the user's ring_k block
//     g       = stem[r, g]                               -> same slot of ring_g
//     pooled  = sum_j softmax_j(g_ring[j] + ape[j]) k_ring[j]  over the P ring rows (rows >= P masked by ape = -1e9)
//                                                        -> placed at row p % 32 of the user's pool block (p = a / P)
//
// Rows are moved by one-hot selector matmuls (E_r X lifts row r into row 0; E_s^T Y places row 0 at row s), a
// cache tile is updated as old * (1 - e_s) + E_s^T row, and row sums / column sums are matmuls with the all-ones
// tile (every entry of the result holds the sum, so per-row / per-column scales are plain elementwise products).
// The in-pool softmax is shifted by the first in-pool row (shift invariance; exact to fp32 rounding).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

namespace {

// CB plan (kept in sync with the program factory and the dataflow kernels)
constexpr uint32_t cb_qa = 0;        // bf16 TILE Lq_t: role A stem block [32, Lq]
constexpr uint32_t cb_wq = 1;        // fp32 TILE 2x4: q_a norm weight (row 0), streamed in chunks of 4
constexpr uint32_t cb_kva = 2;       // bf16 TILE L_t: role B stem block [32, L]
constexpr uint32_t cb_kblk = 3;      // bf16 TILE D_t: indexer k block [32, D]
constexpr uint32_t cb_gblk = 4;      // bf16 TILE D_t: indexer gate block [32, D]
constexpr uint32_t cb_wkva = 5;      // fp32 TILE L_t: kv_a norm weight (row 0)
constexpr uint32_t cb_wk = 6;        // fp32 TILE D_t: k LayerNorm weight (row 0)
constexpr uint32_t cb_bk = 7;        // fp32 TILE D_t: k LayerNorm bias (row 0)
constexpr uint32_t cb_ape = 8;       // fp32 TILE D_t: ape block (rows < P: ape, rows >= P: -1e9)
constexpr uint32_t cb_sel = 9;       // bf16 TILE 33: E_0..E_31 (1 at (0, r)), tile 32 all ones
constexpr uint32_t cb_lat_old = 10;  // bf16 TILE L_t: the user's latent page
constexpr uint32_t cb_ring_old = 11; // bf16 TILE 2*D_t: ring_k block (0..D_t), ring_g block (D_t..)
constexpr uint32_t cb_pool_old = 12; // bf16 TILE D_t: the user's current pool block
constexpr uint32_t cb_selrow = 13;   // bf16 TILE 3: E_{lat row}, E_{ring slot}, E_{pool row} (reader-selected)
constexpr uint32_t cb_qout = 14;     // bf16 TILE 2x4: role A output chunks
constexpr uint32_t cb_lat_new = 15;  // bf16 TILE L_t
constexpr uint32_t cb_ring_new = 16; // bf16 TILE 2*D_t
constexpr uint32_t cb_pool_new = 17; // bf16 TILE D_t
constexpr uint32_t cb_tmp = 18;      // fp32 8 (two 4-tile slots)
constexpr uint32_t cb_inv = 19;      // fp32 1
constexpr uint32_t cb_normed = 20;   // fp32 L_t: normed latent block
constexpr uint32_t cb_knorm = 21;    // fp32 D_t: layernormed k block
constexpr uint32_t cb_rows = 22;     // fp32 L_t + 2*D_t: lifted rows (latent, k, g)
constexpr uint32_t cb_selT = 23;     // fp32 3: E_s^T for the latent row, the ring slot, the pool row
constexpr uint32_t cb_keep = 24;     // fp32 3: column masks 1 - e_s
constexpr uint32_t cb_ringf = 25;    // fp32 2*D_t: updated ring k / g blocks (fp32 copies)
constexpr uint32_t cb_pscr = 26;     // fp32 2*D_t: pooling scratch
constexpr uint32_t cb_prow = 27;     // fp32 D_t: pooled row (all rows equal)
constexpr uint32_t cb_run = 28;      // fp32 2: running row-sum (ping-pong)
constexpr uint32_t cb_meta_w = 29;   // (writer only)

constexpr uint32_t DST_TILES = 4;
constexpr uint32_t ONES = 32;
constexpr uint32_t ONE_BITS = 0x3F800000u;

inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }

// o[o_off + i] = A[a_off + i] * B[b_same ? b_off : b_off + i], n tiles (elementwise, any formats)
void mul_blocks(uint32_t a, uint32_t a_off, uint32_t b, uint32_t b_off, bool b_same, uint32_t o, uint32_t o_off, uint32_t n) {
    reconfig_data_format(a, b);
    mul_init(a, b);
    for (uint32_t i0 = 0; i0 < n; i0 += DST_TILES) {
        const uint32_t nn = min_u(n - i0, DST_TILES);
        tile_regs_acquire();
        for (uint32_t j = 0; j < nn; j++) {
            mul_tiles(a, b, a_off + i0 + j, b_same ? b_off : b_off + i0 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, o_off + i0 + j);
        }
        tile_regs_release();
    }
}

// o[o_off + i] = A[a_off + i] * row 0 of B[b_off + i] broadcast over the rows (sub/add variants below)
void mul_bcast_row_blocks(uint32_t a, uint32_t a_off, uint32_t b, uint32_t b_off, uint32_t o, uint32_t o_off, uint32_t n) {
    reconfig_data_format(a, b);
    mul_bcast_rows_init(a, b);
    for (uint32_t i0 = 0; i0 < n; i0 += DST_TILES) {
        const uint32_t nn = min_u(n - i0, DST_TILES);
        tile_regs_acquire();
        for (uint32_t j = 0; j < nn; j++) {
            mul_tiles_bcast_rows(a, b, a_off + i0 + j, b_off + i0 + j, j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, o_off + i0 + j);
        }
        tile_regs_release();
    }
}

// Row sums of X[x_off .. x_off + n) accumulated into a tile whose every entry holds the row sum (matmul with the
// all-ones tile), kept at the front of cb_run; `cont` continues the sum already at the front.
void row_sums(uint32_t x, uint32_t x_off, uint32_t n, bool cont) {
    CircularBuffer run(cb_run);
    bool have = cont;
    for (uint32_t i0 = 0; i0 < n; i0 += 16) {
        const uint32_t nn = min_u(n - i0, 16);
        run.reserve_back(1);
        tile_regs_acquire();
        if (have) {
            reconfig_data_format_srca(cb_run);
            copy_init(cb_run);
            copy_tile(cb_run, 0, 0);
        }
        reconfig_data_format<SrcOrder::Reverse>(x, cb_sel);
        matmul_init(x, cb_sel, 0);
        for (uint32_t j = 0; j < nn; j++) {
            matmul_tiles(x, cb_sel, x_off + i0 + j, ONES, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_run);
        tile_regs_release();
        run.push_back(1);
        if (have) {
            run.pop_front(1);
        }
        run.wait_front(1);
        have = true;
    }
}

// inv = rsqrt(front(cb_run) * mul + eps) -> cb_inv (1 tile), pops the running sum
void finish_inv(uint32_t mul_bits, uint32_t eps_bits) {
    CircularBuffer run(cb_run), inv(cb_inv);
    inv.reserve_back(1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_run);
    copy_init(cb_run);
    copy_tile(cb_run, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, mul_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_inv);
    tile_regs_release();
    inv.push_back(1);
    run.pop_front(1);
    inv.wait_front(1);
}

// rows[o_off + i] = E_r @ X[i], i < n (row r of the block lifted into row 0)
void select_row(uint32_t x, uint32_t r, uint32_t o, uint32_t o_off, uint32_t n) {
    reconfig_data_format<SrcOrder::Reverse>(cb_sel, x);
    matmul_init(cb_sel, x, 0);
    for (uint32_t i0 = 0; i0 < n; i0 += DST_TILES) {
        const uint32_t nn = min_u(n - i0, DST_TILES);
        tile_regs_acquire();
        for (uint32_t j = 0; j < nn; j++) {
            matmul_tiles(cb_sel, x, r, i0 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, o_off + i0 + j);
        }
        tile_regs_release();
    }
}

// new[i] = old[i] * keep (column broadcast, keeps every row but s) + E_s^T @ row[i], packed to `o`. n tiles.
void place_row(uint32_t old, uint32_t old_off, uint32_t keep_off, uint32_t selT_off, uint32_t row, uint32_t row_off,
               uint32_t o, uint32_t o_off, uint32_t n) {
    for (uint32_t i0 = 0; i0 < n; i0 += DST_TILES) {
        const uint32_t nn = min_u(n - i0, DST_TILES);
        tile_regs_acquire();
        reconfig_data_format(old, cb_keep);
        mul_bcast_cols_init(old, cb_keep);
        for (uint32_t j = 0; j < nn; j++) {
            mul_tiles_bcast_cols(old, cb_keep, old_off + i0 + j, keep_off, j);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_selT, row);
        matmul_init(cb_selT, row, 0);
        for (uint32_t j = 0; j < nn; j++) {
            matmul_tiles(cb_selT, row, selT_off, row_off + i0 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(o);
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, o_off + i0 + j);
        }
        pack_reconfig_data_format(cb_tmp);
        tile_regs_release();
    }
}

// o[o_off + i] = copy of A[a_off + i] (format conversion by the packer), n tiles
void copy_blocks(uint32_t a, uint32_t a_off, uint32_t o, uint32_t o_off, uint32_t n) {
    reconfig_data_format_srca(a);
    copy_init(a);
    pack_reconfig_data_format(o);
    for (uint32_t i0 = 0; i0 < n; i0 += DST_TILES) {
        const uint32_t nn = min_u(n - i0, DST_TILES);
        tile_regs_acquire();
        for (uint32_t j = 0; j < nn; j++) {
            copy_tile(a, a_off + i0 + j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, o_off + i0 + j);
        }
        tile_regs_release();
    }
    pack_reconfig_data_format(cb_tmp);
}

template <uint32_t Lq_t>
void role_a(uint32_t inv_lq_bits, uint32_t eps_bits) {
    CircularBuffer qa(cb_qa), wq(cb_wq), tmp(cb_tmp), inv(cb_inv), qout(cb_qout);
    qa.wait_front(Lq_t);
    // sum of squares over the Lq columns of every user row, in chunks of 4 tiles
    for (uint32_t c = 0; c < Lq_t; c += DST_TILES) {
        const uint32_t nn = min_u(Lq_t - c, DST_TILES);
        tmp.reserve_back(DST_TILES);
        mul_blocks(cb_qa, c, cb_qa, c, false, cb_tmp, 0, nn);
        tmp.push_back(DST_TILES);
        tmp.wait_front(DST_TILES);
        row_sums(cb_tmp, 0, nn, c > 0);
        tmp.pop_front(DST_TILES);
    }
    finish_inv(inv_lq_bits, eps_bits);
    // q_resid = x * inv * w, chunk by chunk, packed bf16
    for (uint32_t c = 0; c < Lq_t; c += DST_TILES) {
        const uint32_t nn = min_u(Lq_t - c, DST_TILES);
        tmp.reserve_back(DST_TILES);
        mul_blocks(cb_qa, c, cb_inv, 0, true, cb_tmp, 0, nn);
        tmp.push_back(DST_TILES);
        tmp.wait_front(DST_TILES);
        wq.wait_front(DST_TILES);
        qout.reserve_back(DST_TILES);
        pack_reconfig_data_format(cb_qout);
        mul_bcast_row_blocks(cb_tmp, 0, cb_wq, 0, cb_qout, 0, nn);
        pack_reconfig_data_format(cb_tmp);
        qout.push_back(DST_TILES);
        wq.pop_front(DST_TILES);
        tmp.pop_front(DST_TILES);
    }
    inv.pop_front(1);
    qa.pop_front(Lq_t);
}

template <uint32_t L_t, uint32_t D_t>
void role_b(uint32_t r, uint32_t inv_l_bits, uint32_t eps_rms_bits, uint32_t inv_d_bits, uint32_t eps_ln_bits) {
    CircularBuffer kva(cb_kva), kblk(cb_kblk), gblk(cb_gblk), wkva(cb_wkva), wk(cb_wk), bk(cb_bk), ape(cb_ape);
    CircularBuffer lat_old(cb_lat_old), ring_old(cb_ring_old), pool_old(cb_pool_old), selrow(cb_selrow);
    CircularBuffer lat_new(cb_lat_new), ring_new(cb_ring_new), pool_new(cb_pool_new);
    CircularBuffer tmp(cb_tmp), inv(cb_inv), normed(cb_normed), knorm(cb_knorm), rows(cb_rows);
    CircularBuffer selT(cb_selT), keep(cb_keep), ringf(cb_ringf), pscr(cb_pscr), prow(cb_prow);

    // ---- 1. latent = rmsnorm(kva block) * w_kva  (fp32 block)
    kva.wait_front(L_t);
    wkva.wait_front(L_t);
    normed.reserve_back(L_t);
    for (uint32_t c = 0; c < L_t; c += DST_TILES) {
        const uint32_t nn = min_u(L_t - c, DST_TILES);
        tmp.reserve_back(DST_TILES);
        mul_blocks(cb_kva, c, cb_kva, c, false, cb_tmp, 0, nn);
        tmp.push_back(DST_TILES);
        tmp.wait_front(DST_TILES);
        row_sums(cb_tmp, 0, nn, c > 0);
        tmp.pop_front(DST_TILES);
    }
    finish_inv(inv_l_bits, eps_rms_bits);
    for (uint32_t c = 0; c < L_t; c += DST_TILES) {
        const uint32_t nn = min_u(L_t - c, DST_TILES);
        tmp.reserve_back(DST_TILES);
        mul_blocks(cb_kva, c, cb_inv, 0, true, cb_tmp, 0, nn);
        tmp.push_back(DST_TILES);
        tmp.wait_front(DST_TILES);
        mul_bcast_row_blocks(cb_tmp, 0, cb_wkva, c, cb_normed, c, nn);
        tmp.pop_front(DST_TILES);
    }
    inv.pop_front(1);
    normed.push_back(L_t);
    normed.wait_front(L_t);

    // ---- 2. k = layernorm(k block) * w_k + b_k
    kblk.wait_front(D_t);
    wk.wait_front(D_t);
    bk.wait_front(D_t);
    row_sums(cb_kblk, 0, D_t, false);
    {
        // mean tile (every entry = the row mean) -> cb_inv
        CircularBuffer run(cb_run);
        inv.reserve_back(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_run);
        copy_init(cb_run);
        copy_tile(cb_run, 0, 0);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_d_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_inv);
        tile_regs_release();
        inv.push_back(1);
        run.pop_front(1);
        inv.wait_front(1);
    }
    // xm = x - mean -> tmp slot 0
    tmp.reserve_back(DST_TILES);
    {
        reconfig_data_format(cb_kblk, cb_inv);
        sub_init(cb_kblk, cb_inv);
        tile_regs_acquire();
        for (uint32_t j = 0; j < D_t; j++) {
            sub_tiles(cb_kblk, cb_inv, j, 0, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < D_t; j++) {
            pack_tile<true>(j, cb_tmp, j);
        }
        tile_regs_release();
    }
    tmp.push_back(DST_TILES);
    tmp.wait_front(DST_TILES);
    inv.pop_front(1);
    // squares of xm -> tmp slot 1
    tmp.reserve_back(DST_TILES);
    mul_blocks(cb_tmp, 0, cb_tmp, 0, false, cb_tmp, 0, D_t);
    tmp.push_back(DST_TILES);
    tmp.wait_front(2 * DST_TILES);
    row_sums(cb_tmp, DST_TILES, D_t, false);
    finish_inv(inv_d_bits, eps_ln_bits);  // inv = rsqrt(var + eps)
    // y = xm * inv -> pscr slot 0, * w_k -> pscr slot 1, + b_k -> knorm
    pscr.reserve_back(D_t);
    mul_blocks(cb_tmp, 0, cb_inv, 0, true, cb_pscr, 0, D_t);
    pscr.push_back(D_t);
    pscr.wait_front(D_t);
    pscr.reserve_back(D_t);
    mul_bcast_row_blocks(cb_pscr, 0, cb_wk, 0, cb_pscr, 0, D_t);
    pscr.push_back(D_t);
    pscr.wait_front(2 * D_t);
    knorm.reserve_back(D_t);
    {
        reconfig_data_format(cb_pscr, cb_bk);
        add_bcast_rows_init(cb_pscr, cb_bk);
        tile_regs_acquire();
        for (uint32_t j = 0; j < D_t; j++) {
            add_tiles_bcast_rows(cb_pscr, cb_bk, D_t + j, j, j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < D_t; j++) {
            pack_tile<true>(j, cb_knorm, j);
        }
        tile_regs_release();
    }
    knorm.push_back(D_t);
    knorm.wait_front(D_t);
    pscr.pop_front(2 * D_t);
    inv.pop_front(1);
    tmp.pop_front(2 * DST_TILES);
    kblk.pop_front(D_t);

    // ---- 3. lift row r of the normed latent, the normed k and the raw gate into row 0
    gblk.wait_front(D_t);
    rows.reserve_back(L_t + 2 * D_t);
    select_row(cb_normed, r, cb_rows, 0, L_t);
    select_row(cb_knorm, r, cb_rows, L_t, D_t);
    select_row(cb_gblk, r, cb_rows, L_t + D_t, D_t);
    rows.push_back(L_t + 2 * D_t);
    rows.wait_front(L_t + 2 * D_t);
    normed.pop_front(L_t);
    knorm.pop_front(D_t);
    gblk.pop_front(D_t);
    kva.pop_front(L_t);

    // ---- 4. placement operators: E_s^T (selT) and the keep masks 1 - e_s (column 0) for the three target rows
    selT.reserve_back(3);
    selrow.wait_front(3);
    reconfig_data_format_srca(cb_selrow);
    transpose_init(cb_selrow);
    for (uint32_t t = 0; t < 3; t++) {
        tile_regs_acquire();
        transpose_tile(cb_selrow, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_selT, t);
        tile_regs_release();
    }
    selT.push_back(3);
    selT.wait_front(3);
    keep.reserve_back(3);
    reconfig_data_format(cb_sel, cb_selT);
    sub_init(cb_sel, cb_selT);
    tile_regs_acquire();
    for (uint32_t t = 0; t < 3; t++) {
        sub_tiles(cb_sel, cb_selT, ONES, t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < 3; t++) {
        pack_tile<true>(t, cb_keep, t);
    }
    tile_regs_release();
    keep.push_back(3);
    keep.wait_front(3);

    // ---- 5. latent page: row lat_row <- latent row
    lat_old.wait_front(L_t);
    lat_new.reserve_back(L_t);
    place_row(cb_lat_old, 0, 0, 0, cb_rows, 0, cb_lat_new, 0, L_t);
    lat_new.push_back(L_t);
    lat_old.pop_front(L_t);

    // ---- 6. rings: slot ring_slot <- k row / g row (bf16 for the cache, fp32 for the pooling)
    ring_old.wait_front(2 * D_t);
    ringf.reserve_back(2 * D_t);
    place_row(cb_ring_old, 0, 1, 1, cb_rows, L_t, cb_ringf, 0, D_t);
    place_row(cb_ring_old, D_t, 1, 1, cb_rows, L_t + D_t, cb_ringf, D_t, D_t);
    ringf.push_back(2 * D_t);
    ring_old.pop_front(2 * D_t);
    ringf.wait_front(2 * D_t);
    rows.pop_front(L_t + 2 * D_t);
    // the bf16 ring tiles for the cache are a converted copy of the fp32 ones
    ring_new.reserve_back(2 * D_t);
    copy_blocks(cb_ringf, 0, cb_ring_new, 0, 2 * D_t);
    ring_new.push_back(2 * D_t);

    // ---- 7. pooled = sum_j softmax_j(g[j] + ape[j]) k[j] over the ring rows
    ape.wait_front(D_t);
    // logits = g_ring + ape -> pscr slot 0
    pscr.reserve_back(D_t);
    {
        reconfig_data_format(cb_ringf, cb_ape);
        add_init(cb_ringf, cb_ape);
        tile_regs_acquire();
        for (uint32_t j = 0; j < D_t; j++) {
            add_tiles(cb_ringf, cb_ape, D_t + j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < D_t; j++) {
            pack_tile<true>(j, cb_pscr, j);
        }
        tile_regs_release();
    }
    pscr.push_back(D_t);
    pscr.wait_front(D_t);
    // e = exp(logits - logits[row 0]) -> pscr slot 1
    pscr.reserve_back(D_t);
    {
        reconfig_data_format(cb_pscr, cb_pscr);
        sub_bcast_rows_init(cb_pscr, cb_pscr);
        tile_regs_acquire();
        for (uint32_t j = 0; j < D_t; j++) {
            sub_tiles_bcast_rows(cb_pscr, cb_pscr, j, j, j, 0);
        }
        exp_tile_init();
        for (uint32_t j = 0; j < D_t; j++) {
            exp_tile(j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < D_t; j++) {
            pack_tile<true>(j, cb_pscr, j);
        }
        tile_regs_release();
    }
    pscr.push_back(D_t);
    pscr.wait_front(2 * D_t);
    // 1 / column sums of e (ones^T e: every row of the product holds the column sums) -> tmp slot 0
    tmp.reserve_back(DST_TILES);
    {
        reconfig_data_format<SrcOrder::Reverse>(cb_sel, cb_pscr);
        matmul_init(cb_sel, cb_pscr, 0);
        tile_regs_acquire();
        for (uint32_t j = 0; j < D_t; j++) {
            matmul_tiles(cb_sel, cb_pscr, ONES, D_t + j, j);
        }
        recip_tile_init();
        for (uint32_t j = 0; j < D_t; j++) {
            recip_tile(j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < D_t; j++) {
            pack_tile<true>(j, cb_tmp, j);
        }
        tile_regs_release();
    }
    tmp.push_back(DST_TILES);
    tmp.wait_front(DST_TILES);
    // prob = e * recip -> tmp slot 1 (pscr keeps [logits, e] whole: a wait batch must not straddle the ring end)
    tmp.reserve_back(DST_TILES);
    mul_blocks(cb_pscr, D_t, cb_tmp, 0, false, cb_tmp, 0, D_t);
    tmp.push_back(DST_TILES);
    tmp.wait_front(2 * DST_TILES);
    pscr.pop_front(2 * D_t);
    // prob * k_ring -> pscr slot 0
    pscr.reserve_back(D_t);
    mul_blocks(cb_tmp, DST_TILES, cb_ringf, 0, false, cb_pscr, 0, D_t);
    pscr.push_back(D_t);
    pscr.wait_front(D_t);
    // pooled = column sums of prob * k -> prow (every row equal)
    prow.reserve_back(D_t);
    {
        reconfig_data_format<SrcOrder::Reverse>(cb_sel, cb_pscr);
        matmul_init(cb_sel, cb_pscr, 0);
        tile_regs_acquire();
        for (uint32_t j = 0; j < D_t; j++) {
            matmul_tiles(cb_sel, cb_pscr, ONES, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < D_t; j++) {
            pack_tile<true>(j, cb_prow, j);
        }
        tile_regs_release();
    }
    prow.push_back(D_t);
    prow.wait_front(D_t);
    tmp.pop_front(2 * DST_TILES);
    pscr.pop_front(D_t);
    ringf.pop_front(2 * D_t);
    ape.pop_front(D_t);

    // ---- 8. pool block: row pool_row <- pooled row
    pool_old.wait_front(D_t);
    pool_new.reserve_back(D_t);
    place_row(cb_pool_old, 0, 2, 2, cb_prow, 0, cb_pool_new, 0, D_t);
    pool_new.push_back(D_t);
    pool_old.pop_front(D_t);
    prow.pop_front(D_t);
    selT.pop_front(3);
    keep.pop_front(3);
    wkva.pop_front(L_t);
    wk.pop_front(D_t);
    bk.pop_front(D_t);
    selrow.pop_front(3);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Lq_t = get_compile_time_arg_val(0);
    constexpr uint32_t L_t = get_compile_time_arg_val(1);
    constexpr uint32_t D_t = get_compile_time_arg_val(2);
    constexpr uint32_t eps_rms_bits = get_compile_time_arg_val(3);
    constexpr uint32_t eps_ln_bits = get_compile_time_arg_val(4);
    constexpr uint32_t inv_lq_bits = get_compile_time_arg_val(5);
    constexpr uint32_t inv_l_bits = get_compile_time_arg_val(6);
    constexpr uint32_t inv_d_bits = get_compile_time_arg_val(7);
    const uint32_t role = get_arg_val<uint32_t>(0);
    const uint32_t idx = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_tmp, cb_tmp, cb_tmp);
    CircularBuffer sel(cb_sel);
    sel.wait_front(33);
    if (role == 0) {
        role_a<Lq_t>(inv_lq_bits, eps_rms_bits);
    } else {
        role_b<L_t, D_t>(idx & 31u, inv_l_bits, eps_rms_bits, inv_d_bits, eps_ln_bits);
    }
}
