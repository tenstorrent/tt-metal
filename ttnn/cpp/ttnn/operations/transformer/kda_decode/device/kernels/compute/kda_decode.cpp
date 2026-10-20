// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel of the fused KDA (Kimi Delta Attention) single-token decode block. One work item =
// (head h, 32-user tile row tb); the core walks its items one after another. Per item, on the packed
// matmul outputs (row b = user b, columns of head h), it computes for all 32 users at once:
//
//   conv   = SiLU(taps[3] * raw + sum_j taps[j] * hist[j])      per branch q, k, v  ([32, D] tile blocks)
//   q      = l2norm(conv_q) * K^-0.5 ;  k = l2norm(conv_k)       (FLA: x / sqrt(sum x^2 + 1e-6))
//   decay  = exp(lb * sigmoid(a_exp * (f + dt_bias)))           per key channel  ([32, K])
//   beta   = sigmoid(b_logit)                                     per user (column vector)
//   gate   = sigmoid(og)                                           ([32, V])
//
// then per user r (its row is lifted into row 0 of a tile block by the one-hot selector E_r):
//
//   sd = S * decay_r[:, None] ;  u = beta_r (v_r - k_r . sd) ;  S' = sd + k_r^T (x) u ;  o_r = q_r . S'
//
// S' is written back in place (the state_out CB); o_r is accumulated into row r of the [32, V] output
// block (E_r^T . o_r), which finally gets the gated RMSNorm (rmsnorm(o) * w * gate) and is packed in
// bf16 as the rows of the packed [B, H*V] output (the o_proj input). Row sums use a matmul with an
// all-ones tile (every output column holds the row sum), so a per-row scale is a plain elementwise
// multiply with that tile.
//
// Formats: raw q/k/v tiles and the selector tiles are bf16, the conv history / raw-row RM blocks bf16,
// everything else fp32. Unpack/pack formats are reconfigured explicitly at every format switch.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

namespace {

// CB plan (kept in sync with the program factory and the dataflow kernels)
constexpr uint32_t cb_raw = 0;         // bf16 TILE, 4 tiles per branch, pushed per branch
constexpr uint32_t cb_hist = 1;        // bf16 RM [32 x K] blocks (4 pages each), block (j, branch) in order j-major
constexpr uint32_t cb_taps = 2;        // fp32 TILE 12 tiles: branch b tile i at b*Kt + i, rows 0..3 = taps
constexpr uint32_t cb_gf = 3;          // fp32 TILE 4: forget-gate pre-activation block [32, K]
constexpr uint32_t cb_gog = 4;         // fp32 TILE 4: output-gate pre-activation block [32, V]
constexpr uint32_t cb_gb = 5;          // fp32 TILE 1: beta logits tile (column c = h % 32)
constexpr uint32_t cb_cc = 6;          // fp32 TILE 4: channel constants (row 0 dt_bias, row 1 exp(A_log))
constexpr uint32_t cb_onw = 7;         // fp32 TILE 4: o_norm weight (row 0), once per core
constexpr uint32_t cb_sel = 8;         // bf16 TILE 33: E_0..E_31 (1 at (0, r)), tile 32 = all ones; once per core
constexpr uint32_t cb_S = 9;           // fp32 TILE 16: recurrent state of one user
constexpr uint32_t cb_out = 10;        // bf16 TILE 4: output block [32, V]
constexpr uint32_t cb_state_out = 11;  // fp32 TILE 16: updated state
constexpr uint32_t cb_rawrm = 12;      // bf16 RM [32 x K] block (4 pages): untilized raw rows (new conv history)
constexpr uint32_t cb_qn = 13;         // fp32 4: q after conv + l2norm * K^-0.5
constexpr uint32_t cb_kn = 14;         // fp32 4: k after conv + l2norm
constexpr uint32_t cb_vc = 15;         // fp32 4: v after conv
constexpr uint32_t cb_dec = 16;        // fp32 4: decay block [32, K]
constexpr uint32_t cb_bcol = 17;       // fp32 1: sigmoid(beta logits) column (column 0)
constexpr uint32_t cb_gate = 18;       // fp32 4: sigmoid(og) block [32, V]
constexpr uint32_t cb_ht = 19;         // fp32 12: tilized history, tap j tile i at j*Kt + i
constexpr uint32_t cb_tmp = 20;        // fp32 8: scratch (two 4-tile slots)
constexpr uint32_t cb_inv = 21;        // fp32 1: per-row scale (all columns equal)
constexpr uint32_t cb_rows = 22;       // fp32 16: user r rows: q 0..3, k 4..7, v 8..11, decay 12..15 (row 0 live)
constexpr uint32_t cb_cols = 23;       // fp32 8: decay column 0..3, k column 4..7
constexpr uint32_t cb_vread = 24;      // fp32 4: k . sd
constexpr uint32_t cb_u = 25;          // fp32 4: beta (v - vread)
constexpr uint32_t cb_sd = 26;         // fp32 16: decayed state
constexpr uint32_t cb_s2 = 27;         // fp32 16: updated state (compute copy)
constexpr uint32_t cb_orow = 28;       // fp32 4: o_r
constexpr uint32_t cb_bs = 29;         // fp32 1: beta_r at [0,0]
constexpr uint32_t cb_selT = 30;       // fp32 1: E_r^T
constexpr uint32_t cb_oblk = 31;       // fp32 8: output block accumulator (two 4-tile slots)

constexpr uint32_t DST_TILES = 4;  // fp32 DST tiles per acquire (half-sync)
constexpr uint32_t ONES = 32;      // selector CB tile index of the all-ones tile

inline uint32_t min_u(uint32_t a, uint32_t b) { return a < b ? a : b; }

// ---------------------------------------------------------------- generic helpers (fp32 unless noted)

// out[0..Nt) = A[a_off .. a_off+Kt) (one tile row) @ B[Kt, Nt]
void mm_row(uint32_t a, uint32_t a_off, uint32_t b, uint32_t o, uint32_t Kt, uint32_t Nt) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(Nt);
    reconfig_data_format<SrcOrder::Reverse>(a, b);
    matmul_init(a, b, 0);
    for (uint32_t ni = 0; ni < Nt; ni++) {
        tile_regs_acquire();
        for (uint32_t ki = 0; ki < Kt; ki++) {
            matmul_tiles(a, b, a_off + ki, ki * Nt + ni, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, ni);
        tile_regs_release();
    }
    cb_o.push_back(Nt);
}

// out[i] = transpose(in[in_off + i]), n tiles
void transpose_block(uint32_t in, uint32_t in_off, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    reconfig_data_format_srca(in);
    transpose_init(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        transpose_tile(in, in_off + i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_o.push_back(n);
}

// sd[mi, ni] = S[mi, ni] * col[mi] (column broadcast: row k of every tile scaled by element (k, 0))
void bcast_col_mul(uint32_t a, uint32_t col, uint32_t col_off, uint32_t o, uint32_t Kt, uint32_t Vt) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(Kt * Vt);
    reconfig_data_format(a, col);
    mul_bcast_cols_init(a, col);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES) {
            const uint32_t nn = min_u(Vt - n0, DST_TILES);
            tile_regs_acquire();
            for (uint32_t j = 0; j < nn; j++) {
                mul_tiles_bcast_cols(a, col, mi * Vt + n0 + j, col_off + mi, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nn; j++) {
                pack_tile<true>(j, o, mi * Vt + n0 + j);
            }
            tile_regs_release();
        }
    }
    cb_o.push_back(Kt * Vt);
}

// u = (A[a_off..] - B) * beta[0,0] (row 0 live), n tiles
void delta_rule_residual(uint32_t a, uint32_t a_off, uint32_t b, uint32_t beta, uint32_t o, uint32_t n) {
    constexpr uint32_t per_batch = DST_TILES - 1;
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    for (uint32_t i0 = 0; i0 < n; i0 += per_batch) {
        const uint32_t nn = min_u(n - i0, per_batch);
        tile_regs_acquire();
        reconfig_data_format(a, b);
        sub_init(a, b);
        for (uint32_t j = 0; j < nn; j++) {
            sub_tiles(a, b, a_off + i0 + j, i0 + j, j);
        }
        reconfig_data_format_srca(beta);
        copy_init(beta);
        copy_tile(beta, 0, nn);
        sfpu_mul_bcast_col_init();
        for (uint32_t j = 0; j < nn; j++) {
            sfpu_mul_bcast_col(j, nn);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_o.push_back(n);
}

// S_new = sd + kcol[kc_off..] (x) u, into o and the state output
template <uint32_t Kt, uint32_t Vt>
void rank1_update(uint32_t sd, uint32_t kcol, uint32_t kc_off, uint32_t u, uint32_t o) {
    constexpr uint32_t kv = Kt * Vt;
    CircularBuffer cb_o(o);
    CircularBuffer cb_st(cb_state_out);
    cb_o.reserve_back(kv);
    cb_st.reserve_back(kv);
    for (uint32_t mi = 0; mi < Kt; mi++) {
        for (uint32_t n0 = 0; n0 < Vt; n0 += DST_TILES) {
            const uint32_t nn = min_u(Vt - n0, DST_TILES);
            tile_regs_acquire();
            reconfig_data_format_srca(sd);
            copy_init(sd);
            for (uint32_t j = 0; j < nn; j++) {
                copy_tile(sd, mi * Vt + n0 + j, j);
            }
            reconfig_data_format<SrcOrder::Reverse>(kcol, u);
            matmul_init(kcol, u, 0);
            for (uint32_t j = 0; j < nn; j++) {
                matmul_tiles(kcol, u, kc_off + mi, n0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < nn; j++) {
                pack_tile<true>(j, o, mi * Vt + n0 + j);
                pack_tile<true>(j, cb_state_out, mi * Vt + n0 + j);
            }
            tile_regs_release();
        }
    }
    cb_o.push_back(kv);
    cb_st.push_back(kv);
}

// rows[o_off + i] = E_r @ X[i] for i < n  (row r of the block lifted into row 0, other rows zero)
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

// inv = f(row sums of X[0..n)) as a tile whose every column holds the per-row value:
//   sums via matmul with the all-ones tile, then the SFPU chain  sums * mul_pre + eps -> rsqrt -> * mul_post
void row_inv_norm(uint32_t x, uint32_t n, uint32_t o, uint32_t mul_pre, uint32_t eps, uint32_t mul_post) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(1);
    reconfig_data_format<SrcOrder::Reverse>(x, cb_sel);
    matmul_init(x, cb_sel, 0);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; i++) {
        matmul_tiles(x, cb_sel, i, ONES, 0);
    }
    binop_with_scalar_tile_init();
    if (mul_pre != 0x3F800000u) {
        mul_unary_tile(0, mul_pre);
    }
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    if (mul_post != 0x3F800000u) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, mul_post);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o);
    tile_regs_release();
    cb_o.push_back(1);
}

// o[i] = A[a_off + i] * B[b_off + i] elementwise (b_bcast_same: B[b_off] for every i), n tiles
void mul_blocks(uint32_t a, uint32_t a_off, uint32_t b, uint32_t b_off, bool b_same, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
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
            pack_tile<true>(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_o.push_back(n);
}

// o[i] = A[a_off + i] * row `row` of B[b_off + i] broadcast over the rows, n tiles
void mul_bcast_row_blocks(
    uint32_t a, uint32_t a_off, uint32_t b, uint32_t b_off, uint32_t row, uint32_t o, uint32_t n) {
    CircularBuffer cb_o(o);
    cb_o.reserve_back(n);
    reconfig_data_format(a, b);
    mul_bcast_rows_init(a, b);
    for (uint32_t i0 = 0; i0 < n; i0 += DST_TILES) {
        const uint32_t nn = min_u(n - i0, DST_TILES);
        tile_regs_acquire();
        for (uint32_t j = 0; j < nn; j++) {
            mul_tiles_bcast_rows(a, b, a_off + i0 + j, b_off + i0 + j, j, row);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < nn; j++) {
            pack_tile<true>(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_o.push_back(n);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t parts = get_compile_time_arg_val(4);         // user split of a tile row over cores
    constexpr uint32_t lb_bits = get_compile_time_arg_val(5);       // gate lower bound (-5)
    constexpr uint32_t l2eps_bits = get_compile_time_arg_val(6);    // 1e-6
    constexpr uint32_t normeps_bits = get_compile_time_arg_val(7);  // o_norm eps
    constexpr uint32_t qscale_bits = get_compile_time_arg_val(8);   // K^-0.5
    constexpr uint32_t invv_bits = get_compile_time_arg_val(9);     // 1/V
    constexpr uint32_t ONE_BITS = 0x3F800000u;
    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);

    constexpr uint32_t kv = Kt * Vt;

    // fp32 everywhere by default; the bf16 CBs are reconfigured for explicitly where they are read / written
    compute_kernel_hw_startup(cb_tmp, cb_tmp, cb_tmp);

    CircularBuffer raw(cb_raw), hist(cb_hist), taps(cb_taps), gf(cb_gf), gog(cb_gog), gb(cb_gb), cc(cb_cc);
    CircularBuffer onw(cb_onw), sel(cb_sel), S(cb_S), out(cb_out), rawrm(cb_rawrm);
    CircularBuffer qn(cb_qn), kn(cb_kn), vc(cb_vc), dec(cb_dec), bcol(cb_bcol), gate(cb_gate), ht(cb_ht);
    CircularBuffer tmp(cb_tmp), inv(cb_inv), rows(cb_rows), cols(cb_cols), vread(cb_vread), u(cb_u);
    CircularBuffer sd(cb_sd), s2(cb_s2), orow(cb_orow), bs(cb_bs), selT(cb_selT), oblk(cb_oblk);

    onw.wait_front(Vt);
    sel.wait_front(33);

    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t item = item0 + it * stride;
        const uint32_t h = item / (Bt * parts);
        const uint32_t rem = item % (Bt * parts);
        const uint32_t tb = rem / parts;
        const uint32_t part = rem % parts;
        const uint32_t n_row = min_u(32u, B - tb * 32u);  // live users of this tile row
        const uint32_t up = (n_row + parts - 1) / parts;  // users per part
        const uint32_t u0 = part * up;
        const uint32_t u1 = min_u(u0 + up, n_row);
        if (u0 >= u1) {
            continue;  // nothing for this part (the reader/writer skip it the same way)
        }
        const uint32_t c = h & 31u;  // column of this head's beta logit inside its tile

        taps.wait_front(3 * Kt);
        // ------------------------------------------------------------ 1. conv + SiLU per branch
        for (uint32_t br = 0; br < 3; br++) {
            const uint32_t out_cb = (br == 0) ? cb_tmp : (br == 1 ? cb_tmp : cb_vc);
            // tilize the three history blocks of this branch (tap j -> ht tiles j*Kt ..)
            ht.reserve_back(3 * Kt);
            for (uint32_t j = 0; j < 3; j++) {
                hist.wait_front(Kt);
                reconfig_data_format_srca(cb_hist);
                tilize_init(cb_hist, Kt, cb_ht);
                tilize_block(cb_hist, Kt, cb_ht, 0, j * Kt);
                tilize_uninit(cb_hist, cb_ht);
                hist.pop_front(Kt);
            }
            ht.push_back(3 * Kt);
            ht.wait_front(3 * Kt);
            raw.wait_front(Kt);
            // conv[i] = raw[i] * taps[3] + sum_j ht[j][i] * taps[j]  (row broadcasts of the tap rows), then SiLU
            CircularBuffer cb_o(out_cb);
            cb_o.reserve_back(Kt);
            for (uint32_t i = 0; i < Kt; i++) {
                tile_regs_acquire();
                reconfig_data_format(cb_raw, cb_taps);
                mul_bcast_rows_init(cb_raw, cb_taps);
                mul_tiles_bcast_rows(cb_raw, cb_taps, i, br * Kt + i, 0, 3);
                reconfig_data_format(cb_ht, cb_taps);
                mul_bcast_rows_init(cb_ht, cb_taps);
                for (uint32_t j = 0; j < 3; j++) {
                    mul_tiles_bcast_rows(cb_ht, cb_taps, j * Kt + i, br * Kt + i, 1 + j, j);
                }
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
                add_binary_tile(0, 2, 0);
                add_binary_tile(0, 3, 0);
                silu_tile_init();
                silu_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, out_cb, i);
                tile_regs_release();
            }
            cb_o.push_back(Kt);
            ht.pop_front(3 * Kt);
            // the raw rows become the newest conv history: untilize them (bf16 RM) for the writer
            rawrm.reserve_back(Kt);
            reconfig_data_format_srca(cb_raw);
            pack_untilize_init<4, 4>(cb_raw, cb_rawrm);
            pack_untilize_block<4, 4>(cb_raw, 1, cb_rawrm);
            pack_untilize_uninit(cb_rawrm);
            pack_reconfig_data_format(cb_tmp);
            rawrm.push_back(Kt);
            raw.pop_front(Kt);
            if (br < 2) {
                // l2norm over the K columns of every user row: inv = rsqrt(sum x^2 + eps) [* K^-0.5 for q]
                tmp.wait_front(Kt);
                mul_blocks(cb_tmp, 0, cb_tmp, 0, false, cb_tmp, Kt);  // squares into the second tmp slot
                tmp.wait_front(2 * Kt);
                // the squares are tiles Kt..2Kt-1 of the tmp front
                {
                    CircularBuffer cb_i(cb_inv);
                    cb_i.reserve_back(1);
                    reconfig_data_format<SrcOrder::Reverse>(cb_tmp, cb_sel);
                    matmul_init(cb_tmp, cb_sel, 0);
                    tile_regs_acquire();
                    for (uint32_t i = 0; i < Kt; i++) {
                        matmul_tiles(cb_tmp, cb_sel, Kt + i, ONES, 0);
                    }
                    binop_with_scalar_tile_init();
                    add_unary_tile(0, l2eps_bits);
                    rsqrt_tile_init();
                    rsqrt_tile(0);
                    if (br == 0) {
                        binop_with_scalar_tile_init();
                        mul_unary_tile(0, qscale_bits);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_inv);
                    tile_regs_release();
                    cb_i.push_back(1);
                }
                inv.wait_front(1);
                mul_blocks(cb_tmp, 0, cb_inv, 0, true, br == 0 ? cb_qn : cb_kn, Kt);
                inv.pop_front(1);
                tmp.pop_front(2 * Kt);
            }
        }
        taps.pop_front(3 * Kt);
        qn.wait_front(Kt);
        kn.wait_front(Kt);
        vc.wait_front(Vt);

        // ------------------------------------------------------------ 2. gates
        gf.wait_front(Kt);
        cc.wait_front(Kt);
        {
            // f + dt_bias (row 0 of cc), then * a_exp (row 1 of cc), sigmoid, * lb, exp -> decay block
            CircularBuffer cb_t(cb_tmp);
            cb_t.reserve_back(Kt);
            reconfig_data_format(cb_gf, cb_cc);
            add_bcast_rows_init(cb_gf, cb_cc);
            tile_regs_acquire();
            for (uint32_t i = 0; i < Kt; i++) {
                add_tiles_bcast_rows(cb_gf, cb_cc, i, i, i, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < Kt; i++) {
                pack_tile<true>(i, cb_tmp, i);
            }
            tile_regs_release();
            cb_t.push_back(Kt);
            tmp.wait_front(Kt);
            CircularBuffer cb_d(cb_dec);
            cb_d.reserve_back(Kt);
            reconfig_data_format(cb_tmp, cb_cc);
            mul_bcast_rows_init(cb_tmp, cb_cc);
            tile_regs_acquire();
            for (uint32_t i = 0; i < Kt; i++) {
                mul_tiles_bcast_rows(cb_tmp, cb_cc, i, i, i, 1);
            }
            sigmoid_tile_init();
            for (uint32_t i = 0; i < Kt; i++) {
                sigmoid_tile(i);
            }
            binop_with_scalar_tile_init();
            for (uint32_t i = 0; i < Kt; i++) {
                mul_unary_tile(i, lb_bits);
            }
            exp_tile_init();
            for (uint32_t i = 0; i < Kt; i++) {
                exp_tile(i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < Kt; i++) {
                pack_tile<true>(i, cb_dec, i);
            }
            tile_regs_release();
            cb_d.push_back(Kt);
            tmp.pop_front(Kt);
        }
        gf.pop_front(Kt);
        cc.pop_front(Kt);
        dec.wait_front(Kt);
        // beta column: gb . E_c^T puts column c of the logits tile into column 0; sigmoid
        gb.wait_front(1);
        {
            CircularBuffer cb_b(cb_bcol);
            cb_b.reserve_back(1);
            reconfig_data_format<SrcOrder::Reverse>(cb_gb, cb_sel);
            matmul_init(cb_gb, cb_sel, 1);
            tile_regs_acquire();
            matmul_tiles(cb_gb, cb_sel, 0, c, 0);
            sigmoid_tile_init();
            sigmoid_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_bcol);
            tile_regs_release();
            cb_b.push_back(1);
        }
        gb.pop_front(1);
        bcol.wait_front(1);
        // output gate: sigmoid(og)
        gog.wait_front(Vt);
        {
            CircularBuffer cb_g(cb_gate);
            cb_g.reserve_back(Vt);
            reconfig_data_format_srca(cb_gog);
            copy_init(cb_gog);
            tile_regs_acquire();
            for (uint32_t i = 0; i < Vt; i++) {
                copy_tile(cb_gog, i, i);
            }
            sigmoid_tile_init();
            for (uint32_t i = 0; i < Vt; i++) {
                sigmoid_tile(i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < Vt; i++) {
                pack_tile<true>(i, cb_gate, i);
            }
            tile_regs_release();
            cb_g.push_back(Vt);
        }
        gog.pop_front(Vt);
        gate.wait_front(Vt);

        // ------------------------------------------------------------ 3. recurrence per user
        for (uint32_t r = u0; r < u1; r++) {
            // lift row r of q, k, v, decay into row 0 of tile blocks
            rows.reserve_back(4 * Kt);
            select_row(cb_qn, r, cb_rows, 0, Kt);
            select_row(cb_kn, r, cb_rows, Kt, Kt);
            select_row(cb_vc, r, cb_rows, 2 * Kt, Vt);
            select_row(cb_dec, r, cb_rows, 3 * Kt, Kt);
            rows.push_back(4 * Kt);
            rows.wait_front(4 * Kt);
            // beta_r at [0,0]
            bs.reserve_back(1);
            select_row(cb_bcol, r, cb_bs, 0, 1);
            bs.push_back(1);
            bs.wait_front(1);
            // columns: decay^T (0..Kt), k^T (Kt..2Kt)
            cols.reserve_back(2 * Kt);
            reconfig_data_format_srca(cb_rows);
            transpose_init(cb_rows);
            for (uint32_t i = 0; i < 2 * Kt; i++) {
                tile_regs_acquire();
                transpose_tile(cb_rows, (i < Kt) ? (3 * Kt + i) : (Kt + i - Kt), 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_cols, i);
                tile_regs_release();
            }
            cols.push_back(2 * Kt);
            cols.wait_front(2 * Kt);
            // sd = S * decay (rows of S scaled by the decay column)
            S.wait_front(kv);
            bcast_col_mul(cb_S, cb_cols, 0, cb_sd, Kt, Vt);
            S.pop_front(kv);
            sd.wait_front(kv);
            // vread = k . sd
            mm_row(cb_rows, Kt, cb_sd, cb_vread, Kt, Vt);
            vread.wait_front(Vt);
            // u = beta (v - vread)
            delta_rule_residual(cb_rows, 2 * Kt, cb_vread, cb_bs, cb_u, Vt);
            vread.pop_front(Vt);
            bs.pop_front(1);
            u.wait_front(Vt);
            // S' = sd + k^T (x) u  -> s2 and the state output
            rank1_update<Kt, Vt>(cb_sd, cb_cols, Kt, cb_u, cb_s2);
            u.pop_front(Vt);
            sd.pop_front(kv);
            cols.pop_front(2 * Kt);
            s2.wait_front(kv);
            // o_r = q . S'
            mm_row(cb_rows, 0, cb_s2, cb_orow, Kt, Vt);
            s2.pop_front(kv);
            rows.pop_front(4 * Kt);
            orow.wait_front(Vt);
            // accumulate row r of the output block: oblk += E_r^T . o_r
            transpose_block(cb_sel, r, cb_selT, 1);
            selT.wait_front(1);
            oblk.reserve_back(Vt);
            tile_regs_acquire();
            if (r > u0) {
                reconfig_data_format_srca(cb_oblk);
                copy_init(cb_oblk);
                for (uint32_t i = 0; i < Vt; i++) {
                    copy_tile(cb_oblk, i, i);
                }
            }
            reconfig_data_format<SrcOrder::Reverse>(cb_selT, cb_orow);
            matmul_init(cb_selT, cb_orow, 0);
            for (uint32_t i = 0; i < Vt; i++) {
                matmul_tiles(cb_selT, cb_orow, 0, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < Vt; i++) {
                pack_tile<true>(i, cb_oblk, i);
            }
            tile_regs_release();
            oblk.push_back(Vt);
            if (r > u0) {
                oblk.pop_front(Vt);
            }
            selT.pop_front(1);
            orow.pop_front(Vt);
        }
        qn.pop_front(Kt);
        kn.pop_front(Kt);
        vc.pop_front(Vt);
        dec.pop_front(Kt);
        bcol.pop_front(1);

        // ------------------------------------------------------------ 4. gated RMSNorm of the output block
        oblk.wait_front(Vt);
        mul_blocks(cb_oblk, 0, cb_oblk, 0, false, cb_tmp, Vt);  // squares
        tmp.wait_front(Vt);
        row_inv_norm(cb_tmp, Vt, cb_inv, invv_bits, normeps_bits, ONE_BITS);  // rsqrt(mean(o^2) + eps)
        tmp.pop_front(Vt);
        inv.wait_front(1);
        mul_blocks(cb_oblk, 0, cb_inv, 0, true, cb_tmp, Vt);  // o * inv
        inv.pop_front(1);
        oblk.pop_front(Vt);
        tmp.wait_front(Vt);
        mul_bcast_row_blocks(cb_tmp, 0, cb_onw, 0, 0, cb_tmp, Vt);  // * w  (second tmp slot)
        tmp.wait_front(2 * Vt);
        {
            // * sigmoid(og) -> oblk (fp32 tiles), then untilized into the bf16 RM output block (row r = user r)
            oblk.reserve_back(Vt);
            reconfig_data_format(cb_tmp, cb_gate);
            mul_init(cb_tmp, cb_gate);
            tile_regs_acquire();
            for (uint32_t i = 0; i < Vt; i++) {
                mul_tiles(cb_tmp, cb_gate, Vt + i, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < Vt; i++) {
                pack_tile<true>(i, cb_oblk, i);
            }
            tile_regs_release();
            oblk.push_back(Vt);
            oblk.wait_front(Vt);
            out.reserve_back(Vt);
            reconfig_data_format_srca(cb_oblk);
            pack_untilize_init<4, 4>(cb_oblk, cb_out);
            pack_untilize_block<4, 4>(cb_oblk, 1, cb_out);
            pack_untilize_uninit(cb_out);
            pack_reconfig_data_format(cb_tmp);
            out.push_back(Vt);
            oblk.pop_front(Vt);
        }
        tmp.pop_front(2 * Vt);
        gate.pop_front(Vt);
    }
}
