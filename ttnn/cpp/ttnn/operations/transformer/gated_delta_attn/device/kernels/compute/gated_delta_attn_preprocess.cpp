// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused GDN preprocessing (production decay path). Per (head, chunk) work item it emits the eight
// tensors consumed by gated_delta_attn_seq, replacing the whole Python preamble:
//
//   decay_col   = lower_causal @ g            (inclusive prefix-sum of the gate, as a [C,1] column)
//   decay_exp   = exp(clamp(decay_col))
//   decay_row   = decay_col^T                 ([1,C])
//   L_mask      = tril * exp(clamp(decay_col - decay_row))
//   kk          = (k*beta) @ k^T
//   L_mat       = eye + kk*L_mask - (1-alpha)*diag(kk*L_mask)
//   D_inv       = 1 / diag(L_mat)
//   L_unit      = eye + D_inv * (L_mat - diag(L_mat))
//   v_beta_sc   = D_inv * (v*beta)
//   k_bd_sc     = D_inv * (k*beta*decay_exp)
//   q_decay     = q * decay_exp
//   dl_raw      = ones @ g                     (total gate sum, broadcast)
//   dl_exp      = exp(clamp(dl_raw))           ([1,1] state-decay scalar)
//   k_decay_t   = (k * exp(clamp(dl_raw - decay_col)))^T
//   intra_attn  = (q @ k^T) * L_mask
//   L_inv       = diagonal 32x32 block inverses of L_unit (Horner forward-substitution)
//
// The `clamp` matches the Python clip(x,-20,0): its max=0 is what keeps the upper triangle of
// exp(decay_col-decay_row) finite before the *tril mask zeros it.
//
// NOTE (multi-work-item): this kernel is only correct when each core processes exactly ONE work
// item. Several matmul/reduce/L_inv sequences leave HW/CB state that corrupts a 2nd work item on the
// same core. The host op therefore splits the launch so num_work <= num_cores (see program factory /
// the Python caller), guaranteeing one work item per core.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_q = tt::CBIndex::c_0;
constexpr uint32_t cb_k = tt::CBIndex::c_1;
constexpr uint32_t cb_v = tt::CBIndex::c_2;
constexpr uint32_t cb_beta = tt::CBIndex::c_3;
constexpr uint32_t cb_g = tt::CBIndex::c_4;
constexpr uint32_t cb_triu = tt::CBIndex::c_5;  // unused (prefix-sum uses lower_causal @ g)
constexpr uint32_t cb_tril = tt::CBIndex::c_6;
constexpr uint32_t cb_eye = tt::CBIndex::c_7;
constexpr uint32_t cb_lower = tt::CBIndex::c_8;
constexpr uint32_t cb_eye32 = tt::CBIndex::c_9;

constexpr uint32_t cb_L_unit = tt::CBIndex::c_10;
constexpr uint32_t cb_v_beta_sc = tt::CBIndex::c_11;
constexpr uint32_t cb_k_bd_sc = tt::CBIndex::c_12;
constexpr uint32_t cb_intra = tt::CBIndex::c_13;
constexpr uint32_t cb_q_decay = tt::CBIndex::c_14;
constexpr uint32_t cb_k_decay_t = tt::CBIndex::c_15;
constexpr uint32_t cb_dl_exp = tt::CBIndex::c_16;
constexpr uint32_t cb_L_inv = tt::CBIndex::c_17;

// 16-tile scratch
constexpr uint32_t cb_s0 = tt::CBIndex::c_18;
constexpr uint32_t cb_s1 = tt::CBIndex::c_19;
constexpr uint32_t cb_s2 = tt::CBIndex::c_20;
// decay-path state
constexpr uint32_t cb_ones = tt::CBIndex::c_21;   // all-ones tile
constexpr uint32_t cb_dcol = tt::CBIndex::c_22;   // decay_col [C,1]
constexpr uint32_t cb_drow = tt::CBIndex::c_23;   // decay_row [1,C]
constexpr uint32_t cb_dexp = tt::CBIndex::c_24;   // decay_exp [C,1]
constexpr uint32_t cb_dinv = tt::CBIndex::c_25;   // dl_raw temp, then D_inv [C,1]
constexpr uint32_t cb_lmask = tt::CBIndex::c_26;  // L_mask [C,C]
constexpr uint32_t cb_ktraw = tt::CBIndex::c_27;  // k^T [K,C]
constexpr uint32_t cb_ddexp = tt::CBIndex::c_28;  // decay_diff_exp [C,1]

// clip(x, -20, 0) bounds as fp32 bit patterns.
constexpr uint32_t k_neg20_bits = 0xC1A00000u;  // -20.0f
constexpr uint32_t k_zero_bits = 0x00000000u;   //  0.0f

// out[0..M*N) = A @ B (standard blocked matmul over K inner tiles). B tile index is n*K+k when
// b_grid_transposed, else k*N+n. Reserves/pushes out; A,B must be waited by the caller.
__attribute__((noinline)) static void mm_block(
    uint32_t a, uint32_t b, uint32_t out, uint32_t M, uint32_t N, uint32_t K, bool b_grid_transposed) {
    mm_init(a, b, out);
    CircularBuffer(out).reserve_back(M * N);
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < K; k++) {
                uint32_t bt = b_grid_transposed ? (n * K + k) : (k * N + n);
                matmul_tiles(a, b, m * K + k, bt, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out, m * N + n);
            tile_regs_release();
        }
    }
    CircularBuffer(out).push_back(M * N);
}

// Elementwise binary out[t] = a[t] <op> b[t] for t in [0,n). Reserves/pushes out.
enum class BinOp { Mul, Sub, Add };
__attribute__((noinline)) static void ew_binary(uint32_t a, uint32_t b, uint32_t out, uint32_t n, BinOp op) {
    CircularBuffer(out).reserve_back(n);
    if (op == BinOp::Mul) {
        mul_tiles_init(a, b);
    } else if (op == BinOp::Sub) {
        sub_tiles_init(a, b);
    } else {
        add_tiles_init(a, b);
    }
    for (uint32_t t = 0; t < n; t++) {
        tile_regs_acquire();
        if (op == BinOp::Mul) {
            mul_tiles(a, b, t, t, 0);
        } else if (op == BinOp::Sub) {
            sub_tiles(a, b, t, t, 0);
        } else {
            add_tiles(a, b, t, t, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, t);
        tile_regs_release();
    }
    CircularBuffer(out).push_back(n);
}

// out[C,X] = a[C,X] * vec[C,1] (per-row scale; vec broadcast across columns). Reserves/pushes out.
__attribute__((noinline)) static void mul_bcast_col(uint32_t a, uint32_t vec, uint32_t out, uint32_t Ct, uint32_t Xt) {
    CircularBuffer(out).reserve_back(Ct * Xt);
    mul_bcast_cols_init_short(a, vec);
    for (uint32_t ct = 0; ct < Ct; ct++) {
        for (uint32_t xt = 0; xt < Xt; xt++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(a, vec, ct * Xt + xt, ct, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out, ct * Xt + xt);
            tile_regs_release();
        }
    }
    CircularBuffer(out).push_back(Ct * Xt);
}

// out[0..n) = exp(clamp(src[t], -20, 0)). Reserves/pushes out; src must be waited by the caller.
__attribute__((noinline)) static void exp_clamp(uint32_t src, uint32_t out, uint32_t n) {
    CircularBuffer(out).reserve_back(n);
    for (uint32_t t = 0; t < n; t++) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(src);
        copy_tile(src, t, 0);
        clamp_tile_init();
        clamp_tile(0, k_neg20_bits, k_zero_bits);
        exp_tile_init();
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out, t);
        tile_regs_release();
    }
    CircularBuffer(out).push_back(n);
}

// out[K,C] = transpose(src[C,K]). Reserves/pushes out; src must be waited by the caller.
__attribute__((noinline)) static void transpose_block(uint32_t src, uint32_t out, uint32_t Ct, uint32_t Kt) {
    CircularBuffer(out).reserve_back(Kt * Ct);
    transpose_init(src);
    for (uint32_t kt = 0; kt < Kt; kt++) {
        for (uint32_t ct = 0; ct < Ct; ct++) {
            tile_regs_acquire();
            transpose_tile(src, ct * Kt + kt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out, kt * Ct + ct);
            tile_regs_release();
        }
    }
    CircularBuffer(out).push_back(Kt * Ct);
}

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);
    constexpr uint32_t alpha_bits = get_compile_time_arg_val(3);
    constexpr uint32_t one_bits = 0x3F800000u;  // 1.0f

    const uint32_t core_work_start = get_arg_val<uint32_t>(0);
    const uint32_t num_work = get_arg_val<uint32_t>(1);
    const uint32_t work_stride = get_arg_val<uint32_t>(2);

    constexpr uint32_t attn_tiles = Ct * Ct;
    constexpr uint32_t out_tiles = Ct * Vt;
    constexpr uint32_t in_kv_tiles = Ct * Kt;
    constexpr uint32_t kdt_tiles = Kt * Ct;
    constexpr uint32_t beta_tiles = Ct;
    constexpr uint32_t g_tiles = Ct;

    // Matmul HW must be primed before the first copy_tile_to_dst_init_short.
    mm_init(cb_q, cb_k, cb_L_unit);

    for (uint32_t work = core_work_start; work < num_work; work += work_stride) {
        CircularBuffer(cb_q).wait_front(in_kv_tiles);
        CircularBuffer(cb_k).wait_front(in_kv_tiles);
        CircularBuffer(cb_v).wait_front(out_tiles);
        CircularBuffer(cb_beta).wait_front(beta_tiles);
        CircularBuffer(cb_g).wait_front(g_tiles);
        CircularBuffer(cb_triu).wait_front(attn_tiles);  // tile (0,1) reused as the all-ones tile
        CircularBuffer(cb_tril).wait_front(attn_tiles);
        CircularBuffer(cb_eye).wait_front(attn_tiles);
        CircularBuffer(cb_lower).wait_front(attn_tiles);
        CircularBuffer(cb_eye32).wait_front(1);

        // Build Ct all-ones tiles from triu tile (0,1) (index 1) — entirely ones and host-provided,
        // so reliable for reductions/broadcasts (in-kernel fill_tile leaves the tile interior zero).
        CircularBuffer(cb_ones).reserve_back(Ct);
        copy_tile_to_dst_init_short(cb_triu);
        for (uint32_t m = 0; m < Ct; m++) {
            tile_regs_acquire();
            copy_tile(cb_triu, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_ones, m);
            tile_regs_release();
        }
        CircularBuffer(cb_ones).push_back(Ct);
        CircularBuffer(cb_ones).wait_front(Ct);

        // ---- decay_col = lower_causal @ g  ([C,1] inclusive prefix-sum) ----
        mm_block(cb_lower, cb_g, cb_dcol, Ct, 1, Ct, /*b_grid_transposed=*/false);
        CircularBuffer(cb_dcol).wait_front(Ct);

        // ---- decay_exp = exp(clamp(decay_col)) ----
        exp_clamp(cb_dcol, cb_dexp, Ct);
        CircularBuffer(cb_dexp).wait_front(Ct);

        // ---- decay_row = transpose(decay_col)  ([1,C]) ----
        CircularBuffer(cb_drow).reserve_back(Ct);
        transpose_init(cb_dcol);
        for (uint32_t i = 0; i < Ct; i++) {
            tile_regs_acquire();
            transpose_tile(cb_dcol, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_drow, i);
            tile_regs_release();
        }
        CircularBuffer(cb_drow).push_back(Ct);
        CircularBuffer(cb_drow).wait_front(Ct);

        // ---- L_mask = tril * exp(clamp(decay_col - decay_row)) ----
        // Mcol[h,w]=decay_col[h] (outer product with ones); Mrow[h,w]=decay_row[w].
        mm_init(cb_dcol, cb_ones, cb_s0);
        CircularBuffer(cb_s0).reserve_back(attn_tiles);
        for (uint32_t m = 0; m < Ct; m++) {
            for (uint32_t n = 0; n < Ct; n++) {
                tile_regs_acquire();
                matmul_tiles(cb_dcol, cb_ones, m, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_s0, m * Ct + n);
                tile_regs_release();
            }
        }
        CircularBuffer(cb_s0).push_back(attn_tiles);

        mm_init(cb_ones, cb_drow, cb_s1);
        CircularBuffer(cb_s1).reserve_back(attn_tiles);
        for (uint32_t m = 0; m < Ct; m++) {
            for (uint32_t n = 0; n < Ct; n++) {
                tile_regs_acquire();
                matmul_tiles(cb_ones, cb_drow, 0, n, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_s1, m * Ct + n);
                tile_regs_release();
            }
        }
        CircularBuffer(cb_s1).push_back(attn_tiles);

        CircularBuffer(cb_s0).wait_front(attn_tiles);
        CircularBuffer(cb_s1).wait_front(attn_tiles);
        ew_binary(cb_s0, cb_s1, cb_s2, attn_tiles, BinOp::Sub);  // L_diff
        CircularBuffer(cb_s0).pop_front(attn_tiles);
        CircularBuffer(cb_s1).pop_front(attn_tiles);
        CircularBuffer(cb_drow).pop_front(Ct);

        CircularBuffer(cb_s2).wait_front(attn_tiles);
        exp_clamp(cb_s2, cb_s0, attn_tiles);  // exp(clamp(L_diff))
        CircularBuffer(cb_s2).pop_front(attn_tiles);

        CircularBuffer(cb_s0).wait_front(attn_tiles);
        ew_binary(cb_s0, cb_tril, cb_lmask, attn_tiles, BinOp::Mul);  // * tril
        CircularBuffer(cb_s0).pop_front(attn_tiles);
        CircularBuffer(cb_lmask).wait_front(attn_tiles);

        // ---- dl_raw = ones @ g  (total gate sum, replicated across rows) ----
        mm_init(cb_ones, cb_g, cb_dinv);
        CircularBuffer(cb_dinv).reserve_back(Ct);
        for (uint32_t m = 0; m < Ct; m++) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Ct; k++) {
                matmul_tiles(cb_ones, cb_g, 0, k, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_dinv, m);
            tile_regs_release();
        }
        CircularBuffer(cb_dinv).push_back(Ct);
        CircularBuffer(cb_dinv).wait_front(Ct);

        // ---- dl_exp = exp(clamp(dl_raw))  ([1,1] scalar at [0,0]) ----
        exp_clamp(cb_dinv, cb_dl_exp, 1);

        // ---- decay_diff_exp = exp(clamp(dl_raw - decay_col)) ----
        ew_binary(cb_dinv, cb_dcol, cb_s2, Ct, BinOp::Sub);  // decay_diff [C,1]
        CircularBuffer(cb_dinv).pop_front(Ct);               // dl_raw done
        CircularBuffer(cb_dcol).pop_front(Ct);               // decay_col done
        CircularBuffer(cb_s2).wait_front(Ct);
        exp_clamp(cb_s2, cb_ddexp, Ct);
        CircularBuffer(cb_s2).pop_front(Ct);
        CircularBuffer(cb_ddexp).wait_front(Ct);

        // ---- k^T (raw), kept for kk and intra_attn ----
        transpose_block(cb_k, cb_ktraw, Ct, Kt);
        CircularBuffer(cb_ktraw).wait_front(kdt_tiles);

        // ---- k_beta = k * beta -> s0 ----
        mul_bcast_col(cb_k, cb_beta, cb_s0, Ct, Kt);
        CircularBuffer(cb_s0).wait_front(in_kv_tiles);

        // ---- dkk = diag(kk) = rowsum(k_beta ⊙ k), via a DENSE matmul(P, ones) row-reduction. ----
        // (Reducing the sparse diagonal matrix D_mat = L_mat⊙eye mis-packs to zero on this arch;
        //  keeping the reduced operand dense avoids that. dkk lands in cb_dcol [C,1], all cols = diag[r].)
        ew_binary(cb_s0, cb_k, cb_s1, in_kv_tiles, BinOp::Mul);  // P = k_beta ⊙ k -> s1
        CircularBuffer(cb_s1).wait_front(in_kv_tiles);
        mm_init(cb_s1, cb_ones, cb_dcol);
        CircularBuffer(cb_dcol).reserve_back(Ct);
        for (uint32_t m = 0; m < Ct; m++) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; k++) {
                matmul_tiles(cb_s1, cb_ones, m * Kt + k, k, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_dcol, m);
            tile_regs_release();
        }
        CircularBuffer(cb_dcol).push_back(Ct);
        CircularBuffer(cb_s1).pop_front(in_kv_tiles);  // P done
        CircularBuffer(cb_dcol).wait_front(Ct);        // dkk = diag(kk)

        // ---- kk = k_beta @ k^T -> s1 ----
        mm_block(cb_s0, cb_ktraw, cb_s1, Ct, Ct, Kt, /*b_grid_transposed=*/false);
        CircularBuffer(cb_s0).pop_front(in_kv_tiles);  // k_beta done

        // ---- kk_lmask = kk ⊙ L_mask -> s0 ----
        CircularBuffer(cb_s1).wait_front(attn_tiles);
        ew_binary(cb_s1, cb_lmask, cb_s0, attn_tiles, BinOp::Mul);
        CircularBuffer(cb_s1).pop_front(attn_tiles);
        CircularBuffer(cb_s0).wait_front(attn_tiles);

        // ---- L_strict = off-diagonal(kk_lmask) = kk_lmask - kk_lmask⊙eye -> s2 (kept).
        //      (= off-diagonal of L_mat, since diagonal regularization only affects the diagonal). ----
        ew_binary(cb_s0, cb_eye, cb_s1, attn_tiles, BinOp::Mul);  // kk_lmask ⊙ eye -> s1
        CircularBuffer(cb_s1).wait_front(attn_tiles);
        ew_binary(cb_s0, cb_s1, cb_s2, attn_tiles, BinOp::Sub);  // L_strict -> s2
        CircularBuffer(cb_s0).pop_front(attn_tiles);             // kk_lmask done
        CircularBuffer(cb_s1).pop_front(attn_tiles);
        CircularBuffer(cb_s2).wait_front(attn_tiles);

        // ---- D_inv = 1 / (1 + alpha*dkk) -> cb_dinv [C,1] (fused SFPU on dkk in DST; avoids the
        //      unreliable in-kernel fill_tile scalar). ----
        CircularBuffer(cb_dinv).reserve_back(Ct);
        for (uint32_t m = 0; m < Ct; m++) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_dcol);
            copy_tile(cb_dcol, m, 0);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, alpha_bits);  // alpha*dkk
            add_unary_tile(0, one_bits);    // 1 + alpha*dkk
            recip_tile_init();
            recip_tile(0);  // 1 / (1 + alpha*dkk)
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_dinv, m);
            tile_regs_release();
        }
        CircularBuffer(cb_dinv).push_back(Ct);
        CircularBuffer(cb_dcol).pop_front(Ct);
        CircularBuffer(cb_dinv).wait_front(Ct);

        // ---- N = D_inv * L_strict;  L_unit = eye + N ----
        mul_bcast_col(cb_s2, cb_dinv, cb_s0, Ct, Ct);  // N -> s0
        CircularBuffer(cb_s2).pop_front(attn_tiles);
        CircularBuffer(cb_s0).wait_front(attn_tiles);
        ew_binary(cb_s0, cb_eye, cb_L_unit, attn_tiles, BinOp::Add);  // L_unit -> output
        CircularBuffer(cb_s0).pop_front(attn_tiles);
        CircularBuffer(cb_L_unit).wait_front(attn_tiles);

        // ---- v_beta_sc = D_inv * (v*beta) ----
        mul_bcast_col(cb_v, cb_beta, cb_s0, Ct, Vt);  // v_beta
        CircularBuffer(cb_s0).wait_front(out_tiles);
        mul_bcast_col(cb_s0, cb_dinv, cb_v_beta_sc, Ct, Vt);
        CircularBuffer(cb_s0).pop_front(out_tiles);

        // ---- k_bd_sc = D_inv * (k*beta*decay_exp) ----
        mul_bcast_col(cb_k, cb_beta, cb_s0, Ct, Kt);  // k_beta
        CircularBuffer(cb_s0).wait_front(in_kv_tiles);
        mul_bcast_col(cb_s0, cb_dexp, cb_s1, Ct, Kt);  // *decay_exp
        CircularBuffer(cb_s0).pop_front(in_kv_tiles);
        CircularBuffer(cb_s1).wait_front(in_kv_tiles);
        mul_bcast_col(cb_s1, cb_dinv, cb_k_bd_sc, Ct, Kt);  // *D_inv
        CircularBuffer(cb_s1).pop_front(in_kv_tiles);
        CircularBuffer(cb_dinv).pop_front(Ct);  // D_inv done

        // ---- q_decay = q * decay_exp ----
        mul_bcast_col(cb_q, cb_dexp, cb_q_decay, Ct, Kt);

        // ---- k_decay_t = (k * decay_diff_exp)^T ----
        mul_bcast_col(cb_k, cb_ddexp, cb_s0, Ct, Kt);  // k_decay
        CircularBuffer(cb_s0).wait_front(in_kv_tiles);
        transpose_block(cb_s0, cb_k_decay_t, Ct, Kt);
        CircularBuffer(cb_s0).pop_front(in_kv_tiles);
        CircularBuffer(cb_ddexp).pop_front(Ct);

        // ---- intra_attn = (q @ k^T) * L_mask ----
        mm_block(cb_q, cb_ktraw, cb_s0, Ct, Ct, Kt, /*b_grid_transposed=*/false);  // qk
        CircularBuffer(cb_s0).wait_front(attn_tiles);
        ew_binary(cb_s0, cb_lmask, cb_intra, attn_tiles, BinOp::Mul);
        CircularBuffer(cb_s0).pop_front(attn_tiles);
        CircularBuffer(cb_lmask).pop_front(attn_tiles);
        CircularBuffer(cb_ktraw).pop_front(kdt_tiles);
        CircularBuffer(cb_dexp).pop_front(Ct);

        // ==================================================================
        // L_inv: diagonal 32x32 block inverses of L_unit via Horner forward substitution.
        //   For each block: neg_N = I - L_block; R = I + neg_N; R <- I + neg_N @ R (30 iters).
        // ==================================================================
        CircularBuffer(cb_L_inv).reserve_back(Ct);
        for (uint32_t b = 0; b < Ct; b++) {
            const uint32_t diag_tile = b * Ct + b;

            // cb_s0 holds neg_N for this block.
            CircularBuffer(cb_s0).reserve_back(1);
            sub_tiles_init(cb_eye32, cb_L_unit);
            tile_regs_acquire();
            sub_tiles(cb_eye32, cb_L_unit, 0, diag_tile, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_s0, 0);
            tile_regs_release();
            CircularBuffer(cb_s0).push_back(1);

            // cb_s1 holds current R.
            CircularBuffer(cb_s0).wait_front(1);
            CircularBuffer(cb_s1).reserve_back(1);
            add_tiles_init(cb_eye32, cb_s0);
            tile_regs_acquire();
            add_tiles(cb_eye32, cb_s0, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_s1, 0);
            tile_regs_release();
            CircularBuffer(cb_s1).push_back(1);

            for (uint32_t iter = 0; iter < 30; iter++) {
                CircularBuffer(cb_s1).wait_front(1);
                CircularBuffer(cb_s2).reserve_back(1);
                mm_init(cb_s0, cb_s1, cb_s2);
                tile_regs_acquire();
                matmul_tiles(cb_s0, cb_s1, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_s2, 0);
                tile_regs_release();
                CircularBuffer(cb_s2).push_back(1);
                CircularBuffer(cb_s1).pop_front(1);

                CircularBuffer(cb_s2).wait_front(1);
                CircularBuffer(cb_s1).reserve_back(1);
                add_tiles_init(cb_eye32, cb_s2);
                tile_regs_acquire();
                add_tiles(cb_eye32, cb_s2, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_s1, 0);
                tile_regs_release();
                CircularBuffer(cb_s1).push_back(1);
                CircularBuffer(cb_s2).pop_front(1);
            }

            CircularBuffer(cb_s1).wait_front(1);
            copy_tile_to_dst_init_short(cb_s1);
            tile_regs_acquire();
            copy_tile(cb_s1, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_L_inv, b);
            tile_regs_release();
            CircularBuffer(cb_s1).pop_front(1);
            CircularBuffer(cb_s0).pop_front(1);
        }
        CircularBuffer(cb_L_inv).push_back(Ct);

        // Pop raw inputs/masks after all consumers are done.
        CircularBuffer(cb_q).pop_front(in_kv_tiles);
        CircularBuffer(cb_k).pop_front(in_kv_tiles);
        CircularBuffer(cb_v).pop_front(out_tiles);
        CircularBuffer(cb_beta).pop_front(beta_tiles);
        CircularBuffer(cb_g).pop_front(g_tiles);
        CircularBuffer(cb_triu).pop_front(attn_tiles);
        CircularBuffer(cb_tril).pop_front(attn_tiles);
        CircularBuffer(cb_eye).pop_front(attn_tiles);
        CircularBuffer(cb_lower).pop_front(attn_tiles);
        CircularBuffer(cb_eye32).pop_front(1);
        CircularBuffer(cb_ones).pop_front(Ct);
    }
}
