// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel (Path A): forward substitution + sequential state scan.
//
// Python preprocessing provides L_inv (4 precomputed diagonal block inverses per chunk).
// This kernel performs:
//   PA2. Forward substitution: v_cor = L_unit^{-1} @ v_beta_sc  (using L_inv from reader)
//   PA3. Forward substitution: k_cum = L_unit^{-1} @ k_bd_sc
//
//   [7-step state update]
//   1. v_prime = k_cum @ S
//   2. v_new   = v_cor - v_prime
//   3. o_inter = q_decay @ S
//   4. intra_v = intra_attn @ v_new
//   5. out     = o_inter + intra_v
//   6. s_upd   = k_decay_t @ v_new
//   7. S       = S * dl_exp + s_upd
//
// Compile-time args: Ct, Kt, Vt  (all must be 4 for chunk_size=key_dim=val_dim=128)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/bcast.h"

// CB indices (must match program factory)
constexpr uint32_t cb_L_unit = tt::CBIndex::c_0;
constexpr uint32_t cb_v_beta_sc = tt::CBIndex::c_1;
constexpr uint32_t cb_k_bd_sc = tt::CBIndex::c_2;
constexpr uint32_t cb_intra_att = tt::CBIndex::c_3;
constexpr uint32_t cb_q_decay = tt::CBIndex::c_4;
constexpr uint32_t cb_k_dt = tt::CBIndex::c_5;
constexpr uint32_t cb_dl_exp = tt::CBIndex::c_6;
// c_7: unused (was identity_32)
constexpr uint32_t cb_S = tt::CBIndex::c_8;  // persistent state

constexpr uint32_t cb_nm_P_a = tt::CBIndex::c_9;   // fwd_rhs scratch
constexpr uint32_t cb_nm_P_b = tt::CBIndex::c_10;  // corr_mm scratch
constexpr uint32_t cb_nm_R_a = tt::CBIndex::c_11;  // temp_rhs scratch
// c_12, c_13: unused (were Neumann-only)

// L_inv diagonal block inverses — pre-loaded per chunk by reader
constexpr uint32_t cb_L_inv_0 = tt::CBIndex::c_14;
constexpr uint32_t cb_L_inv_1 = tt::CBIndex::c_15;
constexpr uint32_t cb_L_inv_2 = tt::CBIndex::c_16;
constexpr uint32_t cb_L_inv_3 = tt::CBIndex::c_17;

constexpr uint32_t cb_v_cor = tt::CBIndex::c_18;
constexpr uint32_t cb_k_cum = tt::CBIndex::c_19;

constexpr uint32_t cb_v_prime = tt::CBIndex::c_20;
constexpr uint32_t cb_v_new = tt::CBIndex::c_21;
constexpr uint32_t cb_o_inter = tt::CBIndex::c_22;
constexpr uint32_t cb_intra_v = tt::CBIndex::c_23;
constexpr uint32_t cb_out = tt::CBIndex::c_24;
constexpr uint32_t cb_s_upd = tt::CBIndex::c_25;
constexpr uint32_t cb_S_tmp = tt::CBIndex::c_26;
constexpr uint32_t cb_final_state = tt::CBIndex::c_27;

// ---------------------------------------------------------------------------
// One row of blocked forward substitution.
//
// Computes row `row_i` of:  out = L_unit^{-1} @ rhs
//
// At entry: out_cb already has rows 0..row_i-1 pushed (Xt tiles each).
// This function pushes Xt more tiles (row_i) to out_cb.
//
// cb_L_unit:     holds Ct*Ct tiles (full L_unit for this chunk), available via wait.
// rhs_cb:        holds Ct*Xt tiles.
// cb_L_inv_row_i: 1 tile = L^{-1}[row_i,row_i], waited on externally (not popped here).
// Scratch: cb_nm_P_a (fwd_rhs Xt tiles), cb_nm_P_b (corr Xt tiles), cb_nm_R_a (temp Xt tiles).
// ---------------------------------------------------------------------------
__attribute__((noinline)) static void fwd_sub_row(
    uint32_t row_i, uint32_t Ct, uint32_t Xt, uint32_t rhs_cb, uint32_t out_cb, uint32_t cb_L_inv_row_i) {
    // Previous rows must be available for the subtraction loop.
    if (row_i > 0) {
        cb_wait_front(out_cb, row_i * Xt);
    }

    // Step 1: fwd_rhs = rhs_cb[row_i * Xt .. (row_i+1)*Xt - 1]
    cb_reserve_back(cb_nm_P_a, Xt);
    copy_tile_to_dst_init_short(rhs_cb);
    for (uint32_t xt = 0; xt < Xt; xt++) {
        tile_regs_acquire();
        copy_tile(rhs_cb, row_i * Xt + xt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_nm_P_a, xt);
        tile_regs_release();
    }
    cb_push_back(cb_nm_P_a, Xt);

    // Step 2: fwd_rhs -= L[row_i, j] @ out[row_j] for each j < row_i
    for (uint32_t j = 0; j < row_i; j++) {
        // corr = L_unit[row_i*Ct + j] @ out_cb[j*Xt .. (j+1)*Xt-1]
        cb_reserve_back(cb_nm_P_b, Xt);
        mm_init(cb_L_unit, out_cb, cb_nm_P_b);
        uint32_t L_tile = row_i * Ct + j;
        for (uint32_t xt = 0; xt < Xt; xt++) {
            tile_regs_acquire();
            matmul_tiles(cb_L_unit, out_cb, L_tile, j * Xt + xt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_nm_P_b, xt);
            tile_regs_release();
        }
        cb_push_back(cb_nm_P_b, Xt);

        // fwd_rhs = fwd_rhs - corr  (via temp nm_R_a)
        cb_wait_front(cb_nm_P_a, Xt);
        cb_wait_front(cb_nm_P_b, Xt);
        cb_reserve_back(cb_nm_R_a, Xt);
        sub_tiles_init(cb_nm_P_a, cb_nm_P_b);
        for (uint32_t xt = 0; xt < Xt; xt++) {
            tile_regs_acquire();
            sub_tiles(cb_nm_P_a, cb_nm_P_b, xt, xt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_nm_R_a, xt);
            tile_regs_release();
        }
        cb_push_back(cb_nm_R_a, Xt);
        cb_pop_front(cb_nm_P_a, Xt);
        cb_pop_front(cb_nm_P_b, Xt);

        // fwd_rhs (nm_P_a) = nm_R_a
        cb_wait_front(cb_nm_R_a, Xt);
        cb_reserve_back(cb_nm_P_a, Xt);
        copy_tile_to_dst_init_short(cb_nm_R_a);
        for (uint32_t xt = 0; xt < Xt; xt++) {
            tile_regs_acquire();
            copy_tile(cb_nm_R_a, xt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_nm_P_a, xt);
            tile_regs_release();
        }
        cb_push_back(cb_nm_P_a, Xt);
        cb_pop_front(cb_nm_R_a, Xt);
    }

    // Step 3: out[row_i * Xt .. (row_i+1)*Xt-1] = L_inv_ii @ fwd_rhs
    cb_wait_front(cb_nm_P_a, Xt);
    cb_wait_front(cb_L_inv_row_i, 1);
    cb_reserve_back(out_cb, Xt);
    mm_init(cb_L_inv_row_i, cb_nm_P_a, out_cb);
    for (uint32_t xt = 0; xt < Xt; xt++) {
        tile_regs_acquire();
        matmul_tiles(cb_L_inv_row_i, cb_nm_P_a, 0, xt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb, xt);
        tile_regs_release();
    }
    cb_push_back(out_cb, Xt);
    cb_pop_front(cb_nm_P_a, Xt);
    // Caller pops cb_L_inv_row_i (if pop_inv=true).
}

// Forward-sub all 4 block rows.
// pop_inv: if true, pop CB14-17 after use (second pass); if false, keep for next pass.
__attribute__((noinline)) static void fwd_sub_4rows(
    uint32_t Ct,
    uint32_t Xt,
    uint32_t rhs_cb,
    uint32_t out_cb,
    uint32_t inv0,
    uint32_t inv1,
    uint32_t inv2,
    uint32_t inv3,
    bool pop_inv) {
    cb_wait_front(inv0, 1);
    fwd_sub_row(0, Ct, Xt, rhs_cb, out_cb, inv0);
    if (pop_inv) {
        cb_pop_front(inv0, 1);
    }

    cb_wait_front(inv1, 1);
    fwd_sub_row(1, Ct, Xt, rhs_cb, out_cb, inv1);
    if (pop_inv) {
        cb_pop_front(inv1, 1);
    }

    cb_wait_front(inv2, 1);
    fwd_sub_row(2, Ct, Xt, rhs_cb, out_cb, inv2);
    if (pop_inv) {
        cb_pop_front(inv2, 1);
    }

    cb_wait_front(inv3, 1);
    fwd_sub_row(3, Ct, Xt, rhs_cb, out_cb, inv3);
    if (pop_inv) {
        cb_pop_front(inv3, 1);
    }
}

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t Kt = get_compile_time_arg_val(1);
    constexpr uint32_t Vt = get_compile_time_arg_val(2);

    const uint32_t num_chunks = get_arg_val<uint32_t>(0);

    constexpr uint32_t state_tiles = Kt * Vt;
    constexpr uint32_t out_tiles = Ct * Vt;
    constexpr uint32_t in_kv_tiles = Ct * Kt;
    constexpr uint32_t attn_tiles = Ct * Ct;
    constexpr uint32_t kdt_tiles = Kt * Ct;

    // Pre-configure hardware UNPACK format registers for float32.
    // TT Metal requires mm_init before any copy_tile_to_dst_init_short call.
    // Without this, the first copy_tile in fwd_sub_row(row_i=0) reads tiles as zeros.
    mm_init(cb_v_beta_sc, cb_S, cb_v_cor);

    // Initial state pre-loaded by reader into cb_S.
    cb_wait_front(cb_S, state_tiles);

    for (uint32_t c = 0; c < num_chunks; c++) {
        // Wait for all per-chunk inputs (loaded by reader).
        cb_wait_front(cb_L_unit, attn_tiles);
        cb_wait_front(cb_v_beta_sc, out_tiles);
        cb_wait_front(cb_k_bd_sc, in_kv_tiles);
        cb_wait_front(cb_intra_att, attn_tiles);
        cb_wait_front(cb_q_decay, in_kv_tiles);
        cb_wait_front(cb_k_dt, kdt_tiles);
        cb_wait_front(cb_dl_exp, 1);
        // CB14-17 (L_inv0..3) loaded per chunk by reader.

        // ==================================================================
        // PA2. Forward substitution: v_cor = L_unit^{-1} @ v_beta_sc
        //      Keep L_inv tiles for PA3 (pop_inv=false).
        // ==================================================================
        fwd_sub_4rows(
            Ct,
            Vt,
            cb_v_beta_sc,
            cb_v_cor,
            cb_L_inv_0,
            cb_L_inv_1,
            cb_L_inv_2,
            cb_L_inv_3,
            /*pop_inv=*/false);

        // ==================================================================
        // PA3. Forward substitution: k_cum = L_unit^{-1} @ k_bd_sc
        //      Pop L_inv tiles after this pass (pop_inv=true).
        // ==================================================================
        fwd_sub_4rows(
            Ct,
            Kt,
            cb_k_bd_sc,
            cb_k_cum,
            cb_L_inv_0,
            cb_L_inv_1,
            cb_L_inv_2,
            cb_L_inv_3,
            /*pop_inv=*/true);

        // Free per-chunk input CBs no longer needed after preprocessing.
        cb_pop_front(cb_L_unit, attn_tiles);
        cb_pop_front(cb_v_beta_sc, out_tiles);
        cb_pop_front(cb_k_bd_sc, in_kv_tiles);

        // ==================================================================
        // 1. v_prime = k_cum @ S   [Ct,Kt] x [Kt,Vt] -> [Ct,Vt]
        // ==================================================================
        cb_wait_front(cb_k_cum, in_kv_tiles);
        cb_reserve_back(cb_v_prime, out_tiles);
        mm_init(cb_k_cum, cb_S, cb_v_prime);
        for (uint32_t ct = 0; ct < Ct; ct++) {
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    matmul_tiles(cb_k_cum, cb_S, ct * Kt + kt, kt * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_v_prime, ct * Vt + vt);
                tile_regs_release();
            }
        }
        cb_push_back(cb_v_prime, out_tiles);
        cb_pop_front(cb_k_cum, in_kv_tiles);

        // ==================================================================
        // 2. v_new = v_cor - v_prime
        // ==================================================================
        cb_wait_front(cb_v_cor, out_tiles);
        cb_wait_front(cb_v_prime, out_tiles);
        cb_reserve_back(cb_v_new, out_tiles);
        sub_tiles_init(cb_v_cor, cb_v_prime);
        for (uint32_t t = 0; t < out_tiles; t++) {
            tile_regs_acquire();
            sub_tiles(cb_v_cor, cb_v_prime, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_v_new, t);
            tile_regs_release();
        }
        cb_push_back(cb_v_new, out_tiles);
        cb_pop_front(cb_v_cor, out_tiles);
        cb_pop_front(cb_v_prime, out_tiles);

        // ==================================================================
        // 3. o_inter = q_decay @ S
        // ==================================================================
        cb_reserve_back(cb_o_inter, out_tiles);
        mm_init(cb_q_decay, cb_S, cb_o_inter);
        for (uint32_t ct = 0; ct < Ct; ct++) {
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    matmul_tiles(cb_q_decay, cb_S, ct * Kt + kt, kt * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_o_inter, ct * Vt + vt);
                tile_regs_release();
            }
        }
        cb_push_back(cb_o_inter, out_tiles);
        cb_pop_front(cb_q_decay, in_kv_tiles);

        // ==================================================================
        // 4. intra_v = intra_attn @ v_new
        // ==================================================================
        cb_wait_front(cb_v_new, out_tiles);
        cb_reserve_back(cb_intra_v, out_tiles);
        mm_init(cb_intra_att, cb_v_new, cb_intra_v);
        for (uint32_t ct = 0; ct < Ct; ct++) {
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t ct2 = 0; ct2 < Ct; ct2++) {
                    matmul_tiles(cb_intra_att, cb_v_new, ct * Ct + ct2, ct2 * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_intra_v, ct * Vt + vt);
                tile_regs_release();
            }
        }
        cb_push_back(cb_intra_v, out_tiles);
        cb_pop_front(cb_intra_att, attn_tiles);

        // ==================================================================
        // 5. out = o_inter + intra_v
        // ==================================================================
        cb_wait_front(cb_o_inter, out_tiles);
        cb_wait_front(cb_intra_v, out_tiles);
        cb_reserve_back(cb_out, out_tiles);
        add_tiles_init(cb_o_inter, cb_intra_v);
        for (uint32_t t = 0; t < out_tiles; t++) {
            tile_regs_acquire();
            add_tiles(cb_o_inter, cb_intra_v, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out, t);
            tile_regs_release();
        }
        cb_push_back(cb_out, out_tiles);
        cb_pop_front(cb_o_inter, out_tiles);
        cb_pop_front(cb_intra_v, out_tiles);

        // ==================================================================
        // 6. s_upd = k_decay_t @ v_new
        // ==================================================================
        cb_reserve_back(cb_s_upd, state_tiles);
        mm_init(cb_k_dt, cb_v_new, cb_s_upd);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t ct2 = 0; ct2 < Ct; ct2++) {
                    matmul_tiles(cb_k_dt, cb_v_new, kt * Ct + ct2, ct2 * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_s_upd, kt * Vt + vt);
                tile_regs_release();
            }
        }
        cb_push_back(cb_s_upd, state_tiles);
        cb_pop_front(cb_v_new, out_tiles);
        cb_pop_front(cb_k_dt, kdt_tiles);

        // ==================================================================
        // 7a. S_tmp = S * dl_exp
        // ==================================================================
        cb_wait_front(cb_s_upd, state_tiles);
        cb_reserve_back(cb_S_tmp, state_tiles);
        mul_tiles_bcast_scalar_init_short(cb_S, cb_dl_exp);
        for (uint32_t t = 0; t < state_tiles; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_S, cb_dl_exp, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_S_tmp, t);
            tile_regs_release();
        }
        cb_push_back(cb_S_tmp, state_tiles);
        cb_pop_front(cb_S, state_tiles);
        cb_pop_front(cb_dl_exp, 1);

        // ==================================================================
        // 7b. S = S_tmp + s_upd  (last chunk writes to cb_final_state)
        // ==================================================================
        cb_wait_front(cb_S_tmp, state_tiles);
        const bool is_last_chunk = (c == num_chunks - 1);
        uint32_t dst_cb = is_last_chunk ? cb_final_state : cb_S;
        cb_reserve_back(dst_cb, state_tiles);
        add_tiles_init(cb_S_tmp, cb_s_upd);
        for (uint32_t t = 0; t < state_tiles; t++) {
            tile_regs_acquire();
            add_tiles(cb_S_tmp, cb_s_upd, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst_cb, t);
            tile_regs_release();
        }
        cb_push_back(dst_cb, state_tiles);
        cb_pop_front(cb_S_tmp, state_tiles);
        cb_pop_front(cb_s_upd, state_tiles);
    }
}
