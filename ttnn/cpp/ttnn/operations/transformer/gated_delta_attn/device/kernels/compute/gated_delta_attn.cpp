// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include "api/dataflow/circular_buffer.h"

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
        CircularBuffer(out_cb).wait_front(row_i * Xt);
    }

    // Step 1: fwd_rhs = rhs_cb[row_i * Xt .. (row_i+1)*Xt - 1]
    CircularBuffer(cb_nm_P_a).reserve_back(Xt);
    copy_tile_to_dst_init_short(rhs_cb);
    for (uint32_t xt = 0; xt < Xt; xt++) {
        tile_regs_acquire();
        copy_tile(rhs_cb, row_i * Xt + xt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_nm_P_a, xt);
        tile_regs_release();
    }
    CircularBuffer(cb_nm_P_a).push_back(Xt);

    // Step 2: fwd_rhs -= L[row_i, j] @ out[row_j] for each j < row_i
    for (uint32_t j = 0; j < row_i; j++) {
        // corr = L_unit[row_i*Ct + j] @ out_cb[j*Xt .. (j+1)*Xt-1]
        CircularBuffer(cb_nm_P_b).reserve_back(Xt);
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
        CircularBuffer(cb_nm_P_b).push_back(Xt);

        // fwd_rhs = fwd_rhs - corr  (via temp nm_R_a)
        CircularBuffer(cb_nm_P_a).wait_front(Xt);
        CircularBuffer(cb_nm_P_b).wait_front(Xt);
        CircularBuffer(cb_nm_R_a).reserve_back(Xt);
        sub_tiles_init(cb_nm_P_a, cb_nm_P_b);
        for (uint32_t xt = 0; xt < Xt; xt++) {
            tile_regs_acquire();
            sub_tiles(cb_nm_P_a, cb_nm_P_b, xt, xt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_nm_R_a, xt);
            tile_regs_release();
        }
        CircularBuffer(cb_nm_R_a).push_back(Xt);
        CircularBuffer(cb_nm_P_a).pop_front(Xt);
        CircularBuffer(cb_nm_P_b).pop_front(Xt);

        // fwd_rhs (nm_P_a) = nm_R_a
        CircularBuffer(cb_nm_R_a).wait_front(Xt);
        CircularBuffer(cb_nm_P_a).reserve_back(Xt);
        copy_tile_to_dst_init_short(cb_nm_R_a);
        for (uint32_t xt = 0; xt < Xt; xt++) {
            tile_regs_acquire();
            copy_tile(cb_nm_R_a, xt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_nm_P_a, xt);
            tile_regs_release();
        }
        CircularBuffer(cb_nm_P_a).push_back(Xt);
        CircularBuffer(cb_nm_R_a).pop_front(Xt);
    }

    // Step 3: out[row_i * Xt .. (row_i+1)*Xt-1] = L_inv_ii @ fwd_rhs
    CircularBuffer(cb_nm_P_a).wait_front(Xt);
    CircularBuffer(cb_L_inv_row_i).wait_front(1);
    CircularBuffer(out_cb).reserve_back(Xt);
    mm_init(cb_L_inv_row_i, cb_nm_P_a, out_cb);
    for (uint32_t xt = 0; xt < Xt; xt++) {
        tile_regs_acquire();
        matmul_tiles(cb_L_inv_row_i, cb_nm_P_a, 0, xt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb, xt);
        tile_regs_release();
    }
    CircularBuffer(out_cb).push_back(Xt);
    CircularBuffer(cb_nm_P_a).pop_front(Xt);
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
    CircularBuffer(inv0).wait_front(1);
    fwd_sub_row(0, Ct, Xt, rhs_cb, out_cb, inv0);
    if (pop_inv) {
        CircularBuffer(inv0).pop_front(1);
    }

    CircularBuffer(inv1).wait_front(1);
    fwd_sub_row(1, Ct, Xt, rhs_cb, out_cb, inv1);
    if (pop_inv) {
        CircularBuffer(inv1).pop_front(1);
    }

    CircularBuffer(inv2).wait_front(1);
    fwd_sub_row(2, Ct, Xt, rhs_cb, out_cb, inv2);
    if (pop_inv) {
        CircularBuffer(inv2).pop_front(1);
    }

    CircularBuffer(inv3).wait_front(1);
    fwd_sub_row(3, Ct, Xt, rhs_cb, out_cb, inv3);
    if (pop_inv) {
        CircularBuffer(inv3).pop_front(1);
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
    CircularBuffer(cb_S).wait_front(state_tiles);

    for (uint32_t c = 0; c < num_chunks; c++) {
        // Wait for all per-chunk inputs (loaded by reader).
        CircularBuffer(cb_L_unit).wait_front(attn_tiles);
        CircularBuffer(cb_v_beta_sc).wait_front(out_tiles);
        CircularBuffer(cb_k_bd_sc).wait_front(in_kv_tiles);
        CircularBuffer(cb_intra_att).wait_front(attn_tiles);
        CircularBuffer(cb_q_decay).wait_front(in_kv_tiles);
        CircularBuffer(cb_k_dt).wait_front(kdt_tiles);
        CircularBuffer(cb_dl_exp).wait_front(1);
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
        CircularBuffer(cb_L_unit).pop_front(attn_tiles);
        CircularBuffer(cb_v_beta_sc).pop_front(out_tiles);
        CircularBuffer(cb_k_bd_sc).pop_front(in_kv_tiles);

        // ==================================================================
        // 1. v_prime = k_cum @ S   [Ct,Kt] x [Kt,Vt] -> [Ct,Vt]
        // ==================================================================
        CircularBuffer(cb_k_cum).wait_front(in_kv_tiles);
        CircularBuffer(cb_v_prime).reserve_back(out_tiles);
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
        CircularBuffer(cb_v_prime).push_back(out_tiles);
        CircularBuffer(cb_k_cum).pop_front(in_kv_tiles);

        // ==================================================================
        // 2. v_new = v_cor - v_prime
        // ==================================================================
        CircularBuffer(cb_v_cor).wait_front(out_tiles);
        CircularBuffer(cb_v_prime).wait_front(out_tiles);
        CircularBuffer(cb_v_new).reserve_back(out_tiles);
        sub_tiles_init(cb_v_cor, cb_v_prime);
        for (uint32_t t = 0; t < out_tiles; t++) {
            tile_regs_acquire();
            sub_tiles(cb_v_cor, cb_v_prime, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_v_new, t);
            tile_regs_release();
        }
        CircularBuffer(cb_v_new).push_back(out_tiles);
        CircularBuffer(cb_v_cor).pop_front(out_tiles);
        CircularBuffer(cb_v_prime).pop_front(out_tiles);

        // ==================================================================
        // 3. o_inter = q_decay @ S
        // ==================================================================
        CircularBuffer(cb_o_inter).reserve_back(out_tiles);
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
        CircularBuffer(cb_o_inter).push_back(out_tiles);
        CircularBuffer(cb_q_decay).pop_front(in_kv_tiles);

        // ==================================================================
        // 4. intra_v = intra_attn @ v_new
        // ==================================================================
        CircularBuffer(cb_v_new).wait_front(out_tiles);
        CircularBuffer(cb_intra_v).reserve_back(out_tiles);
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
        CircularBuffer(cb_intra_v).push_back(out_tiles);
        CircularBuffer(cb_intra_att).pop_front(attn_tiles);

        // ==================================================================
        // 5. out = o_inter + intra_v
        // ==================================================================
        CircularBuffer(cb_o_inter).wait_front(out_tiles);
        CircularBuffer(cb_intra_v).wait_front(out_tiles);
        CircularBuffer(cb_out).reserve_back(out_tiles);
        add_tiles_init(cb_o_inter, cb_intra_v);
        for (uint32_t t = 0; t < out_tiles; t++) {
            tile_regs_acquire();
            add_tiles(cb_o_inter, cb_intra_v, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out, t);
            tile_regs_release();
        }
        CircularBuffer(cb_out).push_back(out_tiles);
        CircularBuffer(cb_o_inter).pop_front(out_tiles);
        CircularBuffer(cb_intra_v).pop_front(out_tiles);

        // ==================================================================
        // 6. s_upd = k_decay_t @ v_new
        // ==================================================================
        CircularBuffer(cb_s_upd).reserve_back(state_tiles);
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
        CircularBuffer(cb_s_upd).push_back(state_tiles);
        CircularBuffer(cb_v_new).pop_front(out_tiles);
        CircularBuffer(cb_k_dt).pop_front(kdt_tiles);

        // ==================================================================
        // 7a. S_tmp = S * dl_exp
        // ==================================================================
        CircularBuffer(cb_s_upd).wait_front(state_tiles);
        CircularBuffer(cb_S_tmp).reserve_back(state_tiles);
        mul_tiles_bcast_scalar_init_short(cb_S, cb_dl_exp);
        for (uint32_t t = 0; t < state_tiles; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(cb_S, cb_dl_exp, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_S_tmp, t);
            tile_regs_release();
        }
        CircularBuffer(cb_S_tmp).push_back(state_tiles);
        CircularBuffer(cb_S).pop_front(state_tiles);
        CircularBuffer(cb_dl_exp).pop_front(1);

        // ==================================================================
        // 7b. S = S_tmp + s_upd  (last chunk writes to cb_final_state)
        // ==================================================================
        CircularBuffer(cb_S_tmp).wait_front(state_tiles);
        const bool is_last_chunk = (c == num_chunks - 1);
        uint32_t dst_cb = is_last_chunk ? cb_final_state : cb_S;
        CircularBuffer(dst_cb).reserve_back(state_tiles);
        add_tiles_init(cb_S_tmp, cb_s_upd);
        for (uint32_t t = 0; t < state_tiles; t++) {
            tile_regs_acquire();
            add_tiles(cb_S_tmp, cb_s_upd, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst_cb, t);
            tile_regs_release();
        }
        CircularBuffer(dst_cb).push_back(state_tiles);
        CircularBuffer(cb_S_tmp).pop_front(state_tiles);
        CircularBuffer(cb_s_upd).pop_front(state_tiles);
    }
}
