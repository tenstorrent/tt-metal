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
// Runtime args: num_chunks

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/bcast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// DFB handles (bound by the program factory)
constexpr auto dfb_L_unit = dfb::L_unit;
constexpr auto dfb_v_beta_sc = dfb::v_beta_sc;
constexpr auto dfb_k_bd_sc = dfb::k_bd_sc;
constexpr auto dfb_intra_att = dfb::intra_att;
constexpr auto dfb_q_decay = dfb::q_decay;
constexpr auto dfb_k_dt = dfb::k_dt;
constexpr auto dfb_dl_exp = dfb::dl_exp;
constexpr auto dfb_S0 = dfb::S0;  // initial state (seeded by reader, read on chunk 0)
constexpr auto dfb_S = dfb::S;    // persistent state

constexpr auto dfb_nm_P_a = dfb::nm_P_a;  // fwd_rhs scratch
constexpr auto dfb_nm_P_b = dfb::nm_P_b;  // corr_mm scratch
constexpr auto dfb_nm_R_a = dfb::nm_R_a;  // temp_rhs scratch
// c_12, c_13: unused (were Neumann-only)

// L_inv diagonal block inverses — pre-loaded per chunk by reader
constexpr auto dfb_L_inv_0 = dfb::L_inv_0;
constexpr auto dfb_L_inv_1 = dfb::L_inv_1;
constexpr auto dfb_L_inv_2 = dfb::L_inv_2;
constexpr auto dfb_L_inv_3 = dfb::L_inv_3;

constexpr auto dfb_v_cor = dfb::v_cor;
constexpr auto dfb_k_cum = dfb::k_cum;

constexpr auto dfb_v_prime = dfb::v_prime;
constexpr auto dfb_v_new = dfb::v_new;
constexpr auto dfb_o_inter = dfb::o_inter;
constexpr auto dfb_intra_v = dfb::intra_v;
constexpr auto dfb_out = dfb::out;
constexpr auto dfb_s_upd = dfb::s_upd;
constexpr auto dfb_S_tmp = dfb::S_tmp;
constexpr auto dfb_final_state = dfb::final_state;

// Reconfigure sources + packer and re-init the matmul MOP for a new in0 @ in1 -> out matmul.
// Matmul maps in0 -> SrcB and in1 -> SrcA, so the source reconfig uses SrcOrder::Reverse.
static inline void matmul_reconfig_and_init(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t out_dfb) {
    reconfig_data_format<SrcOrder::Reverse>(in0_dfb, in1_dfb);
    matmul_init(in0_dfb, in1_dfb);
    pack_reconfig_data_format(out_dfb);
}

// ---------------------------------------------------------------------------
// One row of blocked forward substitution.
//
// Computes row `row_i` of:  out = L_unit^{-1} @ rhs
//
// At entry: out_dfb already has rows 0..row_i-1 pushed (Xt tiles each).
// This function pushes Xt more tiles (row_i) to out_dfb.
//
// dfb_L_unit:     holds Ct*Ct tiles (full L_unit for this chunk), available via wait.
// rhs_dfb:        holds Ct*Xt tiles.
// dfb_L_inv_row_i: 1 tile = L^{-1}[row_i,row_i], waited on externally (not popped here).
// Scratch: dfb_nm_P_a (fwd_rhs Xt tiles), dfb_nm_P_b (corr Xt tiles), dfb_nm_R_a (temp Xt tiles).
// ---------------------------------------------------------------------------
__attribute__((noinline)) static void fwd_sub_row(
    uint32_t row_i, uint32_t Ct, uint32_t Xt, uint32_t rhs_dfb, uint32_t out_dfb, uint32_t dfb_L_inv_row_i) {
    // Previous rows must be available for the subtraction loop.
    if (row_i > 0) {
        DataflowBuffer(out_dfb).wait_front(row_i * Xt);
    }

    // Step 1: fwd_rhs = rhs_dfb[row_i * Xt .. (row_i+1)*Xt - 1]
    DataflowBuffer(dfb_nm_P_a).reserve_back(Xt);
    copy_init(rhs_dfb);
    for (uint32_t xt = 0; xt < Xt; xt++) {
        tile_regs_acquire();
        copy_tile(rhs_dfb, row_i * Xt + xt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb_nm_P_a, xt);
        tile_regs_release();
    }
    DataflowBuffer(dfb_nm_P_a).push_back(Xt);

    // Step 2: fwd_rhs -= L[row_i, j] @ out[row_j] for each j < row_i
    for (uint32_t j = 0; j < row_i; j++) {
        // corr = L_unit[row_i*Ct + j] @ out_dfb[j*Xt .. (j+1)*Xt-1]
        DataflowBuffer(dfb_nm_P_b).reserve_back(Xt);
        matmul_reconfig_and_init(dfb_L_unit, out_dfb, dfb_nm_P_b);
        uint32_t L_tile = row_i * Ct + j;
        for (uint32_t xt = 0; xt < Xt; xt++) {
            tile_regs_acquire();
            matmul_tiles(dfb_L_unit, out_dfb, L_tile, j * Xt + xt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb_nm_P_b, xt);
            tile_regs_release();
        }
        DataflowBuffer(dfb_nm_P_b).push_back(Xt);

        // fwd_rhs = fwd_rhs - corr  (via temp nm_R_a)
        DataflowBuffer(dfb_nm_P_a).wait_front(Xt);
        DataflowBuffer(dfb_nm_P_b).wait_front(Xt);
        DataflowBuffer(dfb_nm_R_a).reserve_back(Xt);
        sub_init(dfb_nm_P_a, dfb_nm_P_b);
        for (uint32_t xt = 0; xt < Xt; xt++) {
            tile_regs_acquire();
            sub_tiles(dfb_nm_P_a, dfb_nm_P_b, xt, xt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb_nm_R_a, xt);
            tile_regs_release();
        }
        DataflowBuffer(dfb_nm_R_a).push_back(Xt);
        DataflowBuffer(dfb_nm_P_a).pop_front(Xt);
        DataflowBuffer(dfb_nm_P_b).pop_front(Xt);

        // fwd_rhs (nm_P_a) = nm_R_a
        DataflowBuffer(dfb_nm_R_a).wait_front(Xt);
        DataflowBuffer(dfb_nm_P_a).reserve_back(Xt);
        copy_init(dfb_nm_R_a);
        for (uint32_t xt = 0; xt < Xt; xt++) {
            tile_regs_acquire();
            copy_tile(dfb_nm_R_a, xt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb_nm_P_a, xt);
            tile_regs_release();
        }
        DataflowBuffer(dfb_nm_P_a).push_back(Xt);
        DataflowBuffer(dfb_nm_R_a).pop_front(Xt);
    }

    // Step 3: out[row_i * Xt .. (row_i+1)*Xt-1] = L_inv_ii @ fwd_rhs
    DataflowBuffer(dfb_nm_P_a).wait_front(Xt);
    DataflowBuffer(dfb_L_inv_row_i).wait_front(1);
    DataflowBuffer(out_dfb).reserve_back(Xt);
    matmul_reconfig_and_init(dfb_L_inv_row_i, dfb_nm_P_a, out_dfb);
    for (uint32_t xt = 0; xt < Xt; xt++) {
        tile_regs_acquire();
        matmul_tiles(dfb_L_inv_row_i, dfb_nm_P_a, 0, xt, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_dfb, xt);
        tile_regs_release();
    }
    DataflowBuffer(out_dfb).push_back(Xt);
    DataflowBuffer(dfb_nm_P_a).pop_front(Xt);
    // Caller pops dfb_L_inv_row_i (if pop_inv=true).
}

// Forward-sub all 4 block rows.
// pop_inv: if true, pop dfb_L_inv_0..3 after use (second pass); if false, keep for next pass.
__attribute__((noinline)) static void fwd_sub_4rows(
    uint32_t Ct,
    uint32_t Xt,
    uint32_t rhs_dfb,
    uint32_t out_dfb,
    uint32_t inv0,
    uint32_t inv1,
    uint32_t inv2,
    uint32_t inv3,
    bool pop_inv) {
    DataflowBuffer(inv0).wait_front(1);
    fwd_sub_row(0, Ct, Xt, rhs_dfb, out_dfb, inv0);
    if (pop_inv) {
        DataflowBuffer(inv0).pop_front(1);
    }

    DataflowBuffer(inv1).wait_front(1);
    fwd_sub_row(1, Ct, Xt, rhs_dfb, out_dfb, inv1);
    if (pop_inv) {
        DataflowBuffer(inv1).pop_front(1);
    }

    DataflowBuffer(inv2).wait_front(1);
    fwd_sub_row(2, Ct, Xt, rhs_dfb, out_dfb, inv2);
    if (pop_inv) {
        DataflowBuffer(inv2).pop_front(1);
    }

    DataflowBuffer(inv3).wait_front(1);
    fwd_sub_row(3, Ct, Xt, rhs_dfb, out_dfb, inv3);
    if (pop_inv) {
        DataflowBuffer(inv3).pop_front(1);
    }
}

void kernel_main() {
    constexpr uint32_t Ct = get_arg(args::Ct);
    constexpr uint32_t Kt = get_arg(args::Kt);
    constexpr uint32_t Vt = get_arg(args::Vt);

    const uint32_t num_chunks = get_arg(args::num_chunks);

    constexpr uint32_t state_tiles = Kt * Vt;
    constexpr uint32_t out_tiles = Ct * Vt;
    constexpr uint32_t in_kv_tiles = Ct * Kt;
    constexpr uint32_t attn_tiles = Ct * Ct;
    constexpr uint32_t kdt_tiles = Kt * Ct;

    // Pre-configure hardware (UNPACK/MATH/PACK) format registers for float32. This one-time
    // HW startup must run before any copy_init call; without it, the first
    // copy_tile in fwd_sub_row(row_i=0) reads tiles as zeros. Matmul maps in0 -> SrcB and
    // in1 -> SrcA, hence SrcOrder::Reverse (per-matmul init is done at each matmul below).
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb_v_beta_sc, dfb_S, dfb_v_cor);

    // Initial state pre-loaded by reader into dfb_S0; later chunks read the state compute recycled into dfb_S.
    DataflowBuffer(dfb_S0).wait_front(state_tiles);

    for (uint32_t c = 0; c < num_chunks; c++) {
        const uint32_t dfb_state = (c == 0) ? dfb_S0 : dfb_S;
        // Wait for all per-chunk inputs (loaded by reader).
        DataflowBuffer(dfb_L_unit).wait_front(attn_tiles);
        DataflowBuffer(dfb_v_beta_sc).wait_front(out_tiles);
        DataflowBuffer(dfb_k_bd_sc).wait_front(in_kv_tiles);
        DataflowBuffer(dfb_intra_att).wait_front(attn_tiles);
        DataflowBuffer(dfb_q_decay).wait_front(in_kv_tiles);
        DataflowBuffer(dfb_k_dt).wait_front(kdt_tiles);
        DataflowBuffer(dfb_dl_exp).wait_front(1);
        // dfb_L_inv_0..3 (L_inv0..3) loaded per chunk by reader.

        // ==================================================================
        // PA2. Forward substitution: v_cor = L_unit^{-1} @ v_beta_sc
        //      Keep L_inv tiles for PA3 (pop_inv=false).
        // ==================================================================
        fwd_sub_4rows(
            Ct,
            Vt,
            dfb_v_beta_sc,
            dfb_v_cor,
            dfb_L_inv_0,
            dfb_L_inv_1,
            dfb_L_inv_2,
            dfb_L_inv_3,
            /*pop_inv=*/false);

        // ==================================================================
        // PA3. Forward substitution: k_cum = L_unit^{-1} @ k_bd_sc
        //      Pop L_inv tiles after this pass (pop_inv=true).
        // ==================================================================
        fwd_sub_4rows(
            Ct,
            Kt,
            dfb_k_bd_sc,
            dfb_k_cum,
            dfb_L_inv_0,
            dfb_L_inv_1,
            dfb_L_inv_2,
            dfb_L_inv_3,
            /*pop_inv=*/true);

        // Free per-chunk input DFBs no longer needed after preprocessing.
        DataflowBuffer(dfb_L_unit).pop_front(attn_tiles);
        DataflowBuffer(dfb_v_beta_sc).pop_front(out_tiles);
        DataflowBuffer(dfb_k_bd_sc).pop_front(in_kv_tiles);

        // ==================================================================
        // 1. v_prime = k_cum @ S   [Ct,Kt] x [Kt,Vt] -> [Ct,Vt]
        // ==================================================================
        DataflowBuffer(dfb_k_cum).wait_front(in_kv_tiles);
        DataflowBuffer(dfb_v_prime).reserve_back(out_tiles);
        matmul_reconfig_and_init(dfb_k_cum, dfb_state, dfb_v_prime);
        for (uint32_t ct = 0; ct < Ct; ct++) {
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    matmul_tiles(dfb_k_cum, dfb_state, ct * Kt + kt, kt * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb_v_prime, ct * Vt + vt);
                tile_regs_release();
            }
        }
        DataflowBuffer(dfb_v_prime).push_back(out_tiles);
        DataflowBuffer(dfb_k_cum).pop_front(in_kv_tiles);

        // ==================================================================
        // 2. v_new = v_cor - v_prime
        // ==================================================================
        DataflowBuffer(dfb_v_cor).wait_front(out_tiles);
        DataflowBuffer(dfb_v_prime).wait_front(out_tiles);
        DataflowBuffer(dfb_v_new).reserve_back(out_tiles);
        sub_init(dfb_v_cor, dfb_v_prime);
        for (uint32_t t = 0; t < out_tiles; t++) {
            tile_regs_acquire();
            sub_tiles(dfb_v_cor, dfb_v_prime, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb_v_new, t);
            tile_regs_release();
        }
        DataflowBuffer(dfb_v_new).push_back(out_tiles);
        DataflowBuffer(dfb_v_cor).pop_front(out_tiles);
        DataflowBuffer(dfb_v_prime).pop_front(out_tiles);

        // ==================================================================
        // 3. o_inter = q_decay @ S
        // ==================================================================
        DataflowBuffer(dfb_o_inter).reserve_back(out_tiles);
        matmul_reconfig_and_init(dfb_q_decay, dfb_state, dfb_o_inter);
        for (uint32_t ct = 0; ct < Ct; ct++) {
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    matmul_tiles(dfb_q_decay, dfb_state, ct * Kt + kt, kt * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb_o_inter, ct * Vt + vt);
                tile_regs_release();
            }
        }
        DataflowBuffer(dfb_o_inter).push_back(out_tiles);
        DataflowBuffer(dfb_q_decay).pop_front(in_kv_tiles);

        // ==================================================================
        // 4. intra_v = intra_attn @ v_new
        // ==================================================================
        DataflowBuffer(dfb_v_new).wait_front(out_tiles);
        DataflowBuffer(dfb_intra_v).reserve_back(out_tiles);
        matmul_reconfig_and_init(dfb_intra_att, dfb_v_new, dfb_intra_v);
        for (uint32_t ct = 0; ct < Ct; ct++) {
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t ct2 = 0; ct2 < Ct; ct2++) {
                    matmul_tiles(dfb_intra_att, dfb_v_new, ct * Ct + ct2, ct2 * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb_intra_v, ct * Vt + vt);
                tile_regs_release();
            }
        }
        DataflowBuffer(dfb_intra_v).push_back(out_tiles);
        DataflowBuffer(dfb_intra_att).pop_front(attn_tiles);

        // ==================================================================
        // 5. out = o_inter + intra_v
        // ==================================================================
        DataflowBuffer(dfb_o_inter).wait_front(out_tiles);
        DataflowBuffer(dfb_intra_v).wait_front(out_tiles);
        DataflowBuffer(dfb_out).reserve_back(out_tiles);
        add_init(dfb_o_inter, dfb_intra_v);
        for (uint32_t t = 0; t < out_tiles; t++) {
            tile_regs_acquire();
            add_tiles(dfb_o_inter, dfb_intra_v, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb_out, t);
            tile_regs_release();
        }
        DataflowBuffer(dfb_out).push_back(out_tiles);
        DataflowBuffer(dfb_o_inter).pop_front(out_tiles);
        DataflowBuffer(dfb_intra_v).pop_front(out_tiles);

        // ==================================================================
        // 6. s_upd = k_decay_t @ v_new
        // ==================================================================
        DataflowBuffer(dfb_s_upd).reserve_back(state_tiles);
        matmul_reconfig_and_init(dfb_k_dt, dfb_v_new, dfb_s_upd);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t ct2 = 0; ct2 < Ct; ct2++) {
                    matmul_tiles(dfb_k_dt, dfb_v_new, kt * Ct + ct2, ct2 * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb_s_upd, kt * Vt + vt);
                tile_regs_release();
            }
        }
        DataflowBuffer(dfb_s_upd).push_back(state_tiles);
        DataflowBuffer(dfb_v_new).pop_front(out_tiles);
        DataflowBuffer(dfb_k_dt).pop_front(kdt_tiles);

        // ==================================================================
        // 7a. S_tmp = S * dl_exp
        // ==================================================================
        DataflowBuffer(dfb_s_upd).wait_front(state_tiles);
        DataflowBuffer(dfb_S_tmp).reserve_back(state_tiles);
        mul_bcast_scalar_init(dfb_state, dfb_dl_exp);
        for (uint32_t t = 0; t < state_tiles; t++) {
            tile_regs_acquire();
            mul_tiles_bcast_scalar(dfb_state, dfb_dl_exp, t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb_S_tmp, t);
            tile_regs_release();
        }
        DataflowBuffer(dfb_S_tmp).push_back(state_tiles);
        DataflowBuffer(dfb_state).pop_front(state_tiles);
        DataflowBuffer(dfb_dl_exp).pop_front(1);

        // ==================================================================
        // 7b. S = S_tmp + s_upd  (last chunk writes to dfb_final_state)
        // ==================================================================
        DataflowBuffer(dfb_S_tmp).wait_front(state_tiles);
        const bool is_last_chunk = (c == num_chunks - 1);
        uint32_t dst_dfb = is_last_chunk ? dfb_final_state : dfb_S;
        DataflowBuffer(dst_dfb).reserve_back(state_tiles);
        add_init(dfb_S_tmp, dfb_s_upd);
        for (uint32_t t = 0; t < state_tiles; t++) {
            tile_regs_acquire();
            add_tiles(dfb_S_tmp, dfb_s_upd, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dst_dfb, t);
            tile_regs_release();
        }
        DataflowBuffer(dst_dfb).push_back(state_tiles);
        DataflowBuffer(dfb_S_tmp).pop_front(state_tiles);
        DataflowBuffer(dfb_s_upd).pop_front(state_tiles);
    }
}
