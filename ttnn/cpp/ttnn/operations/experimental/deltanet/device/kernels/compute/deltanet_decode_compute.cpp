// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused DeltaNet decode compute kernel.
//
// Per-head, single-token recurrent step:
//   1. S_mid  = S * g                       (element-wise scale by decay scalar)
//   2. mem    = k @ S_mid  -> [1, Dv]       (row-vector matvec)
//   3. delta  = (v - mem) * beta            (element-wise)
//   4. S_new  = S_mid + outer(k, delta)     (rank-1 update → cb_state_out)
//   5. out    = q @ S_new  -> [1, Dv]       (row-vector matvec)
//
// Step 4 uses matmul(k_T, delta) for the outer product, where k_T is a
// column-vector tile prepared by the reader kernel via L1 element transpose.

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t cb_state_in   = get_compile_time_arg_val(0);
constexpr uint32_t cb_q          = get_compile_time_arg_val(1);
constexpr uint32_t cb_k          = get_compile_time_arg_val(2);
constexpr uint32_t cb_v          = get_compile_time_arg_val(3);
constexpr uint32_t cb_g          = get_compile_time_arg_val(4);
constexpr uint32_t cb_beta       = get_compile_time_arg_val(5);
constexpr uint32_t cb_output     = get_compile_time_arg_val(6);
constexpr uint32_t cb_state_out  = get_compile_time_arg_val(7);
constexpr uint32_t cb_tmp0       = get_compile_time_arg_val(8);
constexpr uint32_t cb_tmp1       = get_compile_time_arg_val(9);
constexpr uint32_t cb_acc        = get_compile_time_arg_val(10);
constexpr uint32_t Dk_tiles      = get_compile_time_arg_val(11);
constexpr uint32_t Dv_tiles      = get_compile_time_arg_val(12);
constexpr uint32_t cb_state_mid  = get_compile_time_arg_val(13);
constexpr uint32_t cb_k_T        = get_compile_time_arg_val(14);

constexpr uint32_t state_tiles   = Dk_tiles * Dv_tiles;

// ---------------------------------------------------------------------------
// Step 1: S_mid = S * g
// ---------------------------------------------------------------------------
FORCE_INLINE void step1_scale_state() {
    mul_tiles_bcast_scalar_init_short(cb_state_in, cb_g);

    cb_wait_front(cb_state_in, state_tiles);
    cb_wait_front(cb_g, 1);
    cb_reserve_back(cb_state_mid, state_tiles);

    for (uint32_t t = 0; t < state_tiles; t++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_state_in, cb_g, t, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_state_mid);
        tile_regs_release();
    }

    cb_push_back(cb_state_mid, state_tiles);
    cb_pop_front(cb_state_in, state_tiles);
}

// ---------------------------------------------------------------------------
// row_matvec: result = vec @ state_cb  ->  [1, Dv]
// ---------------------------------------------------------------------------
FORCE_INLINE void row_matvec(uint32_t cb_vec, uint32_t cb_state, uint32_t cb_result) {
    mm_init_short(cb_vec, cb_state);
    pack_reconfig_data_format(cb_result);

    cb_wait_front(cb_state, state_tiles);
    cb_wait_front(cb_vec, Dk_tiles);
    cb_reserve_back(cb_result, Dv_tiles);

    for (uint32_t j = 0; j < Dv_tiles; j++) {
        tile_regs_acquire();

        for (uint32_t i = 0; i < Dk_tiles; i++) {
            uint32_t state_tile_idx = i * Dv_tiles + j;
            matmul_tiles(cb_vec, cb_state, i, state_tile_idx, 0);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_result);
        tile_regs_release();
    }

    cb_push_back(cb_result, Dv_tiles);
}

// ---------------------------------------------------------------------------
// Step 3: delta = (v - mem) * beta
// ---------------------------------------------------------------------------
FORCE_INLINE void step3_compute_delta() {
    sub_tiles_init(cb_v, cb_tmp0);
    pack_reconfig_data_format(cb_tmp1);

    cb_wait_front(cb_v, Dv_tiles);
    cb_wait_front(cb_tmp0, Dv_tiles);
    cb_reserve_back(cb_tmp1, Dv_tiles);

    for (uint32_t j = 0; j < Dv_tiles; j++) {
        tile_regs_acquire();
        sub_tiles(cb_v, cb_tmp0, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_tmp1);
        tile_regs_release();
    }

    cb_push_back(cb_tmp1, Dv_tiles);
    cb_pop_front(cb_tmp0, Dv_tiles);

    mul_tiles_bcast_scalar_init_short(cb_tmp1, cb_beta);
    pack_reconfig_data_format(cb_acc);

    cb_wait_front(cb_tmp1, Dv_tiles);
    cb_wait_front(cb_beta, 1);
    cb_reserve_back(cb_acc, Dv_tiles);

    for (uint32_t j = 0; j < Dv_tiles; j++) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_tmp1, cb_beta, j, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_acc);
        tile_regs_release();
    }

    cb_push_back(cb_acc, Dv_tiles);
    cb_pop_front(cb_tmp1, Dv_tiles);
}

// ---------------------------------------------------------------------------
// Step 4: S_new = S_mid + outer(k, delta)
//
// outer(k, delta) is computed via matmul(k_T, delta):
//   k_T is a column vector tile (k_T[r,0] = k[r]), delta is a row vector.
//   matmul(k_T, delta)[r,c] = k_T[r,0] * delta[0,c] = k[r] * delta[c]
//
// For each state tile (i,j):
//   outer_tile = matmul(k_T[i], delta[j])   → pack to cb_tmp1
//   S_new[i,j] = S_mid[i,j] + outer_tile    → pack to cb_state_out
// ---------------------------------------------------------------------------
FORCE_INLINE void step4_rank1_update() {
    cb_wait_front(cb_state_mid, state_tiles);
    cb_wait_front(cb_k_T, Dk_tiles);
    cb_wait_front(cb_acc, Dv_tiles);
    cb_reserve_back(cb_state_out, state_tiles);

    for (uint32_t i = 0; i < Dk_tiles; i++) {
        for (uint32_t j = 0; j < Dv_tiles; j++) {
            uint32_t state_tile_idx = i * Dv_tiles + j;

            // Compute outer product tile via matmul
            cb_reserve_back(cb_tmp1, 1);
            mm_init_short(cb_k_T, cb_acc);
            pack_reconfig_data_format(cb_tmp1);
            tile_regs_acquire();
            matmul_tiles(cb_k_T, cb_acc, i, j, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp1);
            tile_regs_release();
            cb_push_back(cb_tmp1, 1);

            // S_new = S_mid + outer
            cb_wait_front(cb_tmp1, 1);
            add_tiles_init(cb_state_mid, cb_tmp1);
            pack_reconfig_data_format(cb_state_out);
            tile_regs_acquire();
            add_tiles(cb_state_mid, cb_tmp1, state_tile_idx, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_out);
            tile_regs_release();
            cb_pop_front(cb_tmp1, 1);
        }
    }

    cb_push_back(cb_state_out, state_tiles);
    cb_pop_front(cb_state_mid, state_tiles);
    cb_pop_front(cb_k_T, Dk_tiles);
    cb_pop_front(cb_acc, Dv_tiles);
}

void kernel_main() {
    binary_op_init_common(cb_state_in, cb_g, cb_state_mid);

    // Step 1: S_mid = S * g
    step1_scale_state();

    // Step 2: mem = k @ S_mid
    row_matvec(cb_k, cb_state_mid, cb_tmp0);

    // Step 3: delta = (v - mem) * beta
    step3_compute_delta();

    // Step 4: S_new = S_mid + outer(k, delta)
    step4_rank1_update();

    // Step 5: output = q @ S_new
    row_matvec(cb_q, cb_state_out, cb_output);

    // Pop remaining input CBs
    cb_pop_front(cb_q, Dk_tiles);
    cb_pop_front(cb_k, Dk_tiles);
    cb_pop_front(cb_v, Dv_tiles);
    cb_pop_front(cb_g, 1);
    cb_pop_front(cb_beta, 1);
}
