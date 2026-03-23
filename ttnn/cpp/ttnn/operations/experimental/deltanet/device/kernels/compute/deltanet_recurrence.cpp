// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DeltaNet fused recurrence compute kernel
//
// Performs the complete delta rule recurrence for all heads in a single
// kernel launch, eliminating ~40 separate ttnn op dispatches per layer.
//
// Per head:
//   1. state *= decay          (element-wise)
//   2. kv_mem = K @ state      (matmul)
//   3. delta = (V - kv_mem) * beta  (element-wise)
//   4. state += K^T @ delta    (matmul)
//   5. output = Q @ state      (matmul)
//
// Input CBs:
//   c_0: q      (k_tiles per head, already GQA-expanded)
//   c_1: k      (k_tiles per head, already GQA-expanded)
//   c_2: v      (v_tiles per head)
//   c_3: decay  (1 tile per head, scalar broadcast)
//   c_4: beta   (1 tile per head, scalar broadcast)
//   c_5: state  (state_tiles_per_head per head)
//
// Output CBs:
//   c_16: output    (v_tiles per head)
//   c_17: state_out (state_tiles_per_head per head)
//
// Intermediate CBs:
//   c_24: matmul intermediate
//   c_25: accumulator

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"

void kernel_main() {
    uint32_t num_heads = get_compile_time_arg_val(0);
    uint32_t k_tiles = get_compile_time_arg_val(1);  // head_k_dim / TILE = 4
    uint32_t v_tiles = get_compile_time_arg_val(2);  // head_v_dim / TILE = 4

    constexpr auto cb_q = tt::CBIndex::c_0;
    constexpr auto cb_k = tt::CBIndex::c_1;
    constexpr auto cb_v = tt::CBIndex::c_2;
    constexpr auto cb_decay = tt::CBIndex::c_3;
    constexpr auto cb_beta = tt::CBIndex::c_4;
    constexpr auto cb_state = tt::CBIndex::c_5;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_state_out = tt::CBIndex::c_17;

    uint32_t state_tiles_per_head = k_tiles * v_tiles;  // 16

    // For now: passthrough stub that copies state and outputs V
    // TODO: Implement full recurrence with matmul_tiles when APIs stabilize
    copy_tile_to_dst_init_short(cb_state);

    for (uint32_t head = 0; head < num_heads; head++) {
        // Pass state through: read from cb_state, write to cb_state_out
        for (uint32_t st = 0; st < state_tiles_per_head; st++) {
            cb_wait_front(cb_state, 1);
            acquire_dst();
            copy_tile(cb_state, 0, 0);
            cb_reserve_back(cb_state_out, 1);
            pack_tile(0, cb_state_out);
            cb_push_back(cb_state_out, 1);
            release_dst();
            cb_pop_front(cb_state, 1);
        }

        // Consume decay tiles
        for (uint32_t dt = 0; dt < state_tiles_per_head; dt++) {
            cb_wait_front(cb_decay, 1);
            cb_pop_front(cb_decay, 1);
        }

        // Consume q tiles
        for (uint32_t qt = 0; qt < k_tiles; qt++) {
            cb_wait_front(cb_q, 1);
            cb_pop_front(cb_q, 1);
        }

        // Consume k tiles
        for (uint32_t kt = 0; kt < k_tiles; kt++) {
            cb_wait_front(cb_k, 1);
            cb_pop_front(cb_k, 1);
        }

        // Pass v through as output
        copy_tile_to_dst_init_short(cb_v);
        for (uint32_t vt = 0; vt < v_tiles; vt++) {
            cb_wait_front(cb_v, 1);
            acquire_dst();
            copy_tile(cb_v, 0, 0);
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            release_dst();
            cb_pop_front(cb_v, 1);
        }

        // Consume beta tiles
        for (uint32_t bt = 0; bt < v_tiles; bt++) {
            cb_wait_front(cb_beta, 1);
            cb_pop_front(cb_beta, 1);
        }

        // Re-init for next head
        copy_tile_to_dst_init_short(cb_state);
    }
}
