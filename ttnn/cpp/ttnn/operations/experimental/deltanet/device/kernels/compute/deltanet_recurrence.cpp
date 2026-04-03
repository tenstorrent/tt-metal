// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// DeltaNet compute — FULL recurrence with delta rule.
//
// Per head:
//   Step 1: scratch = state * decay
//   Step 2: kv_mem = K @ scratch              (v_tiles output)
//   Step 3: delta = (V - kv_mem) * beta       (v_tiles)
//   Step 4: outer = K^T x delta               (k*v state_tiles — outer product)
//           scratch += outer
//   Step 5: output = Q @ scratch              (v_tiles output)
//   Step 6: scratch → state_out

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"

void kernel_main() {
    uint32_t num_heads = get_compile_time_arg_val(0);
    uint32_t k_tiles = get_compile_time_arg_val(1);
    uint32_t v_tiles = get_compile_time_arg_val(2);

    constexpr auto cb_q = tt::CBIndex::c_0;
    constexpr auto cb_k = tt::CBIndex::c_1;
    constexpr auto cb_v = tt::CBIndex::c_2;
    constexpr auto cb_decay = tt::CBIndex::c_3;
    constexpr auto cb_beta = tt::CBIndex::c_4;
    constexpr auto cb_state = tt::CBIndex::c_5;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_state_out = tt::CBIndex::c_17;
    constexpr auto cb_scratch = tt::CBIndex::c_24;   // working state (k*v tiles)
    constexpr auto cb_scratch2 = tt::CBIndex::c_25;  // Q+K cache (2*k tiles)
    constexpr auto cb_kv_mem = tt::CBIndex::c_26;    // step 2 result (v tiles)
    constexpr auto cb_delta = tt::CBIndex::c_27;     // step 3 result (v tiles)
    constexpr auto cb_outer = tt::CBIndex::c_28;     // step 4: outer product + updated state (k*v tiles)

    uint32_t state_tiles = k_tiles * v_tiles;

    mm_init(cb_scratch2, cb_scratch, cb_out);

    for (uint32_t head = 0; head < num_heads; head++) {
        // ====== Step 1: scratch = state * decay ======
        mul_tiles_init(cb_state, cb_decay);
        for (uint32_t st = 0; st < state_tiles; st++) {
            cb_wait_front(cb_state, 1);
            cb_wait_front(cb_decay, 1);
            acquire_dst();
            mul_tiles(cb_state, cb_decay, 0, 0, 0);
            cb_reserve_back(cb_scratch, 1);
            pack_tile(0, cb_scratch);
            cb_push_back(cb_scratch, 1);
            release_dst();
            cb_pop_front(cb_state, 1);
            cb_pop_front(cb_decay, 1);
        }
        cb_wait_front(cb_scratch, state_tiles);

        // ====== Cache Q[0..k-1] then K[k..2k-1] into scratch2 ======
        copy_tile_to_dst_init_short(cb_q);
        for (uint32_t i = 0; i < k_tiles; i++) {
            cb_wait_front(cb_q, 1);
            acquire_dst();
            copy_tile(cb_q, 0, 0);
            cb_reserve_back(cb_scratch2, 1);
            pack_tile(0, cb_scratch2);
            cb_push_back(cb_scratch2, 1);
            release_dst();
            cb_pop_front(cb_q, 1);
        }
        for (uint32_t i = 0; i < k_tiles; i++) {
            cb_wait_front(cb_k, 1);
            acquire_dst();
            copy_tile(cb_k, 0, 0);
            cb_reserve_back(cb_scratch2, 1);
            pack_tile(0, cb_scratch2);
            cb_push_back(cb_scratch2, 1);
            release_dst();
            cb_pop_front(cb_k, 1);
        }
        cb_wait_front(cb_scratch2, 2 * k_tiles);

        // ====== Step 2: kv_mem[j] = sum_k K[k] * scratch[k,j] ======
        mm_init_short(cb_scratch2, cb_scratch);
        pack_reconfig_data_format(cb_kv_mem);
        reconfig_data_format(cb_scratch2, cb_scratch);
        for (uint32_t j = 0; j < v_tiles; j++) {
            acquire_dst();
            for (uint32_t k = 0; k < k_tiles; k++) {
                matmul_block(cb_scratch2, cb_scratch, k_tiles + k, k * v_tiles + j, 0, false, 1, 1, 1);
            }
            cb_reserve_back(cb_kv_mem, 1);
            pack_tile(0, cb_kv_mem);
            cb_push_back(cb_kv_mem, 1);
            release_dst();
        }
        cb_wait_front(cb_kv_mem, v_tiles);

        // ====== Step 3a: delta = V - kv_mem ======
        sub_tiles_init(cb_v, cb_kv_mem);
        pack_reconfig_data_format(cb_delta);
        for (uint32_t j = 0; j < v_tiles; j++) {
            cb_wait_front(cb_v, 1);
            acquire_dst();
            sub_tiles(cb_v, cb_kv_mem, 0, j, 0);
            cb_reserve_back(cb_delta, 1);
            pack_tile(0, cb_delta);
            cb_push_back(cb_delta, 1);
            release_dst();
            cb_pop_front(cb_v, 1);
        }
        cb_pop_front(cb_kv_mem, v_tiles);
        cb_wait_front(cb_delta, v_tiles);

        // ====== Step 3b: delta *= beta → store in kv_mem, then move to delta ======
        mul_tiles_init(cb_delta, cb_beta);
        pack_reconfig_data_format(cb_kv_mem);
        for (uint32_t j = 0; j < v_tiles; j++) {
            cb_wait_front(cb_beta, 1);
            acquire_dst();
            mul_tiles(cb_delta, cb_beta, j, 0, 0);
            cb_reserve_back(cb_kv_mem, 1);
            pack_tile(0, cb_kv_mem);
            cb_push_back(cb_kv_mem, 1);
            release_dst();
            cb_pop_front(cb_beta, 1);
        }
        cb_pop_front(cb_delta, v_tiles);
        cb_wait_front(cb_kv_mem, v_tiles);
        // Move kv_mem → delta
        copy_tile_to_dst_init_short(cb_kv_mem);
        pack_reconfig_data_format(cb_delta);
        for (uint32_t j = 0; j < v_tiles; j++) {
            acquire_dst();
            copy_tile(cb_kv_mem, j, 0);
            cb_reserve_back(cb_delta, 1);
            pack_tile(0, cb_delta);
            cb_push_back(cb_delta, 1);
            release_dst();
        }
        cb_pop_front(cb_kv_mem, v_tiles);
        cb_wait_front(cb_delta, v_tiles);

        // ====== Step 4: outer product K^T x delta, then scratch += outer ======
        // 4a: Compute outer[k,j] = K[k] * delta[j] for all k,j → cb_outer
        mm_init_short(cb_scratch2, cb_delta);
        pack_reconfig_data_format(cb_outer);
        reconfig_data_format(cb_scratch2, cb_delta);
        for (uint32_t k = 0; k < k_tiles; k++) {
            for (uint32_t j = 0; j < v_tiles; j++) {
                acquire_dst();
                matmul_block(cb_scratch2, cb_delta, k_tiles + k, j, 0, false, 1, 1, 1);
                cb_reserve_back(cb_outer, 1);
                pack_tile(0, cb_outer);
                cb_push_back(cb_outer, 1);
                release_dst();
            }
        }
        cb_pop_front(cb_delta, v_tiles);
        cb_wait_front(cb_outer, state_tiles);

        // 4b: scratch += outer (tile-by-tile add, result into kv_mem as temp, one at a time)
        //     Then pop both and copy result back to scratch.
        //     Strategy: pop scratch + outer together, add tile-by-tile, push to scratch.
        //     But we can't push back to scratch while popping from it.
        //     Instead: add into cb_delta (reuse, has v_tiles capacity — but we need state_tiles).
        //
        //     Better: just pop scratch, pop outer, add → push to state_out temporarily,
        //     then copy state_out → scratch. But state_out is only 2 tiles.
        //
        //     Simplest correct approach: process one tile at a time.
        //     Pop 1 from scratch, pop 1 from outer, add → push 1 to kv_mem.
        //     After all state_tiles, pop all kv_mem, copy to scratch.
        //     BUT kv_mem only has v_tiles capacity.
        //
        //     OK: I need to do this in chunks of v_tiles (kv_mem capacity).
        //     For each k-row: add v_tiles pairs, store in kv_mem, then copy to scratch.
        for (uint32_t k = 0; k < k_tiles; k++) {
            // Add v_tiles pairs: scratch[k*v..k*v+v-1] + outer[k*v..k*v+v-1]
            add_tiles_init(cb_scratch, cb_outer);
            pack_reconfig_data_format(cb_kv_mem);
            for (uint32_t j = 0; j < v_tiles; j++) {
                acquire_dst();
                add_tiles(cb_scratch, cb_outer, 0, 0, 0);
                cb_reserve_back(cb_kv_mem, 1);
                pack_tile(0, cb_kv_mem);
                cb_push_back(cb_kv_mem, 1);
                release_dst();
                cb_pop_front(cb_scratch, 1);
                cb_pop_front(cb_outer, 1);
            }
            // Copy kv_mem → scratch for this row
            cb_wait_front(cb_kv_mem, v_tiles);
            copy_tile_to_dst_init_short(cb_kv_mem);
            pack_reconfig_data_format(cb_scratch);
            for (uint32_t j = 0; j < v_tiles; j++) {
                acquire_dst();
                copy_tile(cb_kv_mem, 0, 0);
                cb_reserve_back(cb_scratch, 1);
                pack_tile(0, cb_scratch);
                cb_push_back(cb_scratch, 1);
                release_dst();
                cb_pop_front(cb_kv_mem, 1);
            }
        }
        cb_wait_front(cb_scratch, state_tiles);

        // ====== Step 5: output[j] = sum_k Q[k] * scratch[k,j] ======
        mm_init_short(cb_scratch2, cb_scratch);
        pack_reconfig_data_format(cb_out);
        reconfig_data_format(cb_scratch2, cb_scratch);
        for (uint32_t j = 0; j < v_tiles; j++) {
            acquire_dst();
            for (uint32_t k = 0; k < k_tiles; k++) {
                matmul_block(cb_scratch2, cb_scratch, k, k * v_tiles + j, 0, false, 1, 1, 1);
            }
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            release_dst();
        }
        cb_pop_front(cb_scratch2, 2 * k_tiles);

        // ====== Step 6: scratch → state_out ======
        copy_tile_to_dst_init_short(cb_scratch);
        pack_reconfig_data_format(cb_state_out);
        for (uint32_t st = 0; st < state_tiles; st++) {
            acquire_dst();
            copy_tile(cb_scratch, 0, 0);
            cb_reserve_back(cb_state_out, 1);
            pack_tile(0, cb_state_out);
            cb_push_back(cb_state_out, 1);
            release_dst();
            cb_pop_front(cb_scratch, 1);
        }
    }
}
