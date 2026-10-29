// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t cb_seed = get_compile_time_arg_val(0);
constexpr uint32_t cb_tail = get_compile_time_arg_val(1);
constexpr uint32_t cb_m = get_compile_time_arg_val(2);
constexpr uint32_t cb_base = get_compile_time_arg_val(3);
constexpr uint32_t cb_diff = get_compile_time_arg_val(4);
constexpr uint32_t cb_out = get_compile_time_arg_val(5);
constexpr uint32_t cb_a = get_compile_time_arg_val(6);
constexpr uint32_t cb_b = get_compile_time_arg_val(7);
constexpr uint32_t cb_final = get_compile_time_arg_val(8);
constexpr uint32_t cb_ctl = get_compile_time_arg_val(9);
constexpr uint32_t kt = get_compile_time_arg_val(10);
constexpr uint32_t state_tiles = kt * kt;

// output_c = F0_c + (FI_c - F0_c) @ seed per chunk, and this core's tile rows of final = A @ seed + B, accumulated in
// the FP32 DST on top of the copied F0 (or B).
void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_m, cb_seed, cb_out);
    cb_wait_front(cb_ctl, 1);
    const uint32_t chunks = read_tile_value(cb_ctl, 0, 0);
    const uint32_t group_chunks = read_tile_value(cb_ctl, 0, 1);
    const uint32_t final = read_tile_value(cb_ctl, 0, 2);
    const uint32_t final_tail = read_tile_value(cb_ctl, 0, 3);
    const uint32_t need_group = read_tile_value(cb_ctl, 0, 4);
    const uint32_t need_tail = read_tile_value(cb_ctl, 0, 5);
    const uint32_t first_row = read_tile_value(cb_ctl, 0, 6);
    const uint32_t row_stride = read_tile_value(cb_ctl, 0, 7);
    cb_pop_front(cb_ctl, 1);
    // The differences first, for all of this core's chunks (they need no seed, so they run while the seeds arrive);
    // then per chunk F0's row is copied into DST and the K-sum accumulates on top of it.
    if (chunks != 0) {
        cb_wait_front(cb_m, chunks * kt);
        cb_wait_front(cb_base, chunks * kt);
        cb_reserve_back(cb_diff, chunks * kt);
        pack_reconfig_data_format(cb_diff);
        reconfig_data_format(cb_m, cb_base);
        sub_init(cb_m, cb_base);
        for (uint32_t t = 0; t < chunks * kt; t += kt) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < kt; ++i) {
                sub_tiles(cb_m, cb_base, t + i, t + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < kt; ++i) {
                pack_tile(i, cb_diff);
            }
            tile_regs_release();
        }
        cb_push_back(cb_diff, chunks * kt);
        cb_wait_front(cb_diff, chunks * kt);
        pack_reconfig_data_format(cb_out);
    }
    if (need_group) {
        cb_wait_front(cb_seed, state_tiles);
    }
    if (need_tail) {
        cb_wait_front(cb_tail, state_tiles);
    }
    for (uint32_t c = 0; c < chunks; ++c) {
        const uint32_t seed = c < group_chunks ? cb_seed : cb_tail;
        cb_reserve_back(cb_out, kt);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_base);
        copy_init(cb_base);
        for (uint32_t j = 0; j < kt; ++j) {
            copy_tile(cb_base, c * kt + j, j);
        }
        reconfig_data_format<SrcOrder::Reverse>(cb_diff, seed);
        matmul_block_init(cb_diff, seed, false, kt, 1, kt);
        for (uint32_t k = 0; k < kt; ++k) {
            matmul_block(cb_diff, seed, c * kt + k, k * kt, 0, false, kt, 1, kt);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < kt; ++j) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, kt);
    }
    if (chunks != 0) {
        cb_pop_front(cb_m, chunks * kt);
        cb_pop_front(cb_base, chunks * kt);
        cb_pop_front(cb_diff, chunks * kt);
    }
    // This core's tile rows of final = A @ seed + B.
    if (final) {
        const uint32_t seed = final_tail ? cb_tail : cb_seed;
        pack_reconfig_data_format(cb_final);
        for (uint32_t row = first_row; row < kt; row += row_stride) {
            cb_wait_front(cb_a, kt);
            cb_wait_front(cb_b, kt);
            cb_reserve_back(cb_final, kt);
            tile_regs_acquire();
            reconfig_data_format_srca(cb_b);
            copy_init(cb_b);
            for (uint32_t c = 0; c < kt; ++c) {
                copy_tile(cb_b, c, c);
            }
            reconfig_data_format<SrcOrder::Reverse>(cb_a, seed);
            matmul_block_init(cb_a, seed, false, kt, 1, kt);
            for (uint32_t k = 0; k < kt; ++k) {
                matmul_block(cb_a, seed, k, k * kt, 0, false, kt, 1, kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < kt; ++r) {
                pack_tile(r, cb_final);
            }
            tile_regs_release();
            cb_push_back(cb_final, kt);
            cb_pop_front(cb_a, kt);
            cb_pop_front(cb_b, kt);
        }
    }
    if (need_group) {
        cb_pop_front(cb_seed, state_tiles);
    }
    if (need_tail) {
        cb_pop_front(cb_tail, state_tiles);
    }
}
