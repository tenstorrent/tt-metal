// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t old_running_mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t old_running_var_has_value = get_compile_time_arg_val(1) == 1;

    constexpr auto cb_batch_mean = get_compile_time_arg_val(2);  // batch mean
    constexpr auto cb_batch_var = get_compile_time_arg_val(3);   // batch var
    constexpr auto cb_out0 = get_compile_time_arg_val(4);
    constexpr auto cb_old_running_mean = get_compile_time_arg_val(5);      // old running mean tensor
    constexpr auto cb_old_running_var = get_compile_time_arg_val(6);       // old running var tensor
    constexpr auto cb_updated_running_mean = get_compile_time_arg_val(7);  // updated running mean tensor
    constexpr auto cb_updated_running_var = get_compile_time_arg_val(8);   // updated running var tensor
    constexpr auto cb_momentum = get_compile_time_arg_val(9);              // momentum
    constexpr auto cb_one = get_compile_time_arg_val(10);                  // stores 1
    constexpr auto cb_tmp1 = get_compile_time_arg_val(11);                 // tmp 1
    constexpr auto cb_tmp2 = get_compile_time_arg_val(12);                 // tmp 2
    constexpr auto cb_tmp3 = get_compile_time_arg_val(13);                 // tmp 3

    unary_op_init_common(cb_batch_mean, cb_out0);
    constexpr uint32_t onetile = 1;

    // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        tile_regs_acquire();
        cb_wait_front(cb_one, 1);
        cb_wait_front(cb_momentum, 1);

        if constexpr (old_running_mean_has_value) {
            // 1 - momentum
            cb_reserve_back(cb_tmp1, onetile);
            sub_binary_tile_init();
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile_to_dst_init_short_with_dt(cb_momentum, cb_one);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_one, i, i * 2);
            }
            copy_tile_to_dst_init_short_with_dt(cb_one, cb_momentum);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_momentum, i, i * 2 + 1);
                sub_binary_tile(i * 2, i * 2 + 1);
                tile_regs_commit();
                pack_tile(i * 2, cb_tmp1);
            }
            tile_regs_release();
            cb_push_back(cb_tmp1, onetile);

            // momentum * batch stat
            cb_wait_front(cb_batch_mean, onetile);
            cb_reserve_back(cb_tmp2, onetile);
            mul_binary_tile_init();
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile_to_dst_init_short_with_dt(cb_momentum, cb_batch_mean);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_batch_mean, i, i * 2);
            }
            copy_tile_to_dst_init_short_with_dt(cb_batch_mean, cb_momentum);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_momentum, i, i * 2 + 1);
                mul_binary_tile(i * 2, i * 2 + 1);
                tile_regs_commit();
                pack_tile(i * 2, cb_tmp2);
            }
            tile_regs_release();
            cb_push_back(cb_tmp2, onetile);
            cb_pop_front(cb_batch_mean, onetile);

            // cb_tmp1 * running stats --> (1 - momentum) * running stats
            cb_wait_front(cb_tmp1, onetile);
            cb_wait_front(cb_old_running_mean, onetile);
            cb_reserve_back(cb_tmp3, onetile);
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile_to_dst_init_short_with_dt(cb_tmp1, cb_old_running_mean);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_old_running_mean, i, i * 2);
            }
            copy_tile_to_dst_init_short_with_dt(cb_old_running_mean, cb_tmp1);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_tmp1, i, i * 2 + 1);
                mul_binary_tile(i * 2, i * 2 + 1);
                tile_regs_commit();
                pack_tile(i * 2, cb_tmp3);
            }
            tile_regs_release();
            cb_push_back(cb_tmp3, onetile);
            cb_pop_front(cb_old_running_mean, onetile);
            cb_pop_front(cb_tmp1, onetile);

            // cb_tmp2 + cb_tmp3 --> (momentum * batch stat) + ((1 - momentum) * running stats)
            cb_wait_front(cb_tmp2, onetile);
            cb_wait_front(cb_tmp3, onetile);

            cb_reserve_back(cb_updated_running_mean, onetile);

            add_binary_tile_init();
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile_to_dst_init_short_with_dt(cb_tmp2, cb_tmp3);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_tmp3, i, i * 2);
            }
            copy_tile_to_dst_init_short_with_dt(cb_tmp3, cb_tmp2);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_tmp2, i, i * 2 + 1);
                add_binary_tile(i * 2, i * 2 + 1);
                tile_regs_commit();
                pack_tile(i * 2, cb_updated_running_mean);
            }
            tile_regs_release();
            cb_push_back(cb_updated_running_mean, onetile);
            cb_pop_front(cb_tmp3, onetile);
            cb_pop_front(cb_tmp2, onetile);
        }
        if constexpr (old_running_var_has_value) {
            // 1 - momentum
            cb_reserve_back(cb_tmp1, onetile);
            sub_binary_tile_init();
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile_to_dst_init_short_with_dt(cb_momentum, cb_one);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_one, i, i * 2);
            }
            copy_tile_to_dst_init_short_with_dt(cb_one, cb_momentum);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_momentum, i, i * 2 + 1);
                sub_binary_tile(i * 2, i * 2 + 1);
                tile_regs_commit();
                pack_tile(i * 2, cb_tmp1);
            }
            tile_regs_release();
            cb_push_back(cb_tmp1, onetile);

            // momentum * batch stat
            cb_wait_front(cb_batch_var, onetile);
            cb_reserve_back(cb_tmp2, onetile);
            mul_binary_tile_init();
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile_to_dst_init_short_with_dt(cb_momentum, cb_batch_var);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_batch_var, i, i * 2);
            }
            copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_momentum);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_momentum, i, i * 2 + 1);
                mul_binary_tile(i * 2, i * 2 + 1);
                tile_regs_commit();
                pack_tile(i * 2, cb_tmp2);
            }
            tile_regs_release();
            cb_push_back(cb_tmp2, onetile);
            cb_pop_front(cb_batch_var, onetile);

            // cb_tmp1 * running stats --> (1 - momentum) * running stats
            cb_wait_front(cb_tmp1, onetile);
            cb_wait_front(cb_old_running_var, onetile);
            cb_reserve_back(cb_tmp3, onetile);
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile_to_dst_init_short_with_dt(cb_tmp1, cb_old_running_var);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_old_running_var, i, i * 2);
            }
            copy_tile_to_dst_init_short_with_dt(cb_old_running_var, cb_tmp1);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_tmp1, i, i * 2 + 1);
                mul_binary_tile(i * 2, i * 2 + 1);
                tile_regs_commit();
                pack_tile(i * 2, cb_tmp3);
            }
            tile_regs_release();
            cb_push_back(cb_tmp3, onetile);
            cb_pop_front(cb_old_running_var, onetile);
            cb_pop_front(cb_tmp1, onetile);

            // cb_tmp2 + cb_tmp3 --> (momentum * batch stat) + ((1 - momentum) * running stats)
            cb_wait_front(cb_tmp2, onetile);
            cb_wait_front(cb_tmp3, onetile);

            cb_reserve_back(cb_updated_running_var, onetile);

            add_binary_tile_init();
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile_to_dst_init_short_with_dt(cb_tmp2, cb_tmp3);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_tmp3, i, i * 2);
            }
            copy_tile_to_dst_init_short_with_dt(cb_tmp3, cb_tmp2);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_tmp2, i, i * 2 + 1);
                add_binary_tile(i * 2, i * 2 + 1);
                tile_regs_commit();
                pack_tile(i * 2, cb_updated_running_var);
            }
            tile_regs_release();
            cb_push_back(cb_updated_running_var, onetile);
            cb_pop_front(cb_tmp3, onetile);
            cb_pop_front(cb_tmp2, onetile);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out0);
    tile_regs_release();
    cb_pop_front(cb_momentum, 1);
    cb_pop_front(cb_one, 1);
    cb_push_back(cb_out0, 1);
}
}  // namespace NAMESPACE
