// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
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

    experimental::CircularBuffer cb_batch_mean_obj(cb_batch_mean);
    experimental::CircularBuffer cb_batch_var_obj(cb_batch_var);
    experimental::CircularBuffer cb_out0_obj(cb_out0);
    experimental::CircularBuffer cb_old_running_mean_obj(cb_old_running_mean);
    experimental::CircularBuffer cb_old_running_var_obj(cb_old_running_var);
    experimental::CircularBuffer cb_updated_running_mean_obj(cb_updated_running_mean);
    experimental::CircularBuffer cb_updated_running_var_obj(cb_updated_running_var);
    experimental::CircularBuffer cb_momentum_obj(cb_momentum);
    experimental::CircularBuffer cb_one_obj(cb_one);
    experimental::CircularBuffer cb_tmp1_obj(cb_tmp1);
    experimental::CircularBuffer cb_tmp2_obj(cb_tmp2);
    experimental::CircularBuffer cb_tmp3_obj(cb_tmp3);

    unary_op_init_common(cb_batch_mean, cb_out0);
    constexpr uint32_t onetile = 1;

    cb_momentum_obj.wait_front(1);
    cb_one_obj.wait_front(1);

    // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        constexpr uint32_t tile_index = 0;

        cb_batch_mean_obj.wait_front(onetile);
        cb_out0_obj.reserve_back(1);

        if constexpr (old_running_mean_has_value) {
            // 1 - momentum
            cb_tmp1_obj.reserve_back(onetile);
            tile_regs_acquire();
            sub_binary_tile_init();
            // Use cb_batch_mean as old_cbid since unary_op_init_common configured the unpacker for it
            copy_tile_to_dst_init_short_with_dt(cb_batch_mean, cb_one);
            copy_tile(cb_one, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(cb_one, cb_momentum);
            copy_tile(cb_momentum, tile_index, tile_index * 2 + 1);
            sub_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, cb_tmp1);
            tile_regs_release();
            cb_tmp1_obj.push_back(onetile);

            // momentum * batch stat
            cb_tmp2_obj.reserve_back(onetile);
            tile_regs_acquire();
            mul_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(cb_momentum, cb_batch_mean);
            copy_tile(cb_batch_mean, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(cb_batch_mean, cb_momentum);
            copy_tile(cb_momentum, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, cb_tmp2);
            tile_regs_release();
            cb_tmp2_obj.push_back(onetile);

            // cb_tmp1 * running stats --> (1 - momentum) * running stats
            cb_tmp1_obj.wait_front(onetile);
            cb_old_running_mean_obj.wait_front(onetile);
            cb_tmp3_obj.reserve_back(onetile);
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(cb_tmp1, cb_old_running_mean);
            copy_tile(cb_old_running_mean, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(cb_old_running_mean, cb_tmp1);
            copy_tile(cb_tmp1, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, cb_tmp3);
            tile_regs_release();
            cb_tmp3_obj.push_back(onetile);

            cb_old_running_mean_obj.pop_front(onetile);
            cb_tmp1_obj.pop_front(onetile);

            // cb_tmp2 + cb_tmp3 --> (momentum * batch stat) + ((1 - momentum) * running stats)
            cb_tmp2_obj.wait_front(onetile);
            cb_tmp3_obj.wait_front(onetile);
            cb_updated_running_mean_obj.reserve_back(onetile);
            tile_regs_acquire();
            add_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(cb_tmp2, cb_tmp3);
            copy_tile(cb_tmp3, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(cb_tmp3, cb_tmp2);
            copy_tile(cb_tmp2, tile_index, tile_index * 2 + 1);
            add_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, cb_updated_running_mean);
            if constexpr (!old_running_var_has_value) {
                pack_tile_with_dt(tile_index * 2, cb_out0);
            }
            tile_regs_release();
            cb_updated_running_mean_obj.push_back(onetile);

            cb_tmp3_obj.pop_front(onetile);
            cb_tmp2_obj.pop_front(onetile);
        }

        cb_batch_mean_obj.pop_front(onetile);

        if constexpr (old_running_var_has_value) {
            // 1 - momentum
            cb_tmp1_obj.reserve_back(onetile);
            tile_regs_acquire();
            sub_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(cb_momentum, cb_one);
            copy_tile(cb_one, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(cb_one, cb_momentum);
            copy_tile(cb_momentum, tile_index, tile_index * 2 + 1);
            sub_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, cb_tmp1);
            tile_regs_release();
            cb_tmp1_obj.push_back(onetile);

            // momentum * batch stat
            cb_batch_var_obj.wait_front(onetile);
            cb_tmp2_obj.reserve_back(onetile);
            tile_regs_acquire();
            mul_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(cb_momentum, cb_batch_var);
            copy_tile(cb_batch_var, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_momentum);
            copy_tile(cb_momentum, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, cb_tmp2);
            tile_regs_release();
            cb_tmp2_obj.push_back(onetile);

            cb_batch_var_obj.pop_front(onetile);

            // cb_tmp1 * running stats --> (1 - momentum) * running stats
            cb_tmp1_obj.wait_front(onetile);
            cb_old_running_var_obj.wait_front(onetile);
            cb_tmp3_obj.reserve_back(onetile);
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(cb_tmp1, cb_old_running_var);
            copy_tile(cb_old_running_var, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(cb_old_running_var, cb_tmp1);
            copy_tile(cb_tmp1, tile_index, tile_index * 2 + 1);
            mul_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, cb_tmp3);
            tile_regs_release();
            cb_tmp3_obj.push_back(onetile);

            cb_old_running_var_obj.pop_front(onetile);
            cb_tmp1_obj.pop_front(onetile);

            // cb_tmp2 + cb_tmp3 --> (momentum * batch stat) + ((1 - momentum) * running stats)
            cb_tmp2_obj.wait_front(onetile);
            cb_tmp3_obj.wait_front(onetile);
            cb_updated_running_var_obj.reserve_back(onetile);
            tile_regs_acquire();
            add_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(cb_tmp2, cb_tmp3);
            copy_tile(cb_tmp3, tile_index, tile_index * 2);
            copy_tile_to_dst_init_short_with_dt(cb_tmp3, cb_tmp2);
            copy_tile(cb_tmp2, tile_index, tile_index * 2 + 1);
            add_binary_tile(tile_index * 2, tile_index * 2 + 1, tile_index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(tile_index * 2, cb_updated_running_var);
            pack_tile_with_dt(tile_index * 2, cb_out0);
            tile_regs_release();
            cb_updated_running_var_obj.push_back(onetile);

            cb_tmp3_obj.pop_front(onetile);
            cb_tmp2_obj.pop_front(onetile);
        }

        cb_out0_obj.push_back(1);
    }
    cb_momentum_obj.pop_front(1);
    cb_one_obj.pop_front(1);
}
