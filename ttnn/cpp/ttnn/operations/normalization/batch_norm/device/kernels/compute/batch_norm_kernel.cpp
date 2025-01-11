// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary.h"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

#include <cstdint>

namespace NAMESPACE {

ALWI void subtract_bcast_tiles(
    uint32_t cb_bcast, uint32_t cb_other, uint32_t cb_out, uint32_t freq, uint32_t tile_start) {
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_bcast, onetile);

    for (uint32_t j = tile_start; j < freq; ++j) {
        cb_wait_front(cb_other, onetile);
        cb_reserve_back(cb_out, onetile);

        tile_regs_acquire();
        sub_tiles(cb_other, cb_bcast, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, onetile);
        cb_pop_front(cb_other, onetile);
    }
    // cb_pop_front(cb_bcast, onetile);
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    constexpr uint32_t weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bias_has_value = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t is_training_mode = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t old_running_mean_has_value = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t old_running_var_has_value = get_compile_time_arg_val(4) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_input = tt::CBIndex::c_0;       // input
    constexpr auto cb_batch_mean = tt::CBIndex::c_1;  // batch_mean
    constexpr auto cb_output_0 =
        tt::CBIndex::c_2;  // output -- > [(input - batch_mean)/(sqrt(batch_var + eps))] * weight
    constexpr auto cb_batch_var = tt::CBIndex::c_3;  // batch_var
    constexpr auto cb_eps = tt::CBIndex::c_4;        // eps
    constexpr auto cb_den = tt::CBIndex::c_5;        // 1/(sqrt(batch_var + eps))
    constexpr auto cb_num = tt::CBIndex::c_6;        // input - batch_mean
    constexpr auto cb_weight = tt::CBIndex::c_16;    // weight tensor
    constexpr auto cb_tmp_1 = tt::CBIndex::c_17;     // (input - batch_mean)/(sqrt(batch_var + eps))
    constexpr auto cb_bias = tt::CBIndex::c_18;      // bias tensor
    constexpr auto cb_old_running_mean = tt::CBIndex::c_25;  // old running mean tensor
    constexpr auto cb_old_running_var = tt::CBIndex::c_26;   // old running var tensor
    constexpr auto cb_updated_running_mean = tt::CBIndex::c_27;  // updated running mean tensor
    constexpr auto cb_updated_running_var = tt::CBIndex::c_28;   // updated running var tensor
    constexpr auto cb_momentum = tt::CBIndex::c_24;              // momentum
    constexpr auto cb_one = tt::CBIndex::c_19;                   // stores 1
    constexpr auto cb_tmp1 = tt::CBIndex::c_29;                  // tmp 1
    constexpr auto cb_tmp2 = tt::CBIndex::c_30;                  // tmp 2
    constexpr auto cb_tmp3 = tt::CBIndex::c_31;                  // tmp 3

    auto cb_bcast = cb_batch_mean;
    auto cb_other = cb_input;

    binary_op_init_common(cb_bcast, cb_other, cb_output_0);

    // input - batch_mean
    sub_tiles_init();
    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;
    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        subtract_bcast_tiles(cb_bcast, cb_other, cb_num, tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        subtract_bcast_tiles(cb_bcast, cb_other, cb_num, remaining_iterations, tile_start);
    }

    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    constexpr auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
    constexpr auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // 1/(sqrt(batch_var + eps))
        cb_reserve_back(cb_den, onetile);
        cb_wait_front(cb_batch_var, 1);
        cb_wait_front(cb_eps, 1);

        tile_regs_acquire();
        add_tiles_init_with_dt(cb_batch_var, cb_eps);
        add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_den);
        tile_regs_release();

        // cb_pop_front(cb_batch_var, 1);
        cb_pop_front(cb_eps, 1);
        cb_push_back(cb_den, onetile);

        // (input - batch_mean)/(sqrt(batch_var + eps)) = result
        cb_reserve_back(cb_affine_or_out, onetile);
        cb_wait_front(cb_num, 1);
        cb_wait_front(cb_den, 1);

        tile_regs_acquire();
        mul_tiles_init_with_dt(cb_num, cb_den);
        mul_tiles(cb_num, cb_den, 0, 0, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_affine_or_out);
        tile_regs_release();

        cb_pop_front(cb_num, 1);
        cb_pop_front(cb_den, 1);
        cb_push_back(cb_affine_or_out, onetile);

        if constexpr (is_training_mode) {
            // updated running stats
            if constexpr (old_running_mean_has_value) {
                sub_tiles_to_cb(cb_one, cb_momentum, cb_tmp1, tile_id, 0, 0, 0);           // 1 - momentum
                mul_tiles_to_cb(cb_momentum, cb_batch_mean, cb_tmp2, 0, tile_id, 0, 0);    // momentum * running stats
                mul_tiles_to_cb(cb_tmp1, cb_old_running_mean, cb_tmp3, 0, tile_id, 1, 0);  // cb_tmp1 * batch stat
                add_tiles_to_cb(cb_tmp2, cb_tmp3, cb_updated_running_mean, 0, 0, 1, 1);
            }

            if constexpr (old_running_var_has_value) {
                sub_tiles_to_cb(cb_one, cb_momentum, cb_tmp1, tile_id, 0, 0, 0);          // 1 - momentum
                mul_tiles_to_cb(cb_momentum, cb_batch_var, cb_tmp2, 0, tile_id, 0, 0);    // momentum * batch stat
                mul_tiles_to_cb(cb_tmp1, cb_old_running_var, cb_tmp3, 0, tile_id, 0, 1);  // cb_tmp1 * running stats
                DPRINT << TSLICE(tt::CBIndex::c_26, 0, SliceRange::hw0_32_16()) << ENDL();
                add_tiles_to_cb(cb_tmp2, cb_tmp3, cb_updated_running_var, 0, 0, 1, 1);
            }
        }

        if constexpr (weight_has_value) {  // result = result * weight
            cb_reserve_back(cb_scaled_output, onetile);
            cb_wait_front(cb_affine_or_out, 1);
            cb_wait_front(cb_weight, 1);

            tile_regs_acquire();
            mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
            mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_scaled_output);
            tile_regs_release();

            cb_pop_front(cb_affine_or_out, 1);
            cb_pop_front(cb_weight, 1);
            cb_push_back(cb_scaled_output, onetile);
        }
        if constexpr (bias_has_value) {  // result = result + bias
            cb_reserve_back(cb_output_0, 1);
            cb_wait_front(cb_tmp_1, 1);
            cb_wait_front(cb_bias, 1);

            tile_regs_acquire();
            add_tiles_init_with_dt(cb_tmp_1, cb_bias);
            add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_output_0);
            tile_regs_release();

            cb_pop_front(cb_tmp_1, 1);
            cb_pop_front(cb_bias, 1);
            cb_push_back(cb_output_0, 1);
        }
    }
}
}  // namespace NAMESPACE
