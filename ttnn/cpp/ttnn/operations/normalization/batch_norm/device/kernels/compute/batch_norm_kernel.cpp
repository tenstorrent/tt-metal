// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary.h"
#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

#include <cstdint>

namespace NAMESPACE {

ALWI void batchnorm_bcast_tiles(
    uint32_t cb_bcast,
    uint32_t cb_other,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t cb_batch_var,
    uint32_t cb_eps,
    uint32_t cb_den,
    uint32_t cb_weight,
    uint32_t cb_bias,
    uint32_t cb_tmp_1,
    uint32_t cb_output_0,
    uint32_t weight_has,
    uint32_t bias_has) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    uint32_t weight_has_value = weight_has;
    uint32_t bias_has_value = bias_has;
    auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
    auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;

    // 1/(sqrt(batch_var + eps))
    cb_reserve_back(cb_den, onetile);
    cb_wait_front(cb_batch_var, onetile);

    tile_regs_acquire();
    add_tiles_init_with_dt(cb_batch_var, cb_eps);
    add_tiles(cb_batch_var, cb_eps, 0, 0, dst0);
    rsqrt_tile_init();
    rsqrt_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_den);
    tile_regs_release();

    cb_pop_front(cb_batch_var, onetile);
    cb_push_back(cb_den, onetile);

    cb_wait_front(cb_bcast, onetile);
    cb_wait_front(cb_den, onetile);
    if (weight_has_value) {
        cb_wait_front(cb_weight, onetile);
    }
    if (bias_has_value) {
        cb_wait_front(cb_bias, onetile);
    }
    for (uint32_t j = tile_start; j < freq; ++j) {
        // input - batch_mean
        cb_wait_front(cb_other, onetile);
        cb_reserve_back(cb_affine_or_out, onetile);

        tile_regs_acquire();
        sub_tiles_init(cb_other, cb_bcast);
        sub_tiles(cb_other, cb_bcast, 0, 0, 0);

        // (input - batch_mean)/(sqrt(batch_var + eps)) = result
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_den, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(0, cb_affine_or_out);
        tile_regs_release();

        cb_push_back(cb_affine_or_out, onetile);
        cb_pop_front(cb_other, onetile);

        // result = result * weight
        if (weight_has_value) {
            cb_reserve_back(cb_scaled_output, onetile);
            cb_wait_front(cb_affine_or_out, 1);

            tile_regs_acquire();
            mul_tiles_init_with_dt(cb_affine_or_out, cb_weight);
            mul_tiles(cb_affine_or_out, cb_weight, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_scaled_output);
            tile_regs_release();

            cb_pop_front(cb_affine_or_out, 1);
            cb_push_back(cb_scaled_output, onetile);
        }

        // result = result + bias
        if (bias_has_value) {
            cb_reserve_back(cb_output_0, onetile);
            cb_wait_front(cb_tmp_1, onetile);

            tile_regs_acquire();
            add_tiles_init_with_dt(cb_tmp_1, cb_bias);
            add_tiles(cb_tmp_1, cb_bias, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_output_0);
            tile_regs_release();

            cb_pop_front(cb_tmp_1, onetile);
            cb_push_back(cb_output_0, onetile);
        }
    }
    cb_pop_front(cb_bcast, onetile);
    cb_pop_front(cb_den, onetile);
    if (weight_has_value) {
        cb_pop_front(cb_weight, onetile);
    }
    if (bias_has_value) {
        cb_pop_front(cb_bias, onetile);
    }
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    constexpr uint32_t weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bias_has_value = get_compile_time_arg_val(1) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_input = get_compile_time_arg_val(2);       // input
    constexpr auto cb_batch_mean = get_compile_time_arg_val(3);  // batch_mean
    constexpr auto cb_output_0 =
        get_compile_time_arg_val(4);  // output -- > [(input - batch_mean)/(sqrt(batch_var + eps))] * weight
    constexpr auto cb_batch_var = get_compile_time_arg_val(5);  // batch_var
    constexpr auto cb_eps = get_compile_time_arg_val(6);        // eps
    constexpr auto cb_den = get_compile_time_arg_val(7);        // 1/(sqrt(batch_var + eps))
    constexpr auto cb_weight = get_compile_time_arg_val(8);     // weight tensor
    constexpr auto cb_tmp_1 = get_compile_time_arg_val(9);      // (input - batch_mean)/(sqrt(batch_var + eps))
    constexpr auto cb_bias = get_compile_time_arg_val(10);      // bias tensor

    auto cb_bcast = cb_batch_mean;
    auto cb_other = cb_input;

    binary_op_init_common(cb_other, cb_bcast, cb_output_0);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    constexpr uint32_t onetile = 1;
    cb_wait_front(cb_eps, onetile);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles(
            cb_bcast,
            cb_other,
            tile_freq,
            tile_start,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_tmp_1,
            cb_output_0,
            weight_has_value,
            bias_has_value);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles(
            cb_bcast,
            cb_other,
            remaining_iterations,
            tile_start,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_tmp_1,
            cb_output_0,
            weight_has_value,
            bias_has_value);
    }

    cb_pop_front(cb_eps, onetile);
}
}  // namespace NAMESPACE
