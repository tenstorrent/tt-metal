// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"

ALWI void batchnorm_bcast_tiles(
    uint32_t dfb_bcast,
    uint32_t dfb_other,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t dfb_batch_var,
    uint32_t dfb_eps,
    uint32_t dfb_den,
    uint32_t dfb_weight,
    uint32_t dfb_bias,
    uint32_t dfb_tmp_1,
    uint32_t dfb_output_0,
    uint32_t weight_has,
    uint32_t bias_has) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    uint32_t weight_has_value = weight_has;
    uint32_t bias_has_value = bias_has;
    auto dfb_affine_or_out = (weight_has_value || bias_has_value) ? dfb_tmp_1 : dfb_output_0;
    auto dfb_scaled_output = (bias_has_value) ? dfb_tmp_1 : dfb_output_0;

    DataflowBuffer dfb_bcast_obj(dfb_bcast);
    DataflowBuffer dfb_other_obj(dfb_other);
    DataflowBuffer dfb_batch_var_obj(dfb_batch_var);
    DataflowBuffer dfb_den_obj(dfb_den);
    DataflowBuffer dfb_weight_obj(dfb_weight);
    DataflowBuffer dfb_bias_obj(dfb_bias);
    DataflowBuffer dfb_tmp_1_obj(dfb_tmp_1);
    DataflowBuffer dfb_output_0_obj(dfb_output_0);
    DataflowBuffer dfb_affine_or_out_obj(dfb_affine_or_out);
    DataflowBuffer dfb_scaled_output_obj(dfb_scaled_output);

    // 1/(sqrt(batch_var + eps))
    dfb_den_obj.reserve_back(onetile);
    dfb_batch_var_obj.wait_front(onetile);

    tile_regs_acquire();
    add_tiles_init_with_dt(dfb_batch_var, dfb_eps);
    add_tiles(dfb_batch_var, dfb_eps, 0, 0, dst0);
    rsqrt_tile_init();
    rsqrt_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, dfb_den);
    tile_regs_release();

    dfb_batch_var_obj.pop_front(onetile);
    dfb_den_obj.push_back(onetile);

    dfb_bcast_obj.wait_front(onetile);
    dfb_den_obj.wait_front(onetile);
    if (weight_has_value) {
        dfb_weight_obj.wait_front(onetile);
    }
    if (bias_has_value) {
        dfb_bias_obj.wait_front(onetile);
    }
    for (uint32_t j = tile_start; j < freq; ++j) {
        // input - batch_mean
        dfb_other_obj.wait_front(onetile);
        dfb_affine_or_out_obj.reserve_back(onetile);

        tile_regs_acquire();
        sub_tiles_init(dfb_other, dfb_bcast);
        sub_tiles(dfb_other, dfb_bcast, 0, 0, 0);

        // (input - batch_mean)/(sqrt(batch_var + eps)) = result
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb_den);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb_den, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(0, dfb_affine_or_out);
        tile_regs_release();

        dfb_affine_or_out_obj.push_back(onetile);
        dfb_other_obj.pop_front(onetile);

        // result = result * weight
        if (weight_has_value) {
            dfb_scaled_output_obj.reserve_back(onetile);
            dfb_affine_or_out_obj.wait_front(1);

            tile_regs_acquire();
            mul_tiles_init_with_dt(dfb_affine_or_out, dfb_weight);
            mul_tiles(dfb_affine_or_out, dfb_weight, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_scaled_output);
            tile_regs_release();

            dfb_affine_or_out_obj.pop_front(1);
            dfb_scaled_output_obj.push_back(onetile);
        }

        // result = result + bias
        if (bias_has_value) {
            dfb_output_0_obj.reserve_back(onetile);
            dfb_tmp_1_obj.wait_front(onetile);

            tile_regs_acquire();
            add_tiles_init_with_dt(dfb_tmp_1, dfb_bias);
            add_tiles(dfb_tmp_1, dfb_bias, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_output_0);
            tile_regs_release();

            dfb_tmp_1_obj.pop_front(onetile);
            dfb_output_0_obj.push_back(onetile);
        }
    }
    dfb_bcast_obj.pop_front(onetile);
    dfb_den_obj.pop_front(onetile);
    if (weight_has_value) {
        dfb_weight_obj.pop_front(onetile);
    }
    if (bias_has_value) {
        dfb_bias_obj.pop_front(onetile);
    }
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    constexpr uint32_t weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bias_has_value = get_compile_time_arg_val(1) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr auto dfb_input = get_compile_time_arg_val(2);       // input
    constexpr auto dfb_batch_mean = get_compile_time_arg_val(3);  // batch_mean
    constexpr auto dfb_output_0 =
        get_compile_time_arg_val(4);  // output -- > [(input - batch_mean)/(sqrt(batch_var + eps))] * weight
    constexpr auto dfb_batch_var = get_compile_time_arg_val(5);  // batch_var
    constexpr auto dfb_eps = get_compile_time_arg_val(6);        // eps
    constexpr auto dfb_den = get_compile_time_arg_val(7);        // 1/(sqrt(batch_var + eps))
    constexpr auto dfb_weight = get_compile_time_arg_val(8);     // weight tensor
    constexpr auto dfb_tmp_1 = get_compile_time_arg_val(9);      // (input - batch_mean)/(sqrt(batch_var + eps))
    constexpr auto dfb_bias = get_compile_time_arg_val(10);      // bias tensor

    auto dfb_bcast = dfb_batch_mean;
    auto dfb_other = dfb_input;

    binary_op_init_common(dfb_other, dfb_bcast, dfb_output_0);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    constexpr uint32_t onetile = 1;
    DataflowBuffer dfb_eps_obj(dfb_eps);
    dfb_eps_obj.wait_front(onetile);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles(
            dfb_bcast,
            dfb_other,
            tile_freq,
            tile_start,
            dfb_batch_var,
            dfb_eps,
            dfb_den,
            dfb_weight,
            dfb_bias,
            dfb_tmp_1,
            dfb_output_0,
            weight_has_value,
            bias_has_value);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles(
            dfb_bcast,
            dfb_other,
            remaining_iterations,
            tile_start,
            dfb_batch_var,
            dfb_eps,
            dfb_den,
            dfb_weight,
            dfb_bias,
            dfb_tmp_1,
            dfb_output_0,
            weight_has_value,
            bias_has_value);
    }

    dfb_eps_obj.pop_front(onetile);
}
