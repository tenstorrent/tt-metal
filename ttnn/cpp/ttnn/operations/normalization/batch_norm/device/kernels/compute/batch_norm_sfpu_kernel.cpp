// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary_sfpu.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#ifdef TYPECAST_OUTPUT
#include "api/compute/eltwise_unary/typecast.h"
#endif

#include <cstdint>

#include "experimental/circular_buffer.h"

ALWI uint32_t batchnorm_bcast_tiles(
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
    [[maybe_unused]] uint32_t cb_output_final,
    uint32_t weight_has,
    uint32_t bias_has,
    uint32_t last_srca_cb) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    uint32_t weight_has_value = weight_has;
    uint32_t bias_has_value = bias_has;
    auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
    auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;

    experimental::CircularBuffer cb_bcast_obj(cb_bcast);
    experimental::CircularBuffer cb_other_obj(cb_other);
    experimental::CircularBuffer cb_batch_var_obj(cb_batch_var);
    experimental::CircularBuffer cb_den_obj(cb_den);
    experimental::CircularBuffer cb_weight_obj(cb_weight);
    experimental::CircularBuffer cb_bias_obj(cb_bias);
    experimental::CircularBuffer cb_tmp_1_obj(cb_tmp_1);
    experimental::CircularBuffer cb_output_0_obj(cb_output_0);
    experimental::CircularBuffer cb_affine_or_out_obj(cb_affine_or_out);
    experimental::CircularBuffer cb_scaled_output_obj(cb_scaled_output);

    // 1/(sqrt(batch_var + eps)) = cb_den
    cb_den_obj.reserve_back(onetile);
    cb_batch_var_obj.wait_front(onetile);

    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_batch_var);
    last_srca_cb = cb_batch_var;
    copy_tile(cb_batch_var, 0, 0);
    add_binary_tile_init();
    copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_eps);
    last_srca_cb = cb_eps;
    copy_tile(cb_eps, 0, 1);
    add_binary_tile(0, 1, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_den);
    tile_regs_release();

    cb_den_obj.push_back(onetile);
    cb_batch_var_obj.pop_front(onetile);

    cb_bcast_obj.wait_front(onetile);  // input - batch_mean
    cb_den_obj.wait_front(onetile);    // (input - batch_mean)/(sqrt(batch_var + eps)) = result
    if (weight_has_value) {            // result = result * weight
        cb_weight_obj.wait_front(onetile);
    }
    if (bias_has_value) {  // result = result + bias
        cb_bias_obj.wait_front(onetile);
    }
    for (uint32_t j = tile_start; j < freq; ++j) {
        cb_other_obj.wait_front(onetile);
        cb_affine_or_out_obj.reserve_back(onetile);

        // (input - batch_mean) * den
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_other);
        last_srca_cb = cb_other;
        copy_tile(cb_other, 0, 0);
        sub_binary_tile_init();
        copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_bcast);
        last_srca_cb = cb_bcast;
        copy_tile(cb_bcast, 0, 1);
        sub_binary_tile(0, 1, 0);

        mul_binary_tile_init();
        copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_den);
        last_srca_cb = cb_den;
        copy_tile(cb_den, 0, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_affine_or_out);
        tile_regs_release();

        cb_other_obj.pop_front(onetile);
        cb_affine_or_out_obj.push_back(onetile);

        if (weight_has_value) {  // result = result * weight
            cb_affine_or_out_obj.wait_front(onetile);
            cb_scaled_output_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_affine_or_out);
            last_srca_cb = cb_affine_or_out;
            copy_tile(cb_affine_or_out, 0, 0);
            mul_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_weight);
            last_srca_cb = cb_weight;
            copy_tile(cb_weight, 0, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_scaled_output);
            tile_regs_release();

            cb_scaled_output_obj.push_back(onetile);
            cb_affine_or_out_obj.pop_front(onetile);
        }

        if (bias_has_value) {  // result = result + bias
            cb_tmp_1_obj.wait_front(onetile);
            cb_output_0_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_tmp_1);
            last_srca_cb = cb_tmp_1;
            copy_tile(cb_tmp_1, 0, 0);
            add_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_bias);
            last_srca_cb = cb_bias;
            copy_tile(cb_bias, 0, 1);
            add_binary_tile(0, 1, 0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, cb_output_0);
            tile_regs_release();

            cb_output_0_obj.push_back(onetile);
            cb_tmp_1_obj.pop_front(onetile);
        }

#ifdef TYPECAST_OUTPUT
        cb_output_0_obj.wait_front(onetile);
        experimental::CircularBuffer cb_output_final_obj(cb_output_final);
        cb_output_final_obj.reserve_back(onetile);

        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_output_0);
        last_srca_cb = cb_output_0;
        copy_tile(cb_output_0, 0, 0);
        TYPECAST_OUTPUT_INIT();
        TYPECAST_OUTPUT(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_reconfig_data_format(cb_output_final);
        pack_tile(0, cb_output_final);
        tile_regs_release();

        pack_reconfig_data_format(cb_output_final, cb_output_0);

        cb_output_0_obj.pop_front(onetile);
        cb_output_final_obj.push_back(onetile);
#endif
    }
    cb_bcast_obj.pop_front(onetile);
    cb_den_obj.pop_front(onetile);
    if (weight_has_value) {
        cb_weight_obj.pop_front(onetile);
    }
    if (bias_has_value) {
        cb_bias_obj.pop_front(onetile);
    }
    return last_srca_cb;
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
    constexpr auto cb_output_final = get_compile_time_arg_val(11);  // writer-facing output CB (BF16 when typecast)

    auto cb_bcast = cb_batch_mean;
    auto cb_other = cb_input;

    unary_op_init_common(cb_other, cb_output_0);
    uint32_t last_srca_cb = cb_other;

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    constexpr uint32_t onetile = 1;
    experimental::CircularBuffer cb_eps_obj(cb_eps);
    cb_eps_obj.wait_front(onetile);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        last_srca_cb = batchnorm_bcast_tiles(
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
            cb_output_final,
            weight_has_value,
            bias_has_value,
            last_srca_cb);
    }
    if (remaining_iterations > 0) {
        last_srca_cb = batchnorm_bcast_tiles(
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
            cb_output_final,
            weight_has_value,
            bias_has_value,
            last_srca_cb);
    }

    cb_eps_obj.pop_front(onetile);
}
