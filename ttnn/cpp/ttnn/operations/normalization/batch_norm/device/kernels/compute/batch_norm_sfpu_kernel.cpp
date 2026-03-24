// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary_sfpu.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/typecast.h"

#include <cstdint>

#include "experimental/circular_buffer.h"

template <bool NeedsOutputTypecast, uint32_t TcInFmt, uint32_t TcOutFmt>
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
    uint32_t cb_output_final,
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
    for (uint32_t i = 0; i < onetile; ++i) {
        copy_tile(cb_batch_var, i, i * 2);
    }
    add_binary_tile_init();
    copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_eps);
    last_srca_cb = cb_eps;
    for (uint32_t i = 0; i < onetile; ++i) {
        copy_tile(cb_eps, i, i * 2 + 1);

        add_binary_tile(i * 2, i * 2 + 1, i * 2);
    }
    rsqrt_tile_init();
    for (uint32_t i = 0; i < onetile; ++i) {
        rsqrt_tile(i * 2);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < onetile; ++i) {
        pack_tile(i * 2, cb_den);
    }
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
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_other, i, i * 2);
        }
        sub_binary_tile_init();
        copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_bcast);
        last_srca_cb = cb_bcast;
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_bcast, i, i * 2 + 1);
            sub_binary_tile(i * 2, i * 2 + 1, i * 2);
        }

        mul_binary_tile_init();
        copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_den);
        last_srca_cb = cb_den;
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_den, i, i * 2 + 1);
            mul_binary_tile(i * 2, i * 2 + 1, i * 2);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < onetile; ++i) {
            pack_tile(i * 2, cb_affine_or_out);
        }
        tile_regs_release();

        cb_other_obj.pop_front(onetile);
        cb_affine_or_out_obj.push_back(onetile);

        if (weight_has_value) {  // result = result * weight
            cb_affine_or_out_obj.wait_front(onetile);
            cb_scaled_output_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_affine_or_out);
            last_srca_cb = cb_affine_or_out;
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_affine_or_out, i, i * 2);
            }
            mul_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_weight);
            last_srca_cb = cb_weight;
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_weight, i, i * 2 + 1);
                mul_binary_tile(i * 2, i * 2 + 1, i * 2);
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < onetile; ++i) {
                pack_tile(i * 2, cb_scaled_output);
            }
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
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_tmp_1, i, i * 2);
            }
            add_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_bias);
            last_srca_cb = cb_bias;
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_bias, i, i * 2 + 1);
                add_binary_tile(i * 2, i * 2 + 1, i * 2);
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < onetile; ++i) {
                pack_tile(i * 2, cb_output_0);
            }
            tile_regs_release();

            cb_output_0_obj.push_back(onetile);
            cb_tmp_1_obj.pop_front(onetile);
        }

        if constexpr (NeedsOutputTypecast) {
            cb_output_0_obj.wait_front(onetile);
            experimental::CircularBuffer cb_output_final_obj(cb_output_final);
            cb_output_final_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_cb, cb_output_0);
            last_srca_cb = cb_output_0;
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_output_0, i, i * 2);
            }
            typecast_tile_init<TcInFmt, TcOutFmt>();
            for (uint32_t i = 0; i < onetile; ++i) {
                typecast_tile<TcInFmt, TcOutFmt>(i * 2);
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_reconfig_data_format(cb_output_final);
            for (uint32_t i = 0; i < onetile; ++i) {
                pack_tile(i * 2, cb_output_final);
            }
            tile_regs_release();

            pack_reconfig_data_format(cb_output_final, cb_output_0);

            cb_output_0_obj.pop_front(onetile);
            cb_output_final_obj.push_back(onetile);
        }
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
    constexpr bool needs_output_typecast = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t tc_in_fmt = get_compile_time_arg_val(13);
    constexpr uint32_t tc_out_fmt = get_compile_time_arg_val(14);

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
        last_srca_cb = batchnorm_bcast_tiles<needs_output_typecast, tc_in_fmt, tc_out_fmt>(
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
        last_srca_cb = batchnorm_bcast_tiles<needs_output_typecast, tc_in_fmt, tc_out_fmt>(
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
