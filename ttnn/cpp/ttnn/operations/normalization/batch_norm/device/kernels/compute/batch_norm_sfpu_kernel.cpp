// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/tile_move_copy.h"

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"

// batchnorm_bcast_tiles: For each output tile in [tile_start, freq), computes batch-norm on tiles from cb_other
// (input) broadcast against cb_bcast (batch mean). First builds 1/sqrt(batch_var + eps) in cb_den, then per tile:
// (input - mean) * den, optional multiply by weight, optional add bias. When NeedsOutputTypecast, SFPU typecasts
// from FP32 staging (cb_output_0) to writer-facing cb_output_final. Tracks last_srca_cb in/out so
// copy_tile_to_dst_init_short_with_dt can reconfigure the SrcA unpacker correctly across mixed dtypes.
template <bool NeedsOutputTypecast, uint32_t TcInFmt, uint32_t TcOutFmt>
ALWI uint32_t batchnorm_bcast_tiles(
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
    uint32_t dfb_output_final,
    uint32_t weight_has,
    uint32_t bias_has,
    uint32_t last_srca_dfb) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t index = 0;
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

    // 1/(sqrt(batch_var + eps)) = cb_den
    dfb_den_obj.reserve_back(onetile);
    dfb_batch_var_obj.wait_front(onetile);

    tile_regs_acquire();
    copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_batch_var);
    last_srca_dfb = dfb_batch_var;
    copy_tile(dfb_batch_var, index, index * 2);
    add_binary_tile_init();
    copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_eps);
    last_srca_dfb = dfb_eps;
    copy_tile(dfb_eps, index, index * 2 + 1);
    add_binary_tile(index * 2, index * 2 + 1, index * 2);
    rsqrt_tile_init();
    rsqrt_tile(index * 2);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(index * 2, dfb_den);
    tile_regs_release();

    dfb_den_obj.push_back(onetile);
    dfb_batch_var_obj.pop_front(onetile);

    dfb_bcast_obj.wait_front(onetile);  // input - batch_mean
    dfb_den_obj.wait_front(onetile);    // (input - batch_mean)/(sqrt(batch_var + eps)) = result
    if (weight_has_value) {            // result = result * weight
        dfb_weight_obj.wait_front(onetile);
    }
    if (bias_has_value) {  // result = result + bias
        dfb_bias_obj.wait_front(onetile);
    }
    for (uint32_t j = tile_start; j < freq; ++j) {
        dfb_other_obj.wait_front(onetile);
        dfb_affine_or_out_obj.reserve_back(onetile);

        // (input - batch_mean) * den
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_other);
        last_srca_dfb = dfb_other;
        copy_tile(dfb_other, index, index * 2);
        sub_binary_tile_init();
        copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_bcast);
        last_srca_dfb = dfb_bcast;
        copy_tile(dfb_bcast, index, index * 2 + 1);
        sub_binary_tile(index * 2, index * 2 + 1, index * 2);

        mul_binary_tile_init();
        copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_den);
        last_srca_dfb = dfb_den;
        copy_tile(dfb_den, index, index * 2 + 1);
        mul_binary_tile(index * 2, index * 2 + 1, index * 2);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(index * 2, dfb_affine_or_out);
        tile_regs_release();

        dfb_other_obj.pop_front(onetile);
        dfb_affine_or_out_obj.push_back(onetile);

        if (weight_has_value) {  // result = result * weight
            dfb_affine_or_out_obj.wait_front(onetile);
            dfb_scaled_output_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_affine_or_out);
            last_srca_dfb = dfb_affine_or_out;
            copy_tile(dfb_affine_or_out, index, index * 2);
            mul_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_weight);
            last_srca_dfb = dfb_weight;
            copy_tile(dfb_weight, index, index * 2 + 1);
            mul_binary_tile(index * 2, index * 2 + 1, index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(index * 2, dfb_scaled_output);
            tile_regs_release();

            dfb_scaled_output_obj.push_back(onetile);
            dfb_affine_or_out_obj.pop_front(onetile);
        }

        if (bias_has_value) {  // result = result + bias
            dfb_tmp_1_obj.wait_front(onetile);
            dfb_output_0_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_tmp_1);
            last_srca_dfb = dfb_tmp_1;
            copy_tile(dfb_tmp_1, index, index * 2);
            add_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_bias);
            last_srca_dfb = dfb_bias;
            copy_tile(dfb_bias, index, index * 2 + 1);
            add_binary_tile(index * 2, index * 2 + 1, index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(index * 2, dfb_output_0);
            tile_regs_release();

            dfb_output_0_obj.push_back(onetile);
            dfb_tmp_1_obj.pop_front(onetile);
        }

        if constexpr (NeedsOutputTypecast) {
            dfb_output_0_obj.wait_front(onetile);
            DataflowBuffer dfb_output_final_obj(dfb_output_final);
            dfb_output_final_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_output_0);
            last_srca_dfb = dfb_output_0;
            copy_tile(dfb_output_0, index, index * 2);
            typecast_tile_init<TcInFmt, TcOutFmt>();
            typecast_tile<TcInFmt, TcOutFmt>(index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_reconfig_data_format(dfb_output_final);
            pack_tile(index * 2, dfb_output_final);
            tile_regs_release();

            pack_reconfig_data_format(dfb_output_final, dfb_output_0);

            dfb_output_0_obj.pop_front(onetile);
            dfb_output_final_obj.push_back(onetile);
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
    return last_srca_dfb;
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
    constexpr auto dfb_batch_var = get_compile_time_arg_val(5);      // batch_var
    constexpr auto dfb_eps = get_compile_time_arg_val(6);            // eps
    constexpr auto dfb_den = get_compile_time_arg_val(7);            // 1/(sqrt(batch_var + eps))
    constexpr auto dfb_weight = get_compile_time_arg_val(8);         // weight tensor
    constexpr auto dfb_tmp_1 = get_compile_time_arg_val(9);          // (input - batch_mean)/(sqrt(batch_var + eps))
    constexpr auto dfb_bias = get_compile_time_arg_val(10);          // bias tensor
    constexpr auto dfb_output_final = get_compile_time_arg_val(11);  // writer-facing output CB (BF16 when typecast)
    constexpr bool needs_output_typecast = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t tc_in_fmt = get_compile_time_arg_val(13);
    constexpr uint32_t tc_out_fmt = get_compile_time_arg_val(14);

    auto dfb_bcast = dfb_batch_mean;
    auto dfb_other = dfb_input;

    unary_op_init_common(dfb_other, dfb_output_0);
    uint32_t last_srca_dfb = dfb_other;

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    constexpr uint32_t onetile = 1;
    DataflowBuffer dfb_eps_obj(dfb_eps);
    dfb_eps_obj.wait_front(onetile);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        last_srca_dfb = batchnorm_bcast_tiles<needs_output_typecast, tc_in_fmt, tc_out_fmt>(
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
            dfb_output_final,
            weight_has_value,
            bias_has_value,
            last_srca_dfb);
    }
    if (remaining_iterations > 0) {
        last_srca_dfb = batchnorm_bcast_tiles<needs_output_typecast, tc_in_fmt, tc_out_fmt>(
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
            dfb_output_final,
            weight_has_value,
            bias_has_value,
            last_srca_dfb);
    }

    dfb_eps_obj.pop_front(onetile);
}
