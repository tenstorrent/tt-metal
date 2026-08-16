// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

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

    DataflowBuffer dfb_bcast_obj(dfb_bcast);          // batch_mean, broadcast against the input
    DataflowBuffer dfb_other_obj(dfb_other);          // input tiles
    DataflowBuffer dfb_batch_var_obj(dfb_batch_var);  // batch_var
    DataflowBuffer dfb_den_obj(dfb_den);              // 1/(sqrt(batch_var + eps))
    DataflowBuffer dfb_weight_obj(dfb_weight);        // weight tensor
    DataflowBuffer dfb_bias_obj(dfb_bias);            // bias tensor
    DataflowBuffer dfb_tmp_1_obj(dfb_tmp_1);          // (input - batch_mean)/(sqrt(batch_var + eps))
    // output -- > [(input - batch_mean)/(sqrt(batch_var + eps))] * weight
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
        sub_init(dfb_other, dfb_bcast);
        sub_tiles(dfb_other, dfb_bcast, 0, 0, 0);

        // (input - batch_mean)/(sqrt(batch_var + eps)) = result
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb_den);
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(dfb_den, 0, 0);
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
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);
    constexpr uint32_t weight_has_value = get_arg(args::weight_has_value) == 1;
    constexpr uint32_t bias_has_value = get_arg(args::bias_has_value) == 1;

    if (num_tiles == 0) {
        return;
    }

    // The batch mean is the broadcast operand of the subtraction; the input tiles are the other one.
    constexpr auto dfb_bcast = dfb::batch_mean;
    constexpr auto dfb_other = dfb::input;

    compute_kernel_hw_startup(dfb_other, dfb_bcast, dfb::out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    constexpr uint32_t onetile = 1;
    DataflowBuffer dfb_eps_obj(dfb::eps);  // one tile of eps, filled by the reader
    dfb_eps_obj.wait_front(onetile);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles(
            dfb_bcast,
            dfb_other,
            tile_freq,
            tile_start,
            dfb::batch_var,
            dfb::eps,
            dfb::den,
            dfb::weight,
            dfb::bias,
            dfb::temp_1,
            dfb::out,
            weight_has_value,
            bias_has_value);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles(
            dfb_bcast,
            dfb_other,
            remaining_iterations,
            tile_start,
            dfb::batch_var,
            dfb::eps,
            dfb::den,
            dfb::weight,
            dfb::bias,
            dfb::temp_1,
            dfb::out,
            weight_has_value,
            bias_has_value);
    }

    dfb_eps_obj.pop_front(onetile);
}
