// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/kernel/compute/dest_format_helpers.hpp"

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Pack dest that must be a real DFB in every compile: the optional buffer if this program has it,
// otherwise a fallback that is always present.
constexpr DFBBindingToken dest(DFBBindingToken opt, DFBBindingToken) { return opt; }
constexpr DFBBindingToken dest(NullDFBBindingToken, DFBBindingToken fallback) { return fallback; }

// (input - mean) * den is staged in temp_1 only while weight or bias still has to be applied.
constexpr auto dfb_affine_or_out = dest(dfb::temp_1, dfb::out);

template <typename WeightTok, typename BiasTok>
ALWI void batchnorm_bcast_tiles(
    DFBBindingToken dfb_bcast,
    DFBBindingToken dfb_other,
    uint32_t freq,
    uint32_t tile_start,
    DFBBindingToken dfb_batch_var,
    DFBBindingToken dfb_eps,
    DFBBindingToken dfb_den,
    WeightTok dfb_weight,
    BiasTok dfb_bias,
    DFBBindingToken dfb_output_0) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    // After *weight: stay on temp_1 if bias still follows, otherwise this is the last affine stage.
    DFBBindingToken dfb_scaled_output = dfb_output_0;
    with_nullable_token(dfb_bias, [&](const DFBBindingToken&) { dfb_scaled_output = dfb_affine_or_out; });

    DataflowBuffer dfb_bcast_obj(dfb_bcast);          // batch_mean, broadcast against the input
    DataflowBuffer dfb_other_obj(dfb_other);          // input tiles
    DataflowBuffer dfb_batch_var_obj(dfb_batch_var);  // batch_var
    DataflowBuffer dfb_den_obj(dfb_den);              // 1/(sqrt(batch_var + eps))
    DataflowBuffer dfb_tmp_1_obj(dfb_affine_or_out);  // (input - batch_mean)/(sqrt(batch_var + eps))
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
    with_nullable_token(dfb_weight, [&](const DFBBindingToken& token) {
        DataflowBuffer w(token);
        w.wait_front(onetile);
    });
    with_nullable_token(dfb_bias, [&](const DFBBindingToken& token) {
        DataflowBuffer b(token);
        b.wait_front(onetile);
    });
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
        with_nullable_token(dfb_weight, [&](const DFBBindingToken& token) {
            dfb_scaled_output_obj.reserve_back(onetile);
            dfb_affine_or_out_obj.wait_front(1);

            tile_regs_acquire();
            mul_tiles_init_with_dt(dfb_affine_or_out, token);
            mul_tiles(dfb_affine_or_out, token, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_scaled_output);
            tile_regs_release();

            dfb_affine_or_out_obj.pop_front(1);
            dfb_scaled_output_obj.push_back(onetile);
        });

        // result = result + bias
        with_nullable_token(dfb_bias, [&](const DFBBindingToken& token) {
            dfb_output_0_obj.reserve_back(onetile);
            dfb_tmp_1_obj.wait_front(onetile);

            tile_regs_acquire();
            add_tiles_init_with_dt(dfb_affine_or_out, token);
            add_tiles(dfb_affine_or_out, token, 0, 0, dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_output_0);
            tile_regs_release();

            dfb_tmp_1_obj.pop_front(onetile);
            dfb_output_0_obj.push_back(onetile);
        });
    }
    dfb_bcast_obj.pop_front(onetile);
    dfb_den_obj.pop_front(onetile);
    with_nullable_token(dfb_weight, [&](const DFBBindingToken& token) {
        DataflowBuffer w(token);
        w.pop_front(onetile);
    });
    with_nullable_token(dfb_bias, [&](const DFBBindingToken& token) {
        DataflowBuffer b(token);
        b.pop_front(onetile);
    });
}

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);

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
            dfb::out);
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
            dfb::out);
    }

    dfb_eps_obj.pop_front(onetile);
}
