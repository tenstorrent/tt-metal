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
#include "experimental/kernel_args.h"

// Pack dest that must be a real DFB in every compile: the optional buffer if this program has it,
// otherwise a fallback that is always present.
constexpr DFBBindingToken dest(DFBBindingToken opt, DFBBindingToken) { return opt; }
constexpr DFBBindingToken dest(NullDFBBindingToken, DFBBindingToken fallback) { return fallback; }

// (input - mean) * den is staged in temp_1 only while weight or bias still has to be applied.
constexpr auto dfb_affine_or_out = dest(dfb::temp_1, dfb::out);

// batchnorm_bcast_tiles: For each output tile in [tile_start, freq), computes batch-norm on tiles from dfb_other
// (input) broadcast against dfb_bcast (batch mean). First builds 1/sqrt(batch_var + eps) in dfb_den, then per tile:
// (input - mean) * den, optional multiply by weight, optional add bias. When NeedsOutputTypecast, SFPU typecasts
// from FP32 staging (dfb_output_0) to writer-facing dfb_output_final. Tracks last_srca_dfb in/out so
// copy_tile_to_dst_init_short_with_dt can reconfigure the SrcA unpacker correctly across mixed dtypes.
template <bool NeedsOutputTypecast, uint32_t TcInFmt, uint32_t TcOutFmt, typename WeightTok, typename BiasTok>
ALWI uint32_t batchnorm_bcast_tiles(
    DFBBindingToken dfb_bcast,
    DFBBindingToken dfb_other,
    uint32_t freq,
    uint32_t tile_start,
    DFBBindingToken dfb_batch_var,
    DFBBindingToken dfb_eps,
    DFBBindingToken dfb_den,
    WeightTok dfb_weight,
    BiasTok dfb_bias,
    DFBBindingToken dfb_output_0,
    DFBBindingToken dfb_output_final,
    uint32_t last_srca_dfb) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t index = 0;

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

    // 1/(sqrt(batch_var + eps)) = dfb_den
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
    with_nullable_token(dfb_weight, [&](const DFBBindingToken& token) {
        DataflowBuffer w(token);
        w.wait_front(onetile);
    });
    with_nullable_token(dfb_bias, [&](const DFBBindingToken& token) {
        DataflowBuffer b(token);
        b.wait_front(onetile);
    });
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

        with_nullable_token(dfb_weight, [&](const DFBBindingToken& token) {  // result = result * weight
            dfb_affine_or_out_obj.wait_front(onetile);
            dfb_scaled_output_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_affine_or_out);
            last_srca_dfb = dfb_affine_or_out;
            copy_tile(dfb_affine_or_out, index, index * 2);
            mul_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, token);
            last_srca_dfb = token;
            copy_tile(token, index, index * 2 + 1);
            mul_binary_tile(index * 2, index * 2 + 1, index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(index * 2, dfb_scaled_output);
            tile_regs_release();

            dfb_scaled_output_obj.push_back(onetile);
            dfb_affine_or_out_obj.pop_front(onetile);
        });

        with_nullable_token(dfb_bias, [&](const DFBBindingToken& token) {  // result = result + bias
            dfb_tmp_1_obj.wait_front(onetile);
            dfb_output_0_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_affine_or_out);
            last_srca_dfb = dfb_affine_or_out;
            copy_tile(dfb_affine_or_out, index, index * 2);
            add_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, token);
            last_srca_dfb = token;
            copy_tile(token, index, index * 2 + 1);
            add_binary_tile(index * 2, index * 2 + 1, index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(index * 2, dfb_output_0);
            tile_regs_release();

            dfb_output_0_obj.push_back(onetile);
            dfb_tmp_1_obj.pop_front(onetile);
        });

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
    with_nullable_token(dfb_weight, [&](const DFBBindingToken& token) {
        DataflowBuffer w(token);
        w.pop_front(onetile);
    });
    with_nullable_token(dfb_bias, [&](const DFBBindingToken& token) {
        DataflowBuffer b(token);
        b.pop_front(onetile);
    });
    return last_srca_dfb;
}

// The writer-facing output DFB is only bound when the accumulation format is wider than the output
// dtype; on the other path the writer drains the compute output directly, so the same kernel-side
// handle has to name a different DFB. The alias is gated at the preprocessor stage because
// dfb::writer_out simply does not exist on the untypecast build.
#ifdef NEEDS_OUTPUT_TYPECAST
constexpr bool needs_output_typecast = true;
constexpr auto dfb_output_final = dfb::writer_out;
#else
constexpr bool needs_output_typecast = false;
constexpr auto dfb_output_final = dfb::out;
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);

    if (num_tiles == 0) {
        return;
    }

    constexpr uint32_t tc_in_fmt = get_arg(args::tc_in_fmt);
    constexpr uint32_t tc_out_fmt = get_arg(args::tc_out_fmt);

    // The batch mean is the broadcast operand of the subtraction; the input tiles are the other one.
    constexpr auto dfb_bcast = dfb::batch_mean;
    constexpr auto dfb_other = dfb::input;

    unary_op_init_common(dfb_other, dfb::out);
    uint32_t last_srca_dfb = dfb_other;

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    constexpr uint32_t onetile = 1;
    DataflowBuffer dfb_eps_obj(dfb::eps);  // one tile of eps, filled by the reader
    dfb_eps_obj.wait_front(onetile);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        last_srca_dfb = batchnorm_bcast_tiles<needs_output_typecast, tc_in_fmt, tc_out_fmt>(
            dfb_bcast,
            dfb_other,
            tile_freq,
            tile_start,
            dfb::batch_var,
            dfb::eps,
            dfb::den,
            dfb::weight,
            dfb::bias,
            dfb::out,
            dfb_output_final,
            last_srca_dfb);
    }
    if (remaining_iterations > 0) {
        last_srca_dfb = batchnorm_bcast_tiles<needs_output_typecast, tc_in_fmt, tc_out_fmt>(
            dfb_bcast,
            dfb_other,
            remaining_iterations,
            tile_start,
            dfb::batch_var,
            dfb::eps,
            dfb::den,
            dfb::weight,
            dfb::bias,
            dfb::out,
            dfb_output_final,
            last_srca_dfb);
    }

    dfb_eps_obj.pop_front(onetile);
}
