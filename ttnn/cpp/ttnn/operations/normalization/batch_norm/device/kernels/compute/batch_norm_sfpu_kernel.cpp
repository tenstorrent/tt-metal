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

// batchnorm_bcast_tiles: For each output tile in [tile_start, freq), computes batch-norm on tiles from dfb_other
// (input) broadcast against dfb_bcast (batch mean). First builds 1/sqrt(batch_var + eps) in dfb_den, then per tile:
// (input - mean) * den, optional multiply by weight, optional add bias. When NeedsOutputTypecast, SFPU typecasts
// from FP32 staging (dfb_output_0) to writer-facing dfb_output_final. Tracks last_srca_dfb in/out so
// copy_tile_to_dst_init_short_with_dt can reconfigure the SrcA unpacker correctly across mixed dtypes.

// Normalize packs to temp_1 when an affine (weight ∨ bias) still has to run; otherwise that
// tile is already the output.
constexpr auto dfb_affine_or_out = engaged_token_between(dfb::temp_1, dfb::out);
// Scale packs to temp_1 when bias still has to be applied; otherwise the scale (or the
// already-normalized tile) is already the output.
constexpr auto dfb_scaled_output = map_nullable_token(
    dfb::bias,
    [](DFBBindingToken const&) { return dfb::temp_1; },
    [&] { return dfb::out; });

template <bool NeedsOutputTypecast, uint32_t TcInFmt, uint32_t TcOutFmt, typename WeightTok, typename BiasTok, typename Tmp1Tok>
ALWI uint32_t batchnorm_bcast_tiles(
    uint32_t dfb_bcast,
    uint32_t dfb_other,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t dfb_batch_var,
    uint32_t dfb_eps,
    uint32_t dfb_den,
    uint32_t dfb_output_0,
    uint32_t dfb_output_final,
    WeightTok dfb_weight,
    BiasTok dfb_bias,
    Tmp1Tok dfb_tmp_1,
    uint32_t last_srca_dfb) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t index = 0;

    DataflowBuffer dfb_bcast_obj(dfb_bcast);          // batch_mean, broadcast against the input
    DataflowBuffer dfb_other_obj(dfb_other);          // input tiles
    DataflowBuffer dfb_batch_var_obj(dfb_batch_var);  // batch_var
    DataflowBuffer dfb_den_obj(dfb_den);              // 1/(sqrt(batch_var + eps))
    auto dfb_weight_obj = construct_nullable_dfb(dfb_weight);
    auto dfb_bias_obj = construct_nullable_dfb(dfb_bias);
    auto dfb_tmp_1_obj = construct_nullable_dfb(dfb_tmp_1);
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
    with_nullable_resource(dfb_weight_obj, [&](DataflowBuffer& dfb_weight_obj) { dfb_weight_obj.wait_front(onetile); });
    with_nullable_resource(dfb_bias_obj, [&](DataflowBuffer& dfb_bias_obj) { dfb_bias_obj.wait_front(onetile); });
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

        with_nullable_resource(dfb_weight_obj, [&](DataflowBuffer& dfb_weight_obj) {  // result = result * weight
            dfb_affine_or_out_obj.wait_front(onetile);
            dfb_scaled_output_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_affine_or_out_obj.get_id());
            last_srca_dfb = dfb_affine_or_out_obj.get_id();
            copy_tile(dfb_affine_or_out_obj.get_id(), index, index * 2);
            mul_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_weight_obj.get_id());
            last_srca_dfb = dfb_weight_obj.get_id();
            copy_tile(dfb_weight_obj.get_id(), index, index * 2 + 1);
            mul_binary_tile(index * 2, index * 2 + 1, index * 2);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(index * 2, dfb_scaled_output);
            tile_regs_release();

            dfb_scaled_output_obj.push_back(onetile);
            dfb_affine_or_out_obj.pop_front(onetile);
        });

        with_nullable_resource(dfb_bias_obj, [&](auto& dfb_bias_obj) {  // result = result + bias
            dfb_tmp_1_obj.wait_front(onetile);
            dfb_output_0_obj.reserve_back(onetile);

            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_tmp_1);
            last_srca_dfb = dfb_tmp_1_obj.get_id();
            copy_tile(dfb_tmp_1, index, index * 2);
            add_binary_tile_init();
            copy_tile_to_dst_init_short_with_dt(last_srca_dfb, dfb_bias_obj.get_id());
            last_srca_dfb = dfb_bias_obj.get_id();
            copy_tile(dfb_bias_obj.get_id(), index, index * 2 + 1);
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
    with_nullable_resource(dfb_weight_obj, [&](DataflowBuffer& dfb_weight_obj) { dfb_weight_obj.pop_front(onetile); });
    with_nullable_resource(dfb_bias_obj, [&](DataflowBuffer& dfb_bias_obj) { dfb_bias_obj.pop_front(onetile); });
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
            dfb::out,
            dfb_output_final,
            dfb::weight,
            dfb::bias,
            dfb::temp_1,
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
            dfb::out,
            dfb_output_final,
            dfb::weight,
            dfb::bias,
            dfb::temp_1,
            last_srca_dfb);
    }

    dfb_eps_obj.pop_front(onetile);
}
