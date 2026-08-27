// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

static constexpr int32_t MAX_NUM_DIMENSIONS = 8;

inline uint32_t get_tidx(
    uint32_t* output_idxes, uint32_t* stride, uint32_t* not_bcast, bool transpose, bool use_h_dim) {
    uint32_t tidx = 0;
    // batch dim
    for (int32_t i = MAX_NUM_DIMENSIONS - 1; i >= 2; --i) {
        tidx += not_bcast[i] * stride[i] * output_idxes[i];
    }

    // last 2-dim
    int32_t i = transpose ? (use_h_dim ? 0 : 1) : (use_h_dim ? 1 : 0);
    tidx += not_bcast[i] * stride[i] * output_idxes[use_h_dim ? 1 : 0];
    return tidx;
}

inline void unravel_output_tidx(uint32_t output_tidx, uint32_t* output_idxes, uint32_t* output_stride) {
    for (int32_t i = MAX_NUM_DIMENSIONS - 1; i >= 0; --i) {
        uint32_t dim = output_tidx / output_stride[i];
        output_idxes[i] = dim;
        output_tidx -= (output_idxes[i] * output_stride[i]);
    }
}

void kernel_main() {
    // compile-time args
    constexpr uint32_t Kt = get_arg(args::Kt);
    bool transpose_input = (get_arg(args::transpose_input) == 1);
    bool transpose_other = (get_arg(args::transpose_other) == 1);
    uint32_t input_mask_h = get_arg(args::input_mask_h);
    uint32_t input_mask_w = get_arg(args::input_mask_w);
    uint32_t other_mask_h = get_arg(args::other_mask_h);
    uint32_t other_mask_w = get_arg(args::other_mask_w);
    constexpr bool is_scalar_bias = map_nullable_token(
        dfb::in4,
        [](DFBBindingToken const&) { return get_arg(args::is_scalar_bias) == 1; },
        [] { return false; });

    // runtime args (named scalars; the input/other/bias base addresses are now supplied by the
    // tensor bindings, so their address RTAs are gone)
    uint32_t output_tile_start_idx = get_arg(args::output_tile_start_idx);
    uint32_t num_output_tiles = get_arg(args::num_output_tiles);

    // The five dimensional arrays are homogeneous, index-addressed collections -> runtime varargs.
    uint32_t input_stride[MAX_NUM_DIMENSIONS];
    uint32_t other_stride[MAX_NUM_DIMENSIONS];
    uint32_t output_stride[MAX_NUM_DIMENSIONS];
    uint32_t input_not_bcast[MAX_NUM_DIMENSIONS];
    uint32_t other_not_bcast[MAX_NUM_DIMENSIONS];

    uint32_t vararg_idx = 0;
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        input_stride[i] = get_vararg(vararg_idx++);
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        other_stride[i] = get_vararg(vararg_idx++);
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        output_stride[i] = get_vararg(vararg_idx++);
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        input_not_bcast[i] = get_vararg(vararg_idx++);
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        other_not_bcast[i] = get_vararg(vararg_idx++);
    }

    constexpr uint32_t cb_id_in0 = dfb::in0;
    constexpr uint32_t cb_id_in1 = dfb::in1;
    constexpr uint32_t cb_id_in2 = dfb::in2;
    constexpr uint32_t cb_id_in3 = dfb::in3;
    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(tensor::input);
    const auto s1 = TensorAccessor(tensor::other);
    auto s_bias = construct_nullable_tensor(tensor::bias);

    // mask
    bool need_input_mask_h = (input_mask_h != 32);
    bool need_input_mask_w = (input_mask_w != 32);

    if (need_input_mask_h || need_input_mask_w) {
        DataflowBuffer dfb_in2(cb_id_in2);
        generate_mask_tiles(dfb_in2, input_mask_h, input_mask_w);
    }

    bool need_other_mask_h = (other_mask_h != 32);
    bool need_other_mask_w = (other_mask_w != 32);
    if (need_other_mask_h || need_other_mask_w) {
        DataflowBuffer dfb_in3(cb_id_in3);
        generate_mask_tiles(dfb_in3, other_mask_h, other_mask_w);
    }

    uint32_t output_tidx = output_tile_start_idx;
    uint32_t input_step_count = (transpose_input) ? (input_stride[1]) : (input_stride[0]);
    uint32_t other_step_count = (transpose_other) ? (other_stride[0]) : (other_stride[1]);

    Noc noc;
    DataflowBuffer dfb_in0(cb_id_in0);
    DataflowBuffer dfb_in1(cb_id_in1);
    const auto in0_tile_bytes = dfb_in0.get_tile_size();
    const auto in1_tile_bytes = dfb_in1.get_tile_size();
    auto dfb_in4 = construct_nullable_dfb(dfb::in4);
    uint32_t in4_tile_bytes = 0;
    with_nullable_dfb(dfb_in4, [&](DataflowBuffer& dfb_in4) {
        in4_tile_bytes = dfb_in4.get_tile_size();
        if (is_scalar_bias && num_output_tiles > 0) {
            dfb_in4.reserve_back(onetile);
            noc.async_read(s_bias, dfb_in4, in4_tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in4.push_back(onetile);
        }
    });

    for (uint32_t n = 0; n < num_output_tiles; n++) {
        uint32_t output_idxes[MAX_NUM_DIMENSIONS];
        unravel_output_tidx(output_tidx, output_idxes, output_stride);
        uint32_t input_tidx = get_tidx(output_idxes, input_stride, input_not_bcast, transpose_input, true);
        uint32_t other_tidx = get_tidx(output_idxes, other_stride, other_not_bcast, transpose_other, false);

        for (uint32_t kt = 0; kt < Kt; kt++) {
            dfb_in0.reserve_back(onetile);
            dfb_in1.reserve_back(onetile);

            noc.async_read(s0, dfb_in0, in0_tile_bytes, {.page_id = input_tidx}, {.offset_bytes = 0});
            noc.async_read(s1, dfb_in1, in1_tile_bytes, {.page_id = other_tidx}, {.offset_bytes = 0});
            noc.async_read_barrier();

            dfb_in0.push_back(onetile);
            dfb_in1.push_back(onetile);

            input_tidx += input_step_count;
            other_tidx += other_step_count;
        }
        with_nullable_token(tensor::bias, dfb::in4, [&](auto const&, DFBBindingToken const&) {
            if constexpr (!is_scalar_bias) {
                uint32_t bias_tidx = output_idxes[0];
                dfb_in4.reserve_back(onetile);
                noc.async_read(s_bias, dfb_in4, in4_tile_bytes, {.page_id = bias_tidx}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_in4.push_back(onetile);
            }
        });

        output_tidx++;
    }
}
