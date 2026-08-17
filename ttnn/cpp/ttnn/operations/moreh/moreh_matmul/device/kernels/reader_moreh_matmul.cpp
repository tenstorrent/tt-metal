// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

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
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    bool transpose_input = (get_compile_time_arg_val(1) == 1);
    bool transpose_other = (get_compile_time_arg_val(2) == 1);
    uint32_t input_mask_h = get_compile_time_arg_val(3);
    uint32_t input_mask_w = get_compile_time_arg_val(4);
    uint32_t other_mask_h = get_compile_time_arg_val(5);
    uint32_t other_mask_w = get_compile_time_arg_val(6);
    constexpr auto input_args = TensorAccessorArgs<7>();
    constexpr auto other_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
#ifdef FUSE_BIAS
    constexpr bool is_scalar_bias = (get_compile_time_arg_val(other_args.next_compile_time_args_offset()) == 1);
    constexpr auto bias_args = TensorAccessorArgs<other_args.next_compile_time_args_offset() + 1>();
#endif

    // runtime args
    ArgFetcher arg_fetcher;
    uint32_t input_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    uint32_t other_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    uint32_t output_tile_start_idx = arg_fetcher.get_next_arg_val<uint32_t>();
    uint32_t num_output_tiles = arg_fetcher.get_next_arg_val<uint32_t>();

    uint32_t input_stride[MAX_NUM_DIMENSIONS];
    uint32_t other_stride[MAX_NUM_DIMENSIONS];
    uint32_t output_stride[MAX_NUM_DIMENSIONS];
    uint32_t input_not_bcast[MAX_NUM_DIMENSIONS];
    uint32_t other_not_bcast[MAX_NUM_DIMENSIONS];

    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        input_stride[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        other_stride[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        output_stride[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        input_not_bcast[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS; ++i) {
        other_not_bcast[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }

#ifdef FUSE_BIAS
    uint32_t bias_addr = arg_fetcher.get_next_arg_val<uint32_t>();
#endif

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;
    constexpr uint32_t cb_id_in3 = 3;
    constexpr uint32_t cb_id_in4 = 4;
    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(input_args, input_addr);
    const auto s1 = TensorAccessor(other_args, other_addr);

#ifdef FUSE_BIAS
    const auto s_bias = TensorAccessor(bias_args, bias_addr);
#endif

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
    const auto in0_tile_bytes = get_tile_size(cb_id_in0);
    const auto in1_tile_bytes = get_tile_size(cb_id_in1);
#ifdef FUSE_BIAS
    DataflowBuffer dfb_in4(cb_id_in4);
    const auto in4_tile_bytes = get_tile_size(cb_id_in4);
#endif

#ifdef FUSE_BIAS
    if (is_scalar_bias && num_output_tiles > 0) {
        dfb_in4.reserve_back(onetile);
        noc.async_read(s_bias, dfb_in4, in4_tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in4.push_back(onetile);
    }
#endif

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
#ifdef FUSE_BIAS
        if constexpr (!is_scalar_bias) {
            uint32_t bias_tidx = output_idxes[0];
            dfb_in4.reserve_back(onetile);
            noc.async_read(s_bias, dfb_in4, in4_tile_bytes, {.page_id = bias_tidx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in4.push_back(onetile);
        }
#endif

        output_tidx++;
    }
}
