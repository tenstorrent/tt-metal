// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

static constexpr int32_t MAX_NUM_DIMENSIONS = 8;

inline uint32_t get_tidx(uint32_t* output_idxes, uint32_t* stride, uint32_t* not_bcast, bool transpose, bool use_h_dim) {
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

inline void unravel_output_tidx(uint32_t output_tidx, uint32_t* output_idxes,  uint32_t* output_stride) {
    for (int32_t i = MAX_NUM_DIMENSIONS - 1; i >= 0;--i) {
        uint32_t dim = output_tidx / output_stride[i];
        output_idxes[i] = dim;
        output_tidx -= (output_idxes[i] * output_stride[i]);
    }
}

void kernel_main() {
    // compile-time args
    constexpr bool input_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool other_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    bool transpose_input = (get_compile_time_arg_val(3) == 1);
    bool transpose_other = (get_compile_time_arg_val(4) == 1);
    uint32_t input_mask_h = get_compile_time_arg_val(5);
    uint32_t input_mask_w = get_compile_time_arg_val(6);
    uint32_t other_mask_h = get_compile_time_arg_val(7);
    uint32_t other_mask_w = get_compile_time_arg_val(8);
    #ifdef FUSE_BIAS
    constexpr bool bias_is_dram = (get_compile_time_arg_val(9) == 1);
    bool is_scalar_bias = (get_compile_time_arg_val(10) == 1);
    bool scalar_bias_loaded = false;
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

    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS;++i) {
        input_stride[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS;++i) {
        other_stride[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS;++i) {
        output_stride[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS;++i) {
        input_not_bcast[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }
    for (int32_t i = 0; i < MAX_NUM_DIMENSIONS;++i) {
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

    const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat in0_data_format = get_dataformat(cb_id_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat in1_data_format = get_dataformat(cb_id_in1);

    const InterleavedAddrGenFast<input_is_dram> s0 = {
        .bank_base_address = input_addr, .page_size = in0_tile_bytes, .data_format = in0_data_format};

    const InterleavedAddrGenFast<other_is_dram> s1 = {
        .bank_base_address = other_addr, .page_size = in1_tile_bytes, .data_format = in1_data_format};

    #ifdef FUSE_BIAS
    const uint32_t in4_tile_bytes = get_tile_size(cb_id_in4);
    const DataFormat in4_data_format = get_dataformat(cb_id_in4);
    const InterleavedAddrGenFast<bias_is_dram> s_bias = {
        .bank_base_address = bias_addr, .page_size = in4_tile_bytes, .data_format = in4_data_format};
    #endif

    // mask
    bool need_input_mask_h = (input_mask_h != 32);
    bool need_input_mask_w = (input_mask_w != 32);

    if (need_input_mask_h || need_input_mask_w) {
        generate_mask_tiles(cb_id_in2, input_mask_h, input_mask_w);
    }

    bool need_other_mask_h = (other_mask_h != 32);
    bool need_other_mask_w = (other_mask_w != 32);
    if (need_other_mask_h || need_other_mask_w) {
        generate_mask_tiles(cb_id_in3, other_mask_h, other_mask_w);
    }

    uint32_t output_tidx = output_tile_start_idx;
    uint32_t input_step_count = (transpose_input) ? (input_stride[1]) : (input_stride[0]);
    uint32_t other_step_count = (transpose_other) ? (other_stride[0]) : (other_stride[1]);

    for (uint32_t n = 0; n < num_output_tiles; n++) {
        uint32_t output_idxes[MAX_NUM_DIMENSIONS];
        unravel_output_tidx(output_tidx, output_idxes, output_stride);
        uint32_t input_tidx = get_tidx(output_idxes, input_stride, input_not_bcast, transpose_input, true);
        uint32_t other_tidx = get_tidx(output_idxes, other_stride, other_not_bcast, transpose_other, false);

        for (uint32_t kt = 0; kt < Kt; kt++) {
            // read input, other tile
            cb_reserve_back(cb_id_in0, onetile);
            cb_reserve_back(cb_id_in1, onetile);

            uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(input_tidx, s0, l1_write_addr_in0);

            uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
            noc_async_read_tile(other_tidx, s1, l1_write_addr_in1);
            noc_async_read_barrier();

            cb_push_back(cb_id_in0, onetile);
            cb_push_back(cb_id_in1, onetile);

            input_tidx += input_step_count;
            other_tidx += other_step_count;
        }
        #ifdef FUSE_BIAS
            if (!is_scalar_bias) {
                uint32_t bias_tidx = output_idxes[0];
                cb_reserve_back(cb_id_in4, onetile);
                uint32_t l1_write_addr_in4 = get_write_ptr(cb_id_in4);
                noc_async_read_tile(bias_tidx, s_bias, l1_write_addr_in4);
                noc_async_read_barrier();
                cb_push_back(cb_id_in4, onetile);
            } else {
                if (!scalar_bias_loaded) {
                    cb_reserve_back(cb_id_in4, onetile);
                    uint32_t l1_write_addr_in4 = get_write_ptr(cb_id_in4);
                    noc_async_read_tile(0, s_bias, l1_write_addr_in4);
                    noc_async_read_barrier();
                    cb_push_back(cb_id_in4, onetile);
                    scalar_bias_loaded = true;
                }
            }
        #endif


        output_tidx++;
    }
}
