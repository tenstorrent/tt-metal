// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
static constexpr int32_t MAX_NUM_DIMENSIONS = 8;

inline uint32_t get_output_grad_tile(
    uint32_t idx,
    uint32_t rank,
    uint32_t* output_grad_dim,
    uint32_t* output_grad_stride,
    uint32_t* input_grad_dim,
    uint32_t* input_grad_stride,
    bool* need_bcast_dim) {
    uint32_t cur_idx[MAX_NUM_DIMENSIONS];

    for (uint32_t i = 0; i < rank; ++i) {
        cur_idx[i] = (need_bcast_dim[i]) ? (0) : ((idx / input_grad_stride[i]) % input_grad_dim[i]);
    }

    uint32_t read_tile_id = 0;
    for (uint32_t i = 0; i < rank; ++i) {
        read_tile_id += (cur_idx[i] * output_grad_stride[i]);
    }

    return read_tile_id;
}

void kernel_main() {
    // compile-time args
    constexpr bool output_grad_is_dram = (get_compile_time_arg_val(0) == 1);
    constexpr uint32_t input_grad_rank = get_compile_time_arg_val(1);

    // runtime args
    ArgFetcher arg_fetcher;

    const auto output_grad_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto num_output_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto start_id = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto num_dim = arg_fetcher.get_next_arg_val<uint32_t>();

    uint32_t output_grad_dim[MAX_NUM_DIMENSIONS];
    for (uint32_t i = 0; i < input_grad_rank; ++i) {
        output_grad_dim[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }

    uint32_t input_grad_dim[MAX_NUM_DIMENSIONS];
    for (uint32_t i = 0; i < input_grad_rank; ++i) {
        input_grad_dim[i] = arg_fetcher.get_next_arg_val<uint32_t>();
    }

    bool need_bcast_dim[MAX_NUM_DIMENSIONS];
    for (uint32_t i = 0; i < input_grad_rank; ++i) {
        need_bcast_dim[i] = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);
    }

    uint32_t output_grad_stride[MAX_NUM_DIMENSIONS];
    output_grad_stride[0] = 1;
    for (uint32_t i = 1; i < input_grad_rank; ++i) {
        output_grad_stride[i] = output_grad_stride[i - 1] * output_grad_dim[i - 1];
    }

    uint32_t input_grad_stride[MAX_NUM_DIMENSIONS];
    input_grad_stride[0] = 1;
    for (uint32_t i = 1; i < input_grad_rank; ++i) {
        input_grad_stride[i] = input_grad_stride[i - 1] * input_grad_dim[i - 1];
    }

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_in1 = tt::CB::c_in1;
    constexpr uint32_t cb_id_in2 = tt::CB::c_in2;

    // zero tile
    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 0.0f;
    fill_cb_with_value(cb_id_in1, scaler.u);

    scaler.f = 1.0f / num_dim;
    fill_cb_with_value(cb_id_in2, scaler.u, 1);

    uint32_t l1_write_addr_in0;
    uint32_t output_grad_tile_bytes = get_tile_size(cb_id_in0);
    const auto output_grad_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<output_grad_is_dram> output_grad_addrg = {
        .bank_base_address = output_grad_addr,
        .page_size = output_grad_tile_bytes,
        .data_format = output_grad_data_format};

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        auto read_tile_id = get_output_grad_tile(
            i, input_grad_rank, output_grad_dim, output_grad_stride, input_grad_dim, input_grad_stride, need_bcast_dim);

        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(read_tile_id, output_grad_addrg, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
}
