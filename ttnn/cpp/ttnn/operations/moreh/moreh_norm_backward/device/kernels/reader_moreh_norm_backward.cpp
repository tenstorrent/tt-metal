// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

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
    // compile time args
    constexpr bool input_is_dram = (get_compile_time_arg_val(0) == 1);
    constexpr bool output_is_dram = (get_compile_time_arg_val(1) == 1);
    constexpr bool output_grad_is_dram = (get_compile_time_arg_val(2) == 1);
    constexpr uint32_t input_grad_rank = get_compile_time_arg_val(3);

    // runtime args
    ArgFetcher arg_fetcher;
    const auto input_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto output_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto output_grad_addr = arg_fetcher.get_next_arg_val<uint32_t>();

    const auto decimal = arg_fetcher.get_next_arg_val<uint32_t>();

    const auto num_output_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    const auto start_id = arg_fetcher.get_next_arg_val<uint32_t>();

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

    uint32_t cb_id{0};
    const auto cb_id_input = cb_id++;
    const auto cb_id_output = cb_id++;
    const auto cb_id_output_grad = cb_id++;
    const auto cb_id_decimal = cb_id++;

    // input
    const uint32_t input_tile_bytes = get_tile_size(cb_id_input);
    const auto input_data_format = get_dataformat(cb_id_input);

    const InterleavedAddrGenFast<input_is_dram> input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    // output
    const uint32_t output_tile_bytes = get_tile_size(cb_id_output);
    const auto output_data_format = get_dataformat(cb_id_output);

    const InterleavedAddrGenFast<output_is_dram> output_addrg = {
        .bank_base_address = output_addr, .page_size = output_tile_bytes, .data_format = output_data_format};

    // output_grad
    const uint32_t output_grad_tile_bytes = get_tile_size(cb_id_output_grad);
    const auto output_grad_data_format = get_dataformat(cb_id_output_grad);

    const InterleavedAddrGenFast<output_grad_is_dram> output_grad_addrg = {
        .bank_base_address = output_grad_addr,
        .page_size = output_grad_tile_bytes,
        .data_format = output_grad_data_format};

    fill_cb_with_value(cb_id_decimal, decimal);

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        uint32_t input_tile_id = i;
        auto read_tile_id = get_output_grad_tile(
            i, input_grad_rank, output_grad_dim, output_grad_stride, input_grad_dim, input_grad_stride, need_bcast_dim);

        cb_reserve_back(cb_id_input, 1);
        const auto input_l1_write_ptr = get_write_ptr(cb_id_input);
        noc_async_read_tile(input_tile_id, input_addrg, input_l1_write_ptr);
        noc_async_read_barrier();
        cb_push_back(cb_id_input, 1);

        cb_reserve_back(cb_id_output, 1);
        const auto output_l1_write_ptr = get_write_ptr(cb_id_output);
        noc_async_read_tile(read_tile_id, output_addrg, output_l1_write_ptr);
        noc_async_read_barrier();
        cb_push_back(cb_id_output, 1);

        cb_reserve_back(cb_id_output_grad, 1);
        const auto output_grad_l1_write_ptr = get_write_ptr(cb_id_output_grad);
        noc_async_read_tile(read_tile_id, output_grad_addrg, output_grad_l1_write_ptr);
        noc_async_read_barrier();
        cb_push_back(cb_id_output_grad, 1);
    }

}  // void kernel_main()
