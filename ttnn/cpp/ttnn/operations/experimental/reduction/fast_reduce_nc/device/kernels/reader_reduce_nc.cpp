// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

inline uint32_t get_read_tile_id(uint32_t output_tile_id, uint32_t reduce_tile_size, uint32_t inner_tile_size) {
    return ((output_tile_id / inner_tile_size) * reduce_tile_size) + (output_tile_id % inner_tile_size);
}

void kernel_main() {
    // compile-time args
    constexpr bool input_is_dram = (get_compile_time_arg_val(0) == 1);
    constexpr uint32_t input_granularity = get_compile_time_arg_val(1);

    // runtime args
    const auto input_addr = get_arg_val<uint32_t>(0);
    const auto num_input_tiles = get_arg_val<uint32_t>(1);
    const auto num_output_tiles = get_arg_val<uint32_t>(2);
    const auto start_id = get_arg_val<uint32_t>(3);
    const auto dim = get_arg_val<uint32_t>(4);
    const auto reduce_tile_size = get_arg_val<uint32_t>(5);
    const auto inner_tile_size = get_arg_val<uint32_t>(6);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t scaler = 0;

    generate_reduce_scaler(cb_id_in1, scaler);

    uint32_t l1_write_addr_in0;
    constexpr uint32_t input_tile_bytes = get_tile_size(cb_id_in0);
    const auto input_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<input_is_dram> input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};
    uint32_t input_granularity_index = 0;

    for (uint32_t i = start_id; i < start_id + num_output_tiles; i++) {
        auto read_tile_id = (dim == 0) ? (i) : (get_read_tile_id(i, reduce_tile_size, inner_tile_size));
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            if (input_granularity_index == 0) {
                cb_reserve_back(cb_id_in0, input_granularity);
                l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            }
            noc_async_read_tile(read_tile_id, input_addrg, l1_write_addr_in0);
            l1_write_addr_in0 += input_tile_bytes;  // correctness error
            read_tile_id += inner_tile_size;
            input_granularity_index++;
            if (input_granularity_index == input_granularity) {
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, input_granularity);
                input_granularity_index = 0;
            }
        }
    }
}
