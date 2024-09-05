// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

inline uint32_t get_read_tile_id(uint32_t tile_id, uint32_t dim, uint32_t CHtWt, uint32_t HtWt) {
    return (dim == 0) ? (tile_id) : (tile_id / HtWt * CHtWt) + (tile_id % HtWt);
}

void kernel_main() {
    const auto input_addr = get_arg_val<uint32_t>(0);
    const auto num_tiles_to_cumsum = get_arg_val<uint32_t>(1);
    const auto num_output_tiles_per_core = get_arg_val<uint32_t>(2);
    const auto input_tile_offset = get_arg_val<uint32_t>(3);
    const auto start_id = get_arg_val<uint32_t>(4);
    const auto input_is_dram = (get_arg_val<uint32_t>(5) == 1);
    const auto HtWt = get_arg_val<uint32_t>(6);
    const auto CHtWt = get_arg_val<uint32_t>(7);
    const auto dim = get_arg_val<uint32_t>(8);
    const auto flip = (get_arg_val<uint32_t>(9) == 1);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 0.0f;
    fill_cb_with_value(cb_id_in1, scaler.u);

    uint32_t l1_write_addr_in0;
    uint32_t input_tile_bytes = get_tile_size(cb_id_in0);
    const auto input_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<true> dram_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};
    const InterleavedAddrGenFast<false> l1_input_addrg = {
        .bank_base_address = input_addr, .page_size = input_tile_bytes, .data_format = input_data_format};

    for (uint32_t i = start_id; i < start_id + num_output_tiles_per_core; i++) {
        auto offset = input_tile_offset;
        auto read_tile_id = get_read_tile_id(i, dim, CHtWt, HtWt);
        if (flip) {
            read_tile_id += (num_tiles_to_cumsum - 1) * offset;
            offset *= -1;
        }

        for (uint32_t j = 0; j < num_tiles_to_cumsum; ++j) {
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            if (input_is_dram) {
                noc_async_read_tile(read_tile_id, dram_input_addrg, l1_write_addr_in0);
            } else {
                noc_async_read_tile(read_tile_id, l1_input_addrg, l1_write_addr_in0);
            }
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            read_tile_id += offset;
        }
    }
}
