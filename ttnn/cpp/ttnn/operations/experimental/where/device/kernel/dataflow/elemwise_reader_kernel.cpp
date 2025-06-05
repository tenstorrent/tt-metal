// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t cond_tensor_base_addr = get_arg_val<uint32_t>(0);
    uint32_t true_tensor_base_addr = get_arg_val<uint32_t>(1);
    uint32_t false_tensor_base_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t tile_ofs = get_arg_val<uint32_t>(4);

    constexpr bool is_cond_tensor_in_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool is_true_tensor_in_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool is_false_tensor_in_dram = get_compile_time_arg_val(2) == 1;

    constexpr uint32_t cb_cond = tt::CBIndex::c_0;
    constexpr uint32_t cb_true_values = tt::CBIndex::c_1;
    constexpr uint32_t cb_false_values = tt::CBIndex::c_2;

    const InterleavedAddrGenFast<is_cond_tensor_in_dram> cond_tensor_addr_gen = {
        .bank_base_address = cond_tensor_base_addr,
        .page_size = get_tile_size(cb_cond),
        .data_format = get_dataformat(cb_cond)};

    const InterleavedAddrGenFast<is_false_tensor_in_dram> true_tensor_addr_gen = {
        .bank_base_address = true_tensor_base_addr,
        .page_size = get_tile_size(cb_true_values),
        .data_format = get_dataformat(cb_true_values)};

    const InterleavedAddrGenFast<is_false_tensor_in_dram> false_tensor_addr_gen = {
        .bank_base_address = false_tensor_base_addr,
        .page_size = get_tile_size(cb_false_values),
        .data_format = get_dataformat(cb_false_values)};

    constexpr uint32_t tile_cnt = 1;
    for (uint32_t tile_id = tile_ofs; tile_id < tile_ofs + num_tiles; tile_id++) {
        cb_reserve_back(cb_cond, tile_cnt);
        noc_async_read_tile(tile_id, cond_tensor_addr_gen, get_write_ptr(cb_cond));

        cb_reserve_back(cb_true_values, tile_cnt);
        noc_async_read_tile(tile_id, true_tensor_addr_gen, get_write_ptr(cb_true_values));

        cb_reserve_back(cb_false_values, tile_cnt);
        noc_async_read_tile(tile_id, false_tensor_addr_gen, get_write_ptr(cb_false_values));

        noc_async_read_barrier();

        cb_push_back(cb_cond, tile_cnt);
        cb_push_back(cb_true_values, tile_cnt);
        cb_push_back(cb_false_values, tile_cnt);
    }
}
