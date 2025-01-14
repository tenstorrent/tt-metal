// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/speculative_execution/speculative_sdpa_decode/device/kernels/dataflow/speculative_dataflow_common.hpp"

void kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t other_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t priority_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t other_priority_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_addr = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t is_dram = get_compile_time_arg_val(1);
    constexpr uint32_t priority_stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t cb_tensor = tt::CBIndex::c_0;
    constexpr uint32_t cb_priority = tt::CBIndex::c_1;

    constexpr uint32_t tile_bytes = get_tile_size(cb_tensor);
    constexpr DataFormat tile_data_format = get_dataformat(cb_tensor);
    const InterleavedAddrGenFast<is_dram> out_writer = {
        .bank_base_address = output_addr, .page_size = tile_bytes, .data_format = tile_data_format};

    auto [max_priority, max_other_priority] =
        get_max_priority<is_dram, 4 /* 4 bytes for float 32 */, B, cb_priority>(priority_addr, other_priority_addr);
    DPRINT << "max_priority: " << max_priority << ENDL();
    DPRINT << "max_other_priority: " << max_other_priority << ENDL();

    // copy input into tensor cb
    uint32_t input_src_base_addr = max_priority > max_other_priority ? input_addr : other_addr;
    const InterleavedAddrGenFast<is_dram> input_reader = {
        .bank_base_address = input_src_base_addr, .page_size = tile_bytes, .data_format = tile_data_format};

    cb_reserve_back(cb_tensor, input_num_tiles);
    uint32_t write_ptr = get_write_ptr(cb_tensor);
    for (uint32_t tile = 0; tile < input_num_tiles; ++tile) {
        noc_async_read_tile(tile, input_reader, write_ptr);
        write_ptr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_tensor, input_num_tiles);

    // copy cb into output
    cb_wait_front(cb_tensor, input_num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_tensor);
    for (uint32_t tile = 0; tile < input_num_tiles; ++tile) {
        noc_async_write_tile(tile, out_writer, l1_read_addr);
        l1_read_addr += tile_bytes;
    }
    cb_pop_front(cb_tensor, input_num_tiles);
    noc_async_writes_flushed();
}
