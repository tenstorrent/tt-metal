// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/heterogeneous_data_structs.hpp"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_pages = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);

    // Debug: Print compile-time args
    DPRINT << "ROOT READER: cb_id=" << (uint32_t)cb_id << ", num_pages=" << (uint32_t)num_pages
           << ", page_size=" << (uint32_t)page_size << ENDL();

    // Runtime args - tensor accessor
    uint32_t rt_args_idx = 0;
    tt_l1_ptr uint32_t* tensor_addr = (tt_l1_ptr uint32_t*)(get_arg_val<uint32_t>(rt_args_idx++));

    // Debug: Print first few runtime args
    DPRINT << "ROOT READER: tensor_addr=" << (uint32_t)tensor_addr << ENDL();

    // Initialize tensor accessor
    const uint32_t src_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t start_id = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_id_in0 = 0;

    // Debug: Print runtime args
    DPRINT << "ROOT READER: src_addr=" << (uint32_t)src_addr << ", num_tiles=" << (uint32_t)num_tiles
           << ", start_id=" << (uint32_t)start_id << ENDL();

    const uint32_t page_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        const uint64_t src_noc_addr = get_noc_addr(i, s);
        noc_async_read(src_noc_addr, l1_write_addr, s.page_size);
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, 1);
    }
}
