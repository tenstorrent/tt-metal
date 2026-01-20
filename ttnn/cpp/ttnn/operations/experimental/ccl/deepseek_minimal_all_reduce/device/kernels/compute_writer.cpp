// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute writer kernel for non-data compute cores
// Writes compute output to output tensor via NOC

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);       // output CB
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);    // tiles this core processes
    constexpr uint32_t tile_offset = get_compile_time_arg_val(2);  // starting tile index
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t data_noc_x = get_compile_time_arg_val(4);
    constexpr uint32_t data_noc_y = get_compile_time_arg_val(5);

    size_t arg_idx = 0;
    const uint32_t output_tensor_addr = get_arg_val<uint32_t>(arg_idx++);

    // Calculate byte offset for this core's tile range
    const uint32_t byte_offset = tile_offset * page_size_bytes;
    const uint32_t total_bytes = num_tiles * page_size_bytes;

    // Wait for compute to finish
    cb_wait_front(cb_out, num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_out);

    // Write output to output tensor on data core
    uint64_t output_noc_addr = get_noc_addr(data_noc_x, data_noc_y, output_tensor_addr + byte_offset);
    noc_async_write(l1_read_addr, output_noc_addr, total_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb_out, num_tiles);
}
