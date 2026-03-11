// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax Writer Kernel (shared for dim=-1 and dim=-2)
// Generic tile writer: drains output CB to DRAM using TensorAccessor

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(0);
    constexpr auto tensor_args = TensorAccessorArgs<1>();

    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    if (num_pages == 0) {
        return;
    }

    const uint32_t page_size = get_tile_size(output_cb_index);
    const auto accessor = TensorAccessor(tensor_args, dst_addr, page_size);

    uint32_t tile_id = start_id;
    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(output_cb_index, 1);
        uint32_t l1_read_addr = get_read_ptr(output_cb_index);
        noc_async_write_tile(tile_id, accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(output_cb_index, 1);
        ++tile_id;
    }
}
