// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file writer.cpp
 * @brief Writer kernel for all-gather operation.
 */

#include <cstdint>
#include "accessor/tensor_accessor.h"

/**
 * Read tiles from input tensor into circular buffer.
 * Grab tiles from circular buffer and write to output tensor.
 */
void kernel_main() {
    /**************** Compile-time args ****************/
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    auto args = TensorAccessorArgs<2>();

    /**************** Run-time args ****************/
    const uint32_t output_tensor_addr = get_arg_val<uint32_t>(0);

    // Use kernel arg to create TensorAccessor
    auto output_tensor_accessor = TensorAccessor(args, output_tensor_addr, page_size);

    // Copy all pages from circular buffer to output tensor
    for (const auto& page : output_tensor_accessor.pages(/* start_page_id */ 0, /* end_page_id */ 10)) {
        cb_wait_front(cb_id, /* num_pages */ 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write(l1_read_addr, page.noc_addr(), page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id, /* num_pages */ 1);
    }
}
