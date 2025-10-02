// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"

void kernel_main() {
    /**************** Compile-time args ****************/
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    auto args = TensorAccessorArgs<2>();

    /**************** Run-time args ****************/
    const uint32_t input_tensor_addr = get_arg_val<uint32_t>(0);

    // Use kernel arg to create TensorAccessor
    auto input_tensor_accessor = TensorAccessor(args, input_tensor_addr, page_size);

    // Copy all pages from input tensor to circular buffer
    for (const auto& page : input_tensor_accessor.pages(/* start_page_id */ 0, /* end_page_id */ 10)) {
        cb_reserve_back(cb_id, /* num_pages */ 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id);
        noc_async_read(page.noc_addr(), l1_write_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(cb_id, /* num_pages */ 1);
    }
}
