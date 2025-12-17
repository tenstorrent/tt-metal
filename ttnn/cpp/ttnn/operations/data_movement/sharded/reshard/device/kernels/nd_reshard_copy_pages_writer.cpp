// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"

// Simple kernel that copies [start_page, end_page) pages from dst to dst.
void kernel_main() {
    auto args_dst = TensorAccessorArgs<0, 0>();
    constexpr uint32_t base_idx_cta = args_dst.next_compile_time_args_offset();
    constexpr uint32_t base_idx_crta = args_dst.next_common_runtime_args_offset();

    constexpr uint32_t cb_id = get_compile_time_arg_val(base_idx_cta);
    constexpr uint32_t page_size = get_compile_time_arg_val(base_idx_cta + 1);

    const uint32_t bank_base_address_dst = get_common_arg_val<uint32_t>(base_idx_crta);

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);

    auto accessor_dst = TensorAccessor(args_dst, bank_base_address_dst, page_size);

    constexpr uint32_t one_tile = 1;
    uint32_t cb_addr = get_write_ptr(cb_id);
    auto pages = accessor_dst.pages(start_page, end_page);
    for (const auto& page : pages) {
        cb_wait_front(cb_id, one_tile);
        noc_async_write(cb_addr, page.noc_addr(), page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_id, one_tile);
    }
}
