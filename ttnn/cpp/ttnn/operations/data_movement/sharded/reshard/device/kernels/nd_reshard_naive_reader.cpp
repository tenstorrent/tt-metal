// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "accessor/tensor_accessor.h"

// Simple kernel that copies [start_page, end_page) pages from src to dst.
void kernel_main() {
    auto args_src = make_tensor_accessor_args<0, 0>();
    constexpr uint32_t base_idx_cta = args_src.compile_time_args_skip();
    constexpr uint32_t base_idx_crta = args_src.runtime_args_skip();

    constexpr uint32_t cb_id = get_compile_time_arg_val(base_idx_cta);
    constexpr uint32_t page_size = get_compile_time_arg_val(base_idx_cta + 1);

    const uint32_t bank_base_address_src = get_common_arg_val<uint32_t>(base_idx_crta);

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);

    auto accessor_src = make_tensor_accessor_from_args(args_src, bank_base_address_src, page_size);

    constexpr uint32_t one_tile = 1;
    for (uint32_t page_id = start_page; page_id < end_page; ++page_id) {
        cb_reserve_back(cb_id, one_tile);
        uint32_t cb_addr = get_write_ptr(cb_id);
        auto noc_addr_src = accessor_src.get_noc_addr(page_id);
        noc_async_read(noc_addr_src, cb_addr, page_size);
        noc_async_read_barrier();
        cb_push_back(cb_id, one_tile);
    }
}
