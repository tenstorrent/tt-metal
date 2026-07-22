// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

// Simple kernel that copies [start_page, end_page) pages from src to dst.
void kernel_main() {
    auto args_src = TensorAccessorArgs<0, 0>();
    constexpr uint32_t base_idx_cta = args_src.next_compile_time_args_offset();
    constexpr uint32_t base_idx_crta = args_src.next_common_runtime_args_offset();

    constexpr uint32_t dfb_id = get_compile_time_arg_val(base_idx_cta);
    constexpr uint32_t page_size = get_compile_time_arg_val(base_idx_cta + 1);

    const uint32_t bank_base_address_src = get_common_arg_val<uint32_t>(base_idx_crta);

    const uint32_t start_page = get_arg_val<uint32_t>(0);
    const uint32_t end_page = get_arg_val<uint32_t>(1);

    auto accessor_src = TensorAccessor(args_src, bank_base_address_src);

    Noc noc;
    DataflowBuffer dfb(dfb_id);

    constexpr uint32_t one_tile = 1;
    auto pages = accessor_src.pages(start_page, end_page);
    for (const auto& page : pages) {
        dfb.reserve_back(one_tile);
        noc.async_read(
            accessor_src, dfb, page_size, {.page_id = page.page_id(), .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb.push_back(one_tile);
    }
}
