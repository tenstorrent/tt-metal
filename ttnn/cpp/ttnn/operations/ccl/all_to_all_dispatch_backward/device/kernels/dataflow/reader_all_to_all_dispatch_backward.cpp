// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for all_to_all_dispatch backward.
//
// Reads grad_output pages sequentially for the assigned page range.
// Each page is one (b_global, s) entry in the grad tensor.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t data_cb_id           = get_compile_time_arg_val(0);
    constexpr uint32_t grad_page_size_bytes = get_compile_time_arg_val(1);
    constexpr auto grad_args = TensorAccessorArgs<2>();

    // ---- Runtime args ----
    const auto grad_tensor_addr = get_arg_val<uint32_t>(0);
    const auto page_start       = get_arg_val<uint32_t>(1);
    const auto page_end         = get_arg_val<uint32_t>(2);

    const auto grad_addrgen = TensorAccessor(grad_args, grad_tensor_addr, grad_page_size_bytes);

    for (uint32_t page_idx = page_start; page_idx < page_end; ++page_idx) {
        cb_reserve_back(data_cb_id, 1);
        const uint32_t grad_l1_addr = get_write_ptr(data_cb_id);
        const uint64_t grad_noc_addr = get_noc_addr(page_idx, grad_addrgen);
        noc_async_read(grad_noc_addr, grad_l1_addr, grad_page_size_bytes);
        noc_async_read_barrier();
        cb_push_back(data_cb_id, 1);
    }
}
