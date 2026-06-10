// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Lightweight data-movement kernel that zeroes a page range of a DRAM tensor.
// Used to zero-pad KV cache regions for migration alignment.
//

#include <cstdint>
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ===== Compile-time args =====
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t cb_zero_buffer_id = get_compile_time_arg_val(1);

    // TensorAccessorArgs for the cache tensor (starting at index 2)
    constexpr auto output_args = TensorAccessorArgs<2>();

    // ===== Runtime args =====
    uint32_t rt_args_idx = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t page_start = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t page_end = get_arg_val<uint32_t>(rt_args_idx++);

    const auto output_addr_gen = TensorAccessor(output_args, output_addr);

    // Pre-zero a NOC_MAX_BURST_SIZE L1 scratch once (overload 1), then write it to each
    // DRAM page (overload 2). No MEM_ZEROS_BASE dependency.
    Noc noc;
    CircularBuffer scratch_cb(cb_zero_buffer_id);
    noc.async_write_zeros(scratch_cb, NOC_MAX_BURST_SIZE);
    noc.write_zeros_l1_barrier();

    for (uint32_t page = page_start; page < page_end; ++page) {
        noc.async_write_zeros(output_addr_gen, page_size, {.page_id = page}, scratch_cb);
    }
    noc.write_zeros_dram_barrier();
}
