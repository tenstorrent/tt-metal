// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// Lightweight data-movement kernel that zeroes a page range of a DRAM tensor.
// Used to zero-pad KV cache regions for migration alignment.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// Fill a CB buffer with zeros via loopback NOC writes from the hardware MEM_ZEROS region.
FORCE_INLINE void fill_zero_buffer(uint32_t cb_id) {
    cb_reserve_back(cb_id, 1);
    uint32_t buf = get_write_ptr(cb_id);
    uint64_t buf_noc = get_noc_addr(NOC_X(my_x[0]), NOC_Y(my_y[0]), buf);
    for (uint32_t off = 0; off < NOC_MAX_BURST_SIZE; off += MEM_ZEROS_SIZE) {
        uint32_t chunk = ((uint32_t)MEM_ZEROS_SIZE < (NOC_MAX_BURST_SIZE - off)) ? (uint32_t)MEM_ZEROS_SIZE
                                                                                 : (NOC_MAX_BURST_SIZE - off);
        noc_async_write(MEM_ZEROS_BASE, buf_noc + off, chunk);
    }
    noc_async_write_barrier();
}

void kernel_main() {
    // ===== Compile-time args =====
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t cb_zero_buffer_id = get_compile_time_arg_val(1);

    // TensorAccessorArgs for the cache tensor (starting at index 2)
    constexpr auto output_args = TensorAccessorArgs<2>();

    // ===== Runtime args =====
    uint32_t rt_args_idx = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t page_start = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t page_end = get_arg_val<uint32_t>(rt_args_idx++);

    const auto output_addr_gen = TensorAccessor(output_args, output_addr, aligned_output_page_size);

    fill_zero_buffer(cb_zero_buffer_id);
    uint32_t zero_buffer_addr = get_write_ptr(cb_zero_buffer_id);

    // Write zeros to each page in the range using TensorAccessor-compatible API
    for (uint32_t page = page_start; page < page_end; page++) {
        noc_async_write_page(page, output_addr_gen, zero_buffer_addr);
    }
    noc_async_write_barrier();
}
