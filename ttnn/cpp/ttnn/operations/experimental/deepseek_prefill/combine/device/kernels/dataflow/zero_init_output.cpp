// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Lightweight data-movement kernel that zeroes a page range of an interleaved
// DRAM output tensor, then signals the combine reader cores via semaphore.
//
// Deployed on a wide core grid (worker cores minus sender cores) so that
// the DRAM zero-init work is distributed across many cores in parallel.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ===== Compile-time args =====
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(0);
    constexpr uint32_t num_sender_cores = get_compile_time_arg_val(1);
    constexpr uint32_t cb_zero_buffer_id = get_compile_time_arg_val(2);

    // TensorAccessorArgs for the output tensor (starting at index 3)
    constexpr auto output_args = TensorAccessorArgs<3>();

    // ===== Runtime args =====
    uint32_t rt_args_idx = 0;
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t page_start = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t page_end = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t zi_done_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);

    // The semaphore was created on all worker cores (including this one),
    // so get_semaphore gives the correct L1 offset for any core with this ID.
    uint32_t zi_done_sem_l1_offset = get_semaphore(zi_done_semaphore_id);

    // Read sender core NOC coordinates for semaphore signaling
    uint64_t sender_sem_noc_addrs[num_sender_cores];
    for (uint32_t c = 0; c < num_sender_cores; c++) {
        uint32_t noc_x = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t noc_y = get_arg_val<uint32_t>(rt_args_idx++);
        sender_sem_noc_addrs[c] = get_noc_addr(noc_x, noc_y, zi_done_sem_l1_offset);
    }

    const auto output_addr_gen = TensorAccessor(output_args, output_addr, aligned_output_page_size);

    // DMA-fill the zero buffer CB from the hardware MEM_ZEROS region
    cb_reserve_back(cb_zero_buffer_id, 1);
    uint32_t zero_buffer_addr = get_write_ptr(cb_zero_buffer_id);
    uint64_t zeros_noc_addr = get_noc_addr(NOC_X(my_x[0]), NOC_Y(my_y[0]), MEM_ZEROS_BASE);
    for (uint32_t offset = 0; offset < NOC_MAX_BURST_SIZE; offset += MEM_ZEROS_SIZE) {
        uint32_t chunk = ((uint32_t)MEM_ZEROS_SIZE < (NOC_MAX_BURST_SIZE - offset)) ? (uint32_t)MEM_ZEROS_SIZE
                                                                                    : (NOC_MAX_BURST_SIZE - offset);
        noc_async_read(zeros_noc_addr, zero_buffer_addr + offset, chunk);
    }
    noc_async_read_barrier();

    // Write zeros to each assigned page using NOC_MAX_BURST_SIZE chunks
    for (uint32_t page = page_start; page < page_end; page++) {
        uint64_t page_noc_addr = get_noc_addr(page, output_addr_gen);
        uint32_t remaining = aligned_output_page_size;
        uint64_t dst_addr = page_noc_addr;

        while (remaining > 0) {
            uint32_t chunk = (remaining > NOC_MAX_BURST_SIZE) ? (uint32_t)NOC_MAX_BURST_SIZE : remaining;
            noc_async_write(zero_buffer_addr, dst_addr, chunk);
            dst_addr += chunk;
            remaining -= chunk;
        }
    }

    noc_async_write_barrier();

    // Signal all sender/reader cores that zero-init is complete
    for (uint32_t c = 0; c < num_sender_cores; c++) {
        noc_semaphore_inc(sender_sem_noc_addrs[c], 1);
    }
}
