// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// This repeat kernel is forked off the concat kernel
// so this kernel currently does one read per tile in output
// but since this is the same tensor and not unique tensors like concat we can
// reduce reads by reusing the tiles we have read instead of constantly
// rereading them

// Make n reads defined by num_reads
// Writes to Specified Circular Buffers in L1
// Expects n provided src_addr, src_noc_x, src_noc_y, and cb_id_in
void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t num_pages_per_block = get_arg_val<uint32_t>(2);
    uint32_t curr_repeat_idx = get_arg_val<uint32_t>(3);
    uint32_t curr_idx_in_block = get_arg_val<uint32_t>(4);
    uint32_t curr_block_start_id = get_arg_val<uint32_t>(5);
    uint32_t curr_id = get_arg_val<uint32_t>(6);
    const uint32_t page_size = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr uint32_t src_is_dram = get_compile_time_arg_val(1);
    constexpr uint32_t num_repeats = get_compile_time_arg_val(2);

    constexpr uint32_t ublock_size_pages = 1;

    InterleavedAddrGen<src_is_dram> src_addr_gen = {.bank_base_address = src_addr, .page_size = page_size};

    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_reserve_back(cb_id_in, ublock_size_pages);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in);
#ifdef WIDTH_REPEAT
        noc_async_read_page(curr_id, src_addr_gen, l1_write_addr);
        uint64_t local_read_addr = get_noc_addr(l1_write_addr);
        l1_write_addr += page_size;
        noc_async_read_barrier();
        for (uint32_t j = 1; j < num_repeats; ++j) {
            noc_async_read(local_read_addr, l1_write_addr, page_size);
            l1_write_addr += page_size;
        }
        curr_id++;
        noc_async_read_barrier();
#else
        noc_async_read_page(curr_id, src_addr_gen, l1_write_addr);
        curr_id++;
        curr_idx_in_block++;

        if (curr_idx_in_block == num_pages_per_block) {
            curr_idx_in_block = 0;
            curr_repeat_idx++;
            if (curr_repeat_idx == num_repeats) {
                curr_repeat_idx = 0;
                curr_block_start_id = curr_id;
            } else {
                curr_id = curr_block_start_id;
            }
        }
        noc_async_read_barrier();
#endif
        cb_push_back(cb_id_in, ublock_size_pages);
    }
}
