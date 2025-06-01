// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// Make n reads defined by num_reads
// Writes to Specified Circular Buffers in L1
// Expects n provided src_addr, src_noc_x, src_noc_y, and cb_id_in
void kernel_main() {
    const uint32_t num_pages = get_arg_val<uint32_t>(0);
    const uint32_t num_tensor_segments = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_in = get_compile_time_arg_val(0);
    constexpr bool rm_layout = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t ublock_size_pages = 1;

    // Read tensor metadata for only the tensors this core needs
    uint32_t arg_idx = 2;
    for (uint32_t seg_id = 0; seg_id < num_tensor_segments; ++seg_id) {
        uint32_t src_addr = get_arg_val<uint32_t>(arg_idx++);
        bool is_dram = get_arg_val<uint32_t>(arg_idx++) == 1;
        uint32_t start_page = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_pages_from_tensor = get_arg_val<uint32_t>(arg_idx++);
        uint32_t page_size = rm_layout ? get_arg_val<uint32_t>(arg_idx++) : get_tile_size(cb_id_in);

        // Create address generator for this tensor
        if (is_dram) {
            InterleavedAddrGen<true> addr_gen = {.bank_base_address = src_addr, .page_size = page_size};

            // Read pages from this tensor
            for (uint32_t page_offset = 0; page_offset < num_pages_from_tensor; ++page_offset) {
                cb_reserve_back(cb_id_in, ublock_size_pages);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in);
                noc_async_read_page(start_page + page_offset, addr_gen, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_id_in, ublock_size_pages);
            }
        } else {
            InterleavedAddrGen<false> addr_gen = {.bank_base_address = src_addr, .page_size = page_size};

            // Read pages from this tensor
            for (uint32_t page_offset = 0; page_offset < num_pages_from_tensor; ++page_offset) {
                cb_reserve_back(cb_id_in, ublock_size_pages);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in);
                noc_async_read_page(start_page + page_offset, addr_gen, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_id_in, ublock_size_pages);
            }
        }
    }
}
