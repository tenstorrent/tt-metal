// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id_in0 = 0;

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    constexpr uint32_t batch_size = 2;

#ifdef STRIDED_L1_ACCESS
    // Strided access: each core reads only from its local L1 bank.
    // Args: {src_addr, total_pages, bank_id, num_l1_banks}
    const uint32_t total_pages = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    for (uint32_t i = bank_id; i < total_pages; i += stride * batch_size) {
        const uint32_t remaining = (total_pages - i + stride - 1) / stride;
        const uint32_t cur_batch = remaining < batch_size ? remaining : batch_size;
        cb_reserve_back(cb_id_in0, cur_batch);
        auto l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t j = 0; j < cur_batch; ++j) {
            noc_async_read_page(i + j * stride, s, l1_write_addr);
            l1_write_addr += page_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, cur_batch);
    }
#else
    // Contiguous access: each core reads a sequential range of pages.
    // Args: {src_addr, num_pages, start_id}
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

// read a ublock of pages from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
        constexpr uint32_t onepage = 1;
        cb_reserve_back(cb_id_in0, onepage);
        const auto l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_page(i, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onepage);
    }
#else
    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; i += batch_size) {
        const uint32_t cur_batch = (end_id - i < batch_size) ? (end_id - i) : batch_size;
        cb_reserve_back(cb_id_in0, cur_batch);
        auto l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t j = 0; j < cur_batch; ++j) {
            noc_async_read_page(i + j, s, l1_write_addr);
            l1_write_addr += page_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, cur_batch);
    }
#endif
#endif
}
