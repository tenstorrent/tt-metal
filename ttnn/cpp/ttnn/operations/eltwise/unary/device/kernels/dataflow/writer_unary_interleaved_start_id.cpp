// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

#ifdef OUT_SHARDED
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    cb_wait_front(cb_id_out, num_pages);
#else

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

#ifdef BATCHED_CB_ACCESS
    constexpr uint32_t batch_size = 2;
#else
    constexpr uint32_t batch_size = 1;
#endif

#ifdef STRIDED_L1_ACCESS
    // Strided access: each core writes only to its local L1 bank.
    // Single-tile loop — local L1 writes have zero NOC latency so
    // batching overhead hurts more than it helps.
    // Args: {dst_addr, total_pages, bank_id, num_l1_banks}
    const uint32_t total_pages = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    for (uint32_t i = bank_id; i < total_pages; i += stride) {
        cb_wait_front(cb_id_out, 1);
        const auto l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_page(i, s, l1_read_addr);
        noc_async_writes_flushed();
        cb_pop_front(cb_id_out, 1);
    }
    noc_async_write_barrier();
#else
    // Contiguous access: each core writes a sequential range of pages.
    // Args: {dst_addr, num_pages, start_id}
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
        constexpr uint32_t onepage = 1;
        cb_wait_front(cb_id_out, onepage);
        const auto l1_read_addr = get_read_ptr(cb_id_out);
        noc_async_write_page(i, s, l1_read_addr);
        noc_async_writes_flushed();
        cb_pop_front(cb_id_out, onepage);
    }
    noc_async_write_barrier();
#else
    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; i += batch_size) {
        const uint32_t cur_batch = (end_id - i < batch_size) ? (end_id - i) : batch_size;
        cb_wait_front(cb_id_out, cur_batch);
        auto l1_read_addr = get_read_ptr(cb_id_out);
        for (uint32_t j = 0; j < cur_batch; ++j) {
            noc_async_write_page(i + j, s, l1_read_addr);
            l1_read_addr += page_bytes;
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_id_out, cur_batch);
    }
    noc_async_write_barrier();
#endif
#endif
#endif
}
