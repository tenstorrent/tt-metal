// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — SEND program reader (NCRISC).
// Reads every page of the sender device's input shard from DRAM into cb_send_pages.
// Pure byte movement (format-agnostic): copies physical pages verbatim. The sender
// writer consumes these pages, coalesces them into fabric packets, and fabric-writes
// them to the receiver device's intermediate landing buffer.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_send_pages = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    // Third arg page_size overrides TensorAccessorArgs::AlignedPageSize, which may be
    // stale on program-cache hits.
    const uint32_t page_bytes = get_arg_val<uint32_t>(3);

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_send_pages, 1);
        const uint32_t l1_write_addr = get_write_ptr(cb_send_pages);
        const uint64_t src_noc_addr = s.get_noc_addr(i);
        noc_async_read(src_noc_addr, l1_write_addr, s.get_aligned_page_size());
        noc_async_read_barrier();
        cb_push_back(cb_send_pages, 1);
    }
}
