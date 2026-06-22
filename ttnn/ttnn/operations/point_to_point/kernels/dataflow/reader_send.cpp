// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point sender reader: stream the input shard's pages from DRAM/L1 into
// cb_sender_pages (NoC read -> L1 CB). The writer_send kernel coalesces these pages
// into fabric packets. Mirrors the proven reference reader (interleaved start-id gen).
//
// CB id for cb_sender_pages is hardcoded to c_0 (matches the program descriptor); the
// only compile-time args are the input TensorAccessorArgs.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t page_bytes = get_arg_val<uint32_t>(3);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_sender_pages = 0;
    constexpr uint32_t onetile = 1;

    // Third runtime arg page_size overrides TensorAccessorArgs::AlignedPageSize, which may be
    // stale on program-cache hits.
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_sender_pages, onetile);
        const uint32_t l1_write_addr = get_write_ptr(cb_sender_pages);

        const uint64_t src_noc_addr = s.get_noc_addr(i);
        noc_async_read(src_noc_addr, l1_write_addr, s.get_aligned_page_size());
        noc_async_read_barrier();

        cb_push_back(cb_sender_pages, onetile);
    }
}
