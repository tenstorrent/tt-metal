// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point receiver writer: pop de-coalesced pages from cb_receiver_pages and write
// them to the output_tensor via a TensorAccessor (L1 CB -> DRAM/L1). Mirrors the proven
// reference unary interleaved writer.
//
// CB id for cb_receiver_pages comes from compile-time arg 0; output TensorAccessorArgs follow.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t page_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_receiver_pages = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t onetile = 1;

    // Third runtime arg page_size overrides TensorAccessorArgs::AlignedPageSize, which may be
    // stale on program-cache hits.
    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_receiver_pages, onetile);
        const uint32_t l1_read_addr = get_read_ptr(cb_receiver_pages);

        const uint64_t dst_noc_addr = s.get_noc_addr(i);
        noc_async_write(l1_read_addr, dst_noc_addr, s.get_aligned_page_size());
        noc_async_write_barrier();

        cb_pop_front(cb_receiver_pages, onetile);
    }
}
