// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — RECEIVE program, writer kernel (BRISC).
//
// Pure interleaved L1 -> DRAM/L1 page streaming. Drains the de-coalesced output
// pages from cb_output_pages and writes them to the receiver device's output
// shard. No fabric, no compute — the established
// writer_unary_interleaved_start_id_gen pattern. This overwrites the seeded
// output shard with the sender's data on the receiver device only.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    // Third TensorAccessor arg (page size) from runtime args overrides the
    // compile-time AlignedPageSize, which may be stale on program-cache hits.
    const uint32_t page_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_output_pages = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_output_pages, onetile);
        const uint32_t l1_read_addr = get_read_ptr(cb_output_pages);

        const uint64_t dst_noc_addr = s.get_noc_addr(i);
        noc_async_write(l1_read_addr, dst_noc_addr, s.get_aligned_page_size());
        noc_async_write_barrier();

        cb_pop_front(cb_output_pages, onetile);
    }
}
