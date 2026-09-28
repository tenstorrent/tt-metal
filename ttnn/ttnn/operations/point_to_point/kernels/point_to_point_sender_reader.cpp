// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — SEND program, reader kernel (NCRISC).
//
// Pure interleaved DRAM/L1 -> L1 page streaming. Reads the sender device's
// input shard one page at a time into cb_input_pages, where sender_writer
// coalesces pages into fabric packets. No fabric, no compute here — this is
// the established reader_unary_interleaved_start_id_gen pattern.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    // Third TensorAccessor arg (page size) from runtime args overrides the
    // compile-time AlignedPageSize, which may be stale on program-cache hits.
    const uint32_t page_bytes = get_arg_val<uint32_t>(3);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_input_pages = 0;
    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_input_pages, onetile);
        const uint32_t l1_write_addr = get_write_ptr(cb_input_pages);

        const uint64_t src_noc_addr = s.get_noc_addr(i);
        noc_async_read(src_noc_addr, l1_write_addr, s.get_aligned_page_size());
        noc_async_read_barrier();

        cb_push_back(cb_input_pages, onetile);
    }
}
