// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — sender_reader (NCRISC), send program at sender_coord.
//
// Pure data movement (no compute). Streams the sender's input shard pages out of
// interleaved DRAM/L1 into cb_shard_pages, one page per push, for the sender_writer
// to coalesce into fabric packets. Pushes exactly `num_pages` — the writer pops the
// same count (CB sync balance).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_shard_pages = get_compile_time_arg_val(0);
    constexpr auto input_args = TensorAccessorArgs<1>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t page_size = get_arg_val<uint32_t>(2);

    // Third argument page_size overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program-cache hits.
    const auto input = TensorAccessor(input_args, input_addr, page_size);

    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_reserve_back(cb_shard_pages, 1);
        const uint32_t l1_write_addr = get_write_ptr(cb_shard_pages);
        noc_async_read(input.get_noc_addr(i), l1_write_addr, input.get_aligned_page_size());
        noc_async_read_barrier();
        cb_push_back(cb_shard_pages, 1);
    }
}
