// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// point_to_point — receiver_writer (BRISC), receive program at receiver_coord.
//
// Pure data movement (no compute). Drains cb_shard_out (de-coalesced output shard
// pages produced by receiver_reader) and stores each page to the output shard's
// interleaved DRAM/L1 via TensorAccessor. Pops exactly `num_pages` — the
// receiver_reader pushes the same count (CB sync balance).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);
    const uint32_t page_size = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_shard_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // Third argument page_size overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program-cache hits.
    const auto s = TensorAccessor(dst_args, dst_addr, page_size);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_shard_out, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_shard_out);
        noc_async_write(l1_read_addr, s.get_noc_addr(i), s.get_aligned_page_size());
        noc_async_write_barrier();
        cb_pop_front(cb_shard_out, 1);
    }
}
