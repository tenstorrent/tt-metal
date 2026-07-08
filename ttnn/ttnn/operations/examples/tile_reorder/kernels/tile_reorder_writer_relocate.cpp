// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tile_reorder writer (BRISC / NoC1) — RELOCATE method (the shuffle).
//
// Drains each whole tile from the CB and writes it to its output page in ONE
// 2 KB NoC write. Big, coalesced transactions -> high achieved DRAM bandwidth.
// Running on NoC1 while the reader reads on NoC0 lets the read and write streams
// overlap. This is what a tile-aware permute/transpose does: the tile just moves.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, dst_addr, page_bytes);

    for (uint32_t p = 0; p < num_pages; ++p) {
        const uint32_t out_idx = start_page + p;

        cb_wait_front(cb_id, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_id);
        noc_async_write(l1_read_addr, out_acc.get_noc_addr(out_idx), page_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id, 1);
    }
}
