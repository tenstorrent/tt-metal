// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tile_reorder reader (NCRISC) — identical for both methods.
//
// For each OUTPUT tile in this core's range, compute the REMAPPED source tile
// (reverse the column-tile index within its row) and whole-page read it into
// CB_IN. Reversing the tile order is just index arithmetic — the transfer is
// one whole 2048 B page. This is the "the reorder itself is free" part; the
// methods differ only in what happens to the tile AFTER it lands in the CB.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(0);  // Wt
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr auto in_args = TensorAccessorArgs<2>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);

    for (uint32_t p = 0; p < num_pages; ++p) {
        const uint32_t out_idx = start_page + p;
        const uint32_t rt = out_idx / tiles_per_row;                              // row-tile
        const uint32_t ct = out_idx % tiles_per_row;                              // column-tile
        const uint32_t src_page = rt * tiles_per_row + (tiles_per_row - 1 - ct);  // reversed column

        cb_reserve_back(cb_in, 1);
        const uint32_t l1_write_addr = get_write_ptr(cb_in);
        noc_async_read(in_acc.get_noc_addr(src_page), l1_write_addr, page_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
