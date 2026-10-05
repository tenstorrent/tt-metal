// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// bank_stagger example reader (NCRISC / NoC0).
//
// For each of this core's work units (one 32-row x chunk_wt-tile block) it reads
// the block's 32 row slices ("sticks") of a width-sharded ROW_MAJOR DRAM tensor
// into the reader->compute CB, then one barrier. With one shard per DRAM bank, a
// row is `pages_per_row` pages (one per shard) and every column block lives in
// exactly one bank.
//
// `stagger_blocks` (compile-time) is the ONLY difference between variants. Off,
// the core walks its units in order, so every core starts on the same column
// block -- the same bank -- and the grid moves from bank to bank together. On,
// it starts at unit `blk_rot` and wraps, so the cores start on different banks.
// Same reads, same sizes, same count, same L1 -- only the order moves. The
// writer walks the units in the same order, so the CB stays in sync.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t tile_rows = 32;
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(0);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(3);
    constexpr bool stagger_blocks = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t pages_per_row = get_compile_time_arg_val(5);
    constexpr auto in_args = TensorAccessorArgs<6>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t unit_start = get_arg_val<uint32_t>(1);
    const uint32_t unit_count = get_arg_val<uint32_t>(2);
    const uint32_t n_w = get_arg_val<uint32_t>(3);      // chunks per tile-row
    const uint32_t blk_rot = get_arg_val<uint32_t>(4);  // < unit_count; used only when stagger_blocks

    const auto in_acc = TensorAccessor(in_args, src_addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t j = 0; j < unit_count; ++j) {
            uint32_t off = j;
            if constexpr (stagger_blocks) {
                off = (j + blk_rot) < unit_count ? (j + blk_rot) : (j + blk_rot - unit_count);
            }
            const uint32_t u = unit_start + off;
            const uint32_t row0 = (u / n_w) * tile_rows;
            const uint32_t col_bytes = (u % n_w) * chunk_row_bytes;
            const uint32_t page_in_row = col_bytes / page_bytes;  // the shard, i.e. the bank
            const uint32_t byte_offset = col_bytes % page_bytes;

            cb_reserve_back(cb_in, chunk_wt);
            const uint32_t l1_addr = get_write_ptr(cb_in);
            for (uint32_t r = 0; r < tile_rows; ++r) {
                noc_async_read(
                    in_acc.get_noc_addr((row0 + r) * pages_per_row + page_in_row, byte_offset),
                    l1_addr + r * chunk_row_bytes,
                    chunk_row_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, chunk_wt);
        }
    }
}
