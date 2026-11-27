// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer (BRISC, NoC1): store_row_block, and with gather_riscs == 2 also the second share of
// gather_row_block on NoC1, issued into the slot the reader is filling (the front slot this RISC
// consumes next; it popped that slot's previous block itself, so the slot is free).
// The store helper waits/pops block_tile_cols pages per <= tile_h-row block and write-barriers each
// block, so no write is in flight when the kernel returns. One helper call per row block lets this
// RISC's gather share interleave with the stores.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"
#include "g6_untilize_gather.hpp"

void kernel_main() {
    using A = g6_untilize::GatherArgs;
    constexpr auto input_args = TensorAccessorArgs<A::next_ct_arg>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr uint32_t block_elems = A::block_tile_cols * A::tile_w;

    const g6_untilize::CoreRows mine = g6_untilize::core_rows();
    const uint32_t first_c = g6_untilize::first_tile_col(mine);
    const auto output = TensorAccessor(output_args, get_common_arg_val<uint32_t>(0));
    [[maybe_unused]] const auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(1));

    for (uint32_t col_block_idx = 0; col_block_idx < A::num_col_blocks; ++col_block_idx) {
        const uint32_t col_elem0 = col_block_idx * block_elems;
        // The last chunk may end inside a partial tile; it still spans block_tile_cols CB pages.
        const uint32_t chunk_elems = (A::row_elems - col_elem0 < block_elems) ? A::row_elems - col_elem0 : block_elems;
        for (uint32_t block_row = 0; block_row < mine.count; block_row += A::tile_h) {
            const uint32_t block_rows = (mine.count - block_row < A::tile_h) ? mine.count - block_row : A::tile_h;
            if constexpr (A::gather_riscs == 2) {
                {
                    DeviceZoneScopedN("G6U-gather-issue-br");
                    g6_untilize::gather_row_block<1>(
                        input,
                        get_read_ptr(A::cb_row_block),
                        mine.start + block_row,
                        block_rows,
                        col_block_idx,
                        first_c);
                }
                {
                    DeviceZoneScopedN("G6U-gather-wait-br");
                    noc_async_read_barrier();
                }
            }
            {
                // Zone-only split: the helper's own wait then returns at once.
                DeviceZoneScopedN("G6U-store-wait");
                cb_wait_front(A::cb_row_block, A::block_tile_cols);
            }
            {
                DeviceZoneScopedN("G6U-store");
                dataflow_kernel_lib::write_sticks_after_untilize<A::cb_row_block>(
                    output, block_rows, chunk_elems * A::elem_bytes, mine.start + block_row, col_elem0 * A::elem_bytes);
            }
        }
    }
}
