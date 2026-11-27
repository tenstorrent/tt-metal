// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader (NCRISC, NoC0): gather_row_block producer. Reserves each block's slot, issues its share of
// the gather (all of it when gather_riscs == 1), waits for its own reads, then pushes the block.

#include "api/dataflow/dataflow_api.h"
#include "g6_untilize_gather.hpp"

void kernel_main() {
    using A = g6_untilize::GatherArgs;
    constexpr auto input_args = TensorAccessorArgs<A::next_ct_arg>();

    const g6_untilize::CoreRows mine = g6_untilize::core_rows();
    const uint32_t first_c = g6_untilize::first_tile_col(mine);
    const auto input = TensorAccessor(input_args, get_common_arg_val<uint32_t>(0));

    for (uint32_t col_block_idx = 0; col_block_idx < A::num_col_blocks; ++col_block_idx) {
        for (uint32_t block_row = 0; block_row < mine.count; block_row += A::tile_h) {
            const uint32_t block_rows = (mine.count - block_row < A::tile_h) ? mine.count - block_row : A::tile_h;
            {
                // Blocks only when the writer still holds every slot (>= buffer_depth + 1 blocks per core).
                DeviceZoneScopedN("G6U-gather-reserve");
                cb_reserve_back(A::cb_row_block, A::block_tile_cols);
            }
            {
                DeviceZoneScopedN("G6U-gather-issue");
                g6_untilize::gather_row_block<0>(
                    input, get_write_ptr(A::cb_row_block), mine.start + block_row, block_rows, col_block_idx, first_c);
            }
            {
                DeviceZoneScopedN("G6U-gather-wait");
                noc_async_read_barrier();
            }
            {
                DeviceZoneScopedN("G6U-gather-push");
                cb_push_back(A::cb_row_block, A::block_tile_cols);
            }
        }
    }
}
