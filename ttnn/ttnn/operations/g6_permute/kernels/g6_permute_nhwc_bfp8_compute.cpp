// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_permute compute (nhwc_tile_bfp8_transpose), per group of 32 input tiles X[h] = [w][c]:
//   transpose_untilize_group: X[h] -> DEST (unpacked bfloat8_b, transposed to [c][w]) -> packed
//     untilized in bfloat16 as 32 rows of 32 values, i.e. row h of a [32, 32 * 32] row-major block.
//   exchange_rows_group: tilize of that block; its 32-column chunk c is output tile c = [h][w].
//     The packer writes bfloat8_b and sets each 16-value group's exponent to the largest of its values.
// bfloat8_b -> bfloat16 and the DEST moves are exact; only the final bfloat8_b pack rounds.
//
// Raw LLK for transpose_untilize_group: kernel_lib has no transpose helper, and the untilize
// helper's unpack init has no transpose, so it cannot pack a transposed tile untilized. The tilize
// step is the tilize helper.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_row_block = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(2);

constexpr uint32_t GROUP_TILES = 32;
constexpr uint32_t dest_tiles = compute_kernel_lib::DEST_AUTO_LIMIT;
static_assert(GROUP_TILES % dest_tiles == 0, "a group is whole DEST batches");

ALWI void transpose_untilize_group() {
    // The previous group's tilize left srcA on the row block.
    reconfig_data_format_srca(cb_input_tiles);
    transpose_init(cb_input_tiles);
    pack_untilize_dest_init<1, 1>(cb_row_block);
    for (uint32_t base = 0; base < GROUP_TILES; base += dest_tiles) {
        // Cumulative wait: each DEST batch starts as soon as its own slots land, whatever the
        // reader's chunk size; the whole group is popped at the end.
        cb_wait_front(cb_input_tiles, base + dest_tiles);
        cb_reserve_back(cb_row_block, dest_tiles);
        tile_regs_acquire();
        transpose_block(cb_input_tiles, base, 0, dest_tiles);
        tile_regs_commit();
        tile_regs_wait();
        // One tile wide, dest_tiles tall: DEST tile i becomes the next page, row base + i of the block.
        pack_untilize_dest<1, 1>(cb_row_block, dest_tiles);
        tile_regs_release();
        cb_push_back(cb_row_block, dest_tiles);
    }
    pack_untilize_uninit(cb_row_block);
    cb_pop_front(cb_input_tiles, GROUP_TILES);
}

void kernel_main() {
    const uint32_t num_groups = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_input_tiles, cb_row_block);
    DataflowBuffer row_block(cb_row_block), output_tiles(cb_output_tiles);

    for (uint32_t g = 0; g < num_groups; ++g) {
        transpose_untilize_group();
        // exchange_rows_group
        compute_kernel_lib::tilize<GROUP_TILES, cb_row_block, cb_output_tiles>(row_block, output_tiles, 1);
    }
}
