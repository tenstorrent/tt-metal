// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize compute — `tilize_block`.
//
// One `compute_kernel_lib::tilize` call per BLOCK: the block's column extent is
// the FIRST template parameter (it is what `tilize_init` programs), and the
// block's row extent is the runtime `num_blocks` argument. The helper's own
// per-tile-row traversal lives inside it, under a single init/uninit pair and a
// single unpack+pack data-format reconfig per block.
//
// No raw LLK: `compute_kernel_lib::tilize` covers this phase completely,
// including fast/regular path selection, the dtype reconfig (which is where the
// value-preserving `dtype=` cast happens, at pack time) and the CB handshake.
// `total_input_pages` is deliberately omitted — both CBs carry tile-sized
// pages, i.e. the helper's symmetric mode, which is what
// `TilizeGranularity::TILE` on the reader side produces.

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_row_blocks = get_compile_time_arg_val(3);  // R
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(4);
    constexpr uint32_t num_w_chunks = get_compile_time_arg_val(5);

    const uint32_t start_block_id = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(1);

    // PREREQUISITE of compute_kernel_lib::tilize (tilize_helpers.hpp:89-93).
    compute_kernel_hw_startup(cb_input_rows, cb_output_tiles);

    for (uint32_t b = 0; b < num_blocks_this_core; ++b) {
        // resolve_block — the identical derivation the reader and writer run.
        const uint32_t block_id = start_block_id + b;
        const uint32_t row_group = block_id / num_w_chunks;
        const uint32_t row_start = (row_group * tensor_row_blocks) / num_row_groups;
        const uint32_t row_end = ((row_group + 1) * tensor_row_blocks) / num_row_groups;
        const uint32_t block_row_extent = row_end - row_start;

        // tilize_block: block_width_tiles (CT) x block_row_extent (RT) tiles.
        compute_kernel_lib::tilize<block_width_tiles, cb_input_rows, cb_output_tiles>(block_row_extent);
    }
}
