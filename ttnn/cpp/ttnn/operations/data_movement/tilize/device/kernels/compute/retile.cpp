// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "internal/circular_buffer_interface.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

// Fused retile: untilize input tiles → intermediate RM → tilize to output tile shape.
//
// A single intermediate RM allocation is used for both untilize output and tilize input, avoiding
// an L1 copy. It is exposed as two aliased CB views over the same L1 (the producer and consumer
// need different tile/face geometry, which is fixed per-CB at program-creation time):
//   mid_cb      — input tile geometry; pack_untilize (producer) writes one input tile of RM/page.
//   mid_view_cb — output tile geometry; llk_unpack_tilize (consumer) reads with the output tile's
//                 face_r_dim/num_faces so it consumes the correct number of RM rows.
// The consumer view has no independent producer, so we point its fifo_rd_ptr at the producer's
// block base each iteration. Never call get_tile_size() on either view; use the out_tile_size CT
// arg (input data format) for byte offsets.
//
// Two regimes, selected at compile time from the tile heights (one must divide the other exactly):
//
//   SHRINK (in_tile_height >= out_tile_height), ratio = in_tile_height / out_tile_height:
//     One untilized input tile-row holds `ratio` output tile-rows of RM. The consumer reads them
//     as `ratio` consecutive output tile-rows via fifo_rd_ptr offsets (their byte offsets are not
//     CB-page aligned, so pops can't express them), then pops the whole input tile-row once.
//
//   GROW (out_tile_height > in_tile_height), ratio = out_tile_height / in_tile_height:
//     `ratio` untilized input tile-rows land contiguously and form exactly one output tile-row of
//     RM. A single tilize reads the whole block (one output tile-row).
//
// Height padding (grow): the padded output can be taller than the real (padded) input — the tail
// input tile-rows do not exist in DRAM. `num_real_input_rows` (runtime) marks how many of this
// core's input tile-rows are real; the rest are filled with zeros directly into the intermediate
// CB (no read of invalid input), so the padded output region is zero.

namespace {

// Producer-side zero fill: reserve `num_pages` on the intermediate CB, zero the L1 bytes (PACK
// owns the valid write pointer), and push. Used for padding input tile-rows in the grow case.
ALWI void fill_zeros_pages(DataflowBuffer& dfb, uint32_t num_pages, uint32_t page_size) {
    dfb.reserve_back(num_pages);
    PACK({
        const uint32_t dst_addr = dfb.get_write_ptr() << cb_addr_shift;
        volatile tt_l1_ptr uint32_t* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
        const uint32_t num_words = (num_pages * page_size) / sizeof(uint32_t);
        for (uint32_t i = 0; i < num_words; ++i) {
            dst_ptr[i] = 0;
        }
    })
    dfb.push_back(num_pages);
}

}  // namespace

void kernel_main() {
    const uint32_t num_input_blocks = get_arg_val<uint32_t>(0);
    const uint32_t num_real_input_rows = get_arg_val<uint32_t>(1);
    if (num_input_blocks == 0) {
        return;
    }

    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(0);
    constexpr uint32_t src_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mid_cb = get_compile_time_arg_val(2);
    constexpr uint32_t mid_view_cb = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb = get_compile_time_arg_val(4);
    constexpr uint32_t in_tile_height = get_compile_time_arg_val(5);
    constexpr uint32_t out_tile_height = get_compile_time_arg_val(6);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(7);
    constexpr uint32_t mid_page_size = get_compile_time_arg_val(8);

    static_assert(in_tile_height > 0 && out_tile_height > 0, "retile kernel requires positive tile heights");
    static_assert(
        (in_tile_height >= out_tile_height && (in_tile_height % out_tile_height) == 0) ||
            (out_tile_height > in_tile_height && (out_tile_height % in_tile_height) == 0),
        "retile kernel requires one tile height to divide the other exactly");

    constexpr bool shrink = in_tile_height >= out_tile_height;
    constexpr uint32_t ratio = shrink ? (in_tile_height / out_tile_height) : (out_tile_height / in_tile_height);

    // Per outer iteration: input tile-rows consumed and output tile-rows produced.
    constexpr uint32_t in_rows_per_iter = shrink ? 1u : ratio;
    constexpr uint32_t out_rows_per_iter = shrink ? ratio : 1u;
    constexpr uint32_t block_pages = in_rows_per_iter * tiles_per_block;  // producer pages per iteration
    constexpr uint32_t words_per_out_tile_row = (tiles_per_block * out_tile_size) >> 4;

    const uint32_t num_iters = num_input_blocks / in_rows_per_iter;

    compute_kernel_hw_startup(src_cb, mid_cb);

    DataflowBuffer mid(mid_cb);
    DataflowBuffer out_dfb(out_cb);

    for (uint32_t b = 0; b < num_iters; ++b) {
        // Split this iteration's input tile-rows into real rows (untilized from the input) and
        // padding rows (beyond the real input) that are zero-filled directly into the intermediate.
        const uint32_t block_in_row_start = b * in_rows_per_iter;
        uint32_t real_rows = 0;
        if (block_in_row_start < num_real_input_rows) {
            const uint32_t rem = num_real_input_rows - block_in_row_start;
            real_rows = rem < in_rows_per_iter ? rem : in_rows_per_iter;
        }
        const uint32_t pad_rows = in_rows_per_iter - real_rows;

        // Untilize the real input tile-row(s) for this output block into the producer view.
        if (real_rows > 0) {
            compute_kernel_lib::untilize<
                tiles_per_block,
                src_cb,
                mid_cb,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(real_rows);
        }
        // Zero the padding tile-row(s) so the padded output region is zero instead of invalid data.
        for (uint32_t k = 0; k < pad_rows; ++k) {
            fill_zeros_pages(mid, tiles_per_block, mid_page_size);
        }

        mid.wait_front(block_pages);
        uint32_t block_rd_ptr = 0;
        UNPACK({ block_rd_ptr = get_local_cb_interface(mid_cb).fifo_rd_ptr; })

        // Consume the block through the output-geometry view. Each output tile-row starts at a
        // byte offset within the block; point the view's rd_ptr there before each tilize.
        tilize_init(mid_view_cb, tiles_per_block, out_cb);
        for (uint32_t r = 0; r < out_rows_per_iter; ++r) {
            UNPACK({ get_local_cb_interface(mid_view_cb).fifo_rd_ptr = block_rd_ptr + r * words_per_out_tile_row; })
            out_dfb.reserve_back(tiles_per_block);
            tilize_block(mid_view_cb, tiles_per_block, out_cb);
            out_dfb.push_back(tiles_per_block);
        }
        tilize_uninit(mid_view_cb, out_cb);

        mid.pop_front(block_pages);

        reconfig_data_format_srca(mid_view_cb, src_cb);
        pack_reconfig_data_format(out_cb, mid_cb);
    }
}
