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
#include "experimental/kernel_args.h"

// Retile: untilize input tiles into an intermediate row-major buffer, then tilize into the output
// tile shape. The intermediate is a single L1 allocation shared by untilize (producer) and tilize
// (consumer) to avoid a copy, exposed as two aliased DFB views because the producer and consumer
// need different fixed tile/face geometry: dfb::mid has the input tile shape, dfb::mid_view the
// output tile shape (its bytes stay in the input data format; conversion happens on the final pack).

namespace {

// PACK owns the valid write pointer, so the zero fill runs inside a PACK block.
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
    const uint32_t num_input_blocks = get_arg(args::num_input_blocks);
    const uint32_t num_real_input_rows = get_arg(args::num_real_input_rows);
    // Shrink-case output cap: emit real rows only. Padded rows would OOB the output DRAM buffer.
    const uint32_t num_real_output_rows = get_arg(args::num_real_output_rows);
    if (num_input_blocks == 0 || num_real_output_rows == 0) {
        return;
    }

    constexpr uint32_t tiles_per_block = get_arg(args::tiles_per_block);
    constexpr uint32_t in_tile_height = get_arg(args::in_tile_height);
    constexpr uint32_t out_tile_height = get_arg(args::out_tile_height);
    constexpr uint32_t out_tile_size = get_arg(args::out_tile_size);
    constexpr uint32_t mid_page_size = get_arg(args::mid_page_size);

    static_assert(in_tile_height > 0 && out_tile_height > 0, "retile kernel requires positive tile heights");
    static_assert(
        (in_tile_height >= out_tile_height && (in_tile_height % out_tile_height) == 0) ||
            (out_tile_height > in_tile_height && (out_tile_height % in_tile_height) == 0),
        "retile kernel requires one tile height to divide the other exactly");

    // Shrink: one input tile-row untilizes to `ratio` output tile-rows. Grow: `ratio` input
    // tile-rows form one output tile-row. One tile height must divide the other exactly.
    constexpr bool shrink = in_tile_height >= out_tile_height;
    constexpr uint32_t ratio = shrink ? (in_tile_height / out_tile_height) : (out_tile_height / in_tile_height);

    constexpr uint32_t in_rows_per_iter = shrink ? 1u : ratio;
    constexpr uint32_t out_rows_per_iter = shrink ? ratio : 1u;
    constexpr uint32_t block_pages = in_rows_per_iter * tiles_per_block;
    constexpr uint32_t words_per_out_tile_row = (tiles_per_block * out_tile_size) >> 4;

    const uint32_t num_iters = num_input_blocks / in_rows_per_iter;

    compute_kernel_hw_startup(dfb::src, dfb::mid);

    DataflowBuffer mid(dfb::mid);
    DataflowBuffer mid_view(dfb::mid_view);
    DataflowBuffer out_dfb(dfb::out);

    uint32_t emitted_output_rows = 0;

    for (uint32_t b = 0; b < num_iters; ++b) {
        // Rows beyond num_real_input_rows are grow-case height padding: they don't exist in DRAM,
        // so they are zero-filled into the intermediate instead of untilized from the input.
        const uint32_t block_in_row_start = b * in_rows_per_iter;
        uint32_t real_rows = 0;
        if (block_in_row_start < num_real_input_rows) {
            const uint32_t rem = num_real_input_rows - block_in_row_start;
            real_rows = rem < in_rows_per_iter ? rem : in_rows_per_iter;
        }
        const uint32_t pad_rows = in_rows_per_iter - real_rows;

        if (real_rows > 0) {
            compute_kernel_lib::untilize<
                tiles_per_block,
                dfb::src,
                dfb::mid,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(real_rows);
        }
        for (uint32_t k = 0; k < pad_rows; ++k) {
            fill_zeros_pages(mid, tiles_per_block, mid_page_size);
        }

        mid.wait_front(block_pages);
        uint32_t block_rd_ptr = 0;
        UNPACK({ block_rd_ptr = mid.get_read_ptr(); })

        // dfb::mid_view aliases the dfb::mid L1 region but has no producer of its own, and its output
        // tile-rows sit at non-page-aligned byte offsets within the block that pops can't express.
        // So set its read cursor directly to the block base plus each output tile-row's offset.
        pack_reconfig_data_format(dfb::mid, dfb::out);
        tilize_init(dfb::mid_view, tiles_per_block, dfb::out);
        for (uint32_t r = 0; r < out_rows_per_iter; ++r) {
            if (emitted_output_rows >= num_real_output_rows) {
                break;
            }
            UNPACK({ mid_view.evil_set_read_ptr(block_rd_ptr + r * words_per_out_tile_row); })
            out_dfb.reserve_back(tiles_per_block);
            tilize_block(dfb::mid_view, tiles_per_block, dfb::out);
            out_dfb.push_back(tiles_per_block);
            ++emitted_output_rows;
        }
        tilize_uninit(dfb::mid_view, dfb::out);

        mid.pop_front(block_pages);

        reconfig_data_format_srca(dfb::mid_view, dfb::src);
        pack_reconfig_data_format(dfb::out, dfb::mid);

        if (emitted_output_rows >= num_real_output_rows) {
            break;
        }
    }
}
