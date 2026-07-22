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

// Retile: untilize input tiles into an intermediate row-major buffer, then tilize into the output
// tile shape. The intermediate is a single L1 allocation shared by untilize (producer) and tilize
// (consumer) to avoid a copy, exposed as two aliased CB views because the producer and consumer
// need different fixed tile/face geometry: mid_cb has the input tile shape, mid_view_cb the output
// tile shape (its bytes stay in the input data format; conversion happens on the final pack).

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

    // Shrink: one input tile-row untilizes to `ratio` output tile-rows. Grow: `ratio` input
    // tile-rows form one output tile-row. One tile height must divide the other exactly.
    constexpr bool shrink = in_tile_height >= out_tile_height;
    constexpr uint32_t ratio = shrink ? (in_tile_height / out_tile_height) : (out_tile_height / in_tile_height);

    constexpr uint32_t in_rows_per_iter = shrink ? 1u : ratio;
    constexpr uint32_t out_rows_per_iter = shrink ? ratio : 1u;
    constexpr uint32_t block_pages = in_rows_per_iter * tiles_per_block;
    constexpr uint32_t words_per_out_tile_row = (tiles_per_block * out_tile_size) >> 4;

    const uint32_t num_iters = num_input_blocks / in_rows_per_iter;

    compute_kernel_hw_startup(src_cb, mid_cb);

    DataflowBuffer mid(mid_cb);
    DataflowBuffer out_dfb(out_cb);

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
                src_cb,
                mid_cb,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(real_rows);
        }
        for (uint32_t k = 0; k < pad_rows; ++k) {
            fill_zeros_pages(mid, tiles_per_block, mid_page_size);
        }

        mid.wait_front(block_pages);
        uint32_t block_rd_ptr = 0;
        UNPACK({ block_rd_ptr = get_local_cb_interface(mid_cb).fifo_rd_ptr; })

        // mid_view_cb aliases the mid_cb L1 region but has no producer of its own, and its output
        // tile-rows sit at non-page-aligned byte offsets within the block that pops can't express.
        // So set its fifo_rd_ptr directly to the block base plus each output tile-row's offset.
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
