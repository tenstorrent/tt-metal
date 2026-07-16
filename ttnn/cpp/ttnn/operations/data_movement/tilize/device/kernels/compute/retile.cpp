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
// Intermediate page size is max(in_tile_size, out_tile_size), passed as a CT arg.
// Never call get_tile_size() on the mid CB.
//
// A single intermediate RM CB (c_1, mid_cb) is used for both untilize output and
// tilize input, avoiding an L1 copy. It is double-buffered so the reader/writer can
// overlap with compute. tilize reads the RM bytes directly, re-interpreting them as
// height_ratio output tile-rows via fifo_rd_ptr manipulation.
//
// Requires in_tile_height >= out_tile_height and in_tile_height % out_tile_height == 0.

void kernel_main() {
    const uint32_t num_input_blocks = get_arg_val<uint32_t>(0);
    if (num_input_blocks == 0) {
        return;
    }

    constexpr uint32_t tiles_per_block = get_compile_time_arg_val(0);
    constexpr uint32_t src_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mid_cb = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb = get_compile_time_arg_val(3);
    constexpr uint32_t in_tile_height = get_compile_time_arg_val(4);
    constexpr uint32_t out_tile_height = get_compile_time_arg_val(5);
    constexpr uint32_t out_tile_size = get_compile_time_arg_val(6);

    static_assert(in_tile_height >= out_tile_height, "retile kernel requires in_tile_height >= out_tile_height");
    static_assert(
        out_tile_height > 0 && (in_tile_height % out_tile_height) == 0,
        "retile kernel requires in_tile_height to be divisible by out_tile_height");

    constexpr uint32_t height_ratio = in_tile_height / out_tile_height;
    constexpr uint32_t words_per_out_tile_row = (tiles_per_block * out_tile_size) >> 4;

    compute_kernel_hw_startup(src_cb, mid_cb);

    DataflowBuffer mid(mid_cb);
    DataflowBuffer out_dfb(out_cb);

    for (uint32_t b = 0; b < num_input_blocks; ++b) {
        compute_kernel_lib::untilize<
            tiles_per_block,
            src_cb,
            mid_cb,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);

        // One untilize block holds height_ratio output tile-rows of RM data in N max-sized pages.
        // tilize reads the same intermediate CB directly (no copy), re-interpreting the RM bytes
        // as height_ratio consecutive output tile-rows via fifo_rd_ptr manipulation.
        mid.wait_front(tiles_per_block);
        uint32_t saved_rd_ptr = 0;
        UNPACK({ saved_rd_ptr = get_local_cb_interface(mid_cb).fifo_rd_ptr; })

        tilize_init(mid_cb, tiles_per_block, out_cb);
        for (uint32_t r = 0; r < height_ratio; ++r) {
            UNPACK({
                if (r > 0) {
                    get_local_cb_interface(mid_cb).fifo_rd_ptr = saved_rd_ptr + r * words_per_out_tile_row;
                }
            })
            out_dfb.reserve_back(tiles_per_block);
            tilize_block(mid_cb, tiles_per_block, out_cb);
            out_dfb.push_back(tiles_per_block);
        }
        UNPACK({ get_local_cb_interface(mid_cb).fifo_rd_ptr = saved_rd_ptr; })
        mid.pop_front(tiles_per_block);
        tilize_uninit(mid_cb, out_cb);

        reconfig_data_format_srca(mid_cb, src_cb);
        pack_reconfig_data_format(out_cb, mid_cb);
    }
}
