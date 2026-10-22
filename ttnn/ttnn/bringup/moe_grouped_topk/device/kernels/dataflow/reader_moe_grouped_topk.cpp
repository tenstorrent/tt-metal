// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in_scores_id = get_named_compile_time_arg_val("cb_in_scores");
    constexpr uint32_t cb_in_bias_id = get_named_compile_time_arg_val("cb_in_bias");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");
    constexpr uint32_t scores_page_size = get_named_compile_time_arg_val("scores_page_size");
    constexpr uint32_t bias_page_size = get_named_compile_time_arg_val("bias_page_size");

    // Get scores and bias tensor accessors
    constexpr auto scores_args = TensorAccessorArgs<0>();
    constexpr auto bias_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();

    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(2);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(3);

    const auto scores_accessor = TensorAccessor(scores_args, scores_addr, scores_page_size);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr, bias_page_size);

    Noc noc;
    CircularBuffer cb_in_scores(cb_in_scores_id);
    CircularBuffer cb_in_bias(cb_in_bias_id);
    const uint32_t scores_tile_bytes = cb_in_scores.get_tile_size();
    const uint32_t bias_tile_bytes = cb_in_bias.get_tile_size();

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        uint32_t base_page = height_tile * width_tiles;
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            uint32_t page = base_page + width_tile;
            cb_in_scores.reserve_back(1);
            cb_in_bias.reserve_back(1);
            noc.async_read(scores_accessor, cb_in_scores, scores_tile_bytes, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read(bias_accessor, cb_in_bias, bias_tile_bytes, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in_scores.push_back(1);
            cb_in_bias.push_back(1);
        }
    }
}
