// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    constexpr uint32_t cb_in_scores = get_named_compile_time_arg_val("cb_in_scores");
    constexpr uint32_t cb_in_bias = get_named_compile_time_arg_val("cb_in_bias");
    constexpr uint32_t width_tiles = get_named_compile_time_arg_val("width_tiles");

    CircularBuffer cb_scores(cb_in_scores);
    CircularBuffer cb_bias(cb_in_bias);

    // Get scores and bias tensor accessors
    constexpr auto scores_args = TensorAccessorArgs<0>();
    constexpr auto bias_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();

    const uint32_t scores_addr = get_arg_val<uint32_t>(0);
    const uint32_t bias_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_height_tile = get_arg_val<uint32_t>(2);
    const uint32_t end_height_tile = get_arg_val<uint32_t>(3);

    const auto scores_accessor = TensorAccessor(scores_args, scores_addr);
    const auto bias_accessor = TensorAccessor(bias_args, bias_addr);

    const uint32_t scores_tile_bytes = get_tile_size(cb_in_scores);
    const uint32_t bias_tile_bytes = get_tile_size(cb_in_bias);

    for (uint32_t height_tile = start_height_tile; height_tile < end_height_tile; height_tile++) {
        uint32_t base_page = height_tile * width_tiles;
        for (uint32_t width_tile = 0; width_tile < width_tiles; width_tile++) {
            uint32_t page = base_page + width_tile;
            cb_scores.reserve_back(1);
            cb_bias.reserve_back(1);
            noc.async_read(
                scores_accessor,
                CoreLocalMem<uint32_t>(cb_scores.get_write_ptr()),
                scores_tile_bytes,
                {.page_id = page},
                {});
            noc.async_read(
                bias_accessor, CoreLocalMem<uint32_t>(cb_bias.get_write_ptr()), bias_tile_bytes, {.page_id = page}, {});
            noc.async_read_barrier();
            cb_scores.push_back(1);
            cb_bias.push_back(1);
        }
    }
}
