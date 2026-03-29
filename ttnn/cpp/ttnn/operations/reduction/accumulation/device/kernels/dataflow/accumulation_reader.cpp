// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

#include "../accumulation_common.hpp"

void kernel_main() {
    constexpr auto input_addrg_args = TensorAccessorArgs<0>();

    uint32_t input_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(1);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    const uint32_t input_tile_offset = get_arg_val<uint32_t>(3);
    const uint32_t start_id = get_arg_val<uint32_t>(4);
    // This is the offset of all dimensions below the accumulation axis
    uint32_t low_rank_offset = get_arg_val<uint32_t>(5);
    // This is the offset of all dimensions above the accumulation axis (HtWt for last two axes)
    uint32_t high_rank_offset = get_arg_val<uint32_t>(6);
    // backward flag (from n-1 to 0)
    const uint32_t flip = get_arg_val<uint32_t>(7);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in_obj(cb_in);

    const uint32_t ublock_size_bytes = get_tile_size(cb_in);
    const uint32_t input_tile_bytes = ublock_size_bytes;

    DPRINT << "rows/core = " << num_rows_per_core << ENDL();
    DPRINT << "tiles/row = " << tiles_per_row << ENDL();
    DPRINT << "cb in = " << cb_in << ENDL();

    const auto input_addrg = TensorAccessor(input_addrg_args, input_base_addr, input_tile_bytes);

    for (uint32_t i = start_id; i < start_id + num_rows_per_core; ++i) {
        DPRINT << "reader i = " << i << ENDL();
        for (uint32_t j = 0; j < tiles_per_row; ++j) {
            DPRINT << "writer j = " << j << ENDL();
            const uint32_t tile_j = flip ? (tiles_per_row - j - 1) : j;
            const uint32_t read_tile_id{
                get_tile_id(low_rank_offset, high_rank_offset, tile_j, tiles_per_row, input_tile_offset)};
            cb_in_obj.reserve_back(ONE_TILE);
            noc.async_read(input_addrg, cb_in_obj, input_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_in_obj.push_back(ONE_TILE);
        }
        ++high_rank_offset;
        if (high_rank_offset >= input_tile_offset) {
            high_rank_offset = 0;
            ++low_rank_offset;
        }
    }
    DPRINT << "reader done" << ENDL();
}
