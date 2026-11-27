// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast reader (NCRISC / NoC0) — load_block.
//
// Walks this core's pages first_page + k * page_stride in blocks of block_tiles: every page of a
// block is issued back to back and the block pays one read barrier. Interleaved neighbours live
// in different banks, so a block is block_extent page reads, never one wide transfer.
//
// Every reserve/push uses the nominal block_tiles, the ragged last block included: the CB holds
// pipeline_depth whole blocks, so a short push would break the ring cycle the consumer expects.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in_page_bytes = get_compile_time_arg_val(1);
    constexpr auto input_args = TensorAccessorArgs<2>();

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t block_tiles = get_common_arg_val<uint32_t>(1);
    const uint32_t page_stride = get_common_arg_val<uint32_t>(2);

    const uint32_t first_page = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t last_block_extent = get_arg_val<uint32_t>(2);

    const auto input = TensorAccessor(input_args, input_addr, in_page_bytes);
    Noc noc;
    DataflowBuffer input_tiles(cb_input_tiles);

    uint32_t page = first_page;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_extent = (block_idx + 1 == num_blocks) ? last_block_extent : block_tiles;
        {
            DeviceZoneScopedN("load_block.reserve");
            input_tiles.reserve_back(block_tiles);
        }
        {
            DeviceZoneScopedN("load_block.read");
            for (uint32_t i = 0; i < block_extent; ++i) {
                noc.async_read(
                    input, input_tiles, in_page_bytes, {.page_id = page}, {.offset_bytes = i * in_page_bytes});
                page += page_stride;
            }
            noc.async_read_barrier();
        }
        input_tiles.push_back(block_tiles);
    }
}
