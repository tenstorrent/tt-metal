// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Multi-threaded variant of reader_unary_stick_layout_split_rows_multicore.cpp: thread t reads the tile-row
// blocks t, t + num_threads, ... of this core's share into a BLOCKED-pattern DFB, so the 32 stick reads a
// block costs are spread over the DM harts while the compute still consumes the blocks in row order.

#include <stdint.h>
#include "api/tile_constants.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t tile_height = get_arg(args::tile_height);
    constexpr uint32_t num_pages_in_row = get_arg(args::num_pages_in_row);
    constexpr uint32_t size_of_valid_data_in_last_page_in_row = get_arg(args::size_of_valid_data_in_last_page_in_row);

    const uint32_t num_rows = get_arg(args::num_rows);
    const uint32_t num_tiles_per_block = get_arg(args::num_tiles_per_block);
    const uint32_t block_width_size = get_arg(args::block_width_size);
    const uint32_t start_page_id = get_arg(args::start_page_id);

    const uint32_t my_thread = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in);

    const uint32_t num_blocks = num_rows / tile_height;
    const uint32_t pages_per_block = tile_height * num_pages_in_row;

    for (uint32_t block = my_thread; block < num_blocks; block += num_threads) {
        uint32_t page_id = start_page_id + block * pages_per_block;
        dfb_in0.reserve_back(num_tiles_per_block);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        for (uint32_t k = 0; k < tile_height; k++) {
            for (uint32_t l = 0; l < num_pages_in_row; l++) {
                uint32_t width_size =
                    (l == num_pages_in_row - 1) ? size_of_valid_data_in_last_page_in_row : block_width_size;
                CoreLocalMem<uint32_t> dst(l1_write_addr);
                noc.async_read(s, dst, width_size, {.page_id = page_id, .offset_bytes = 0}, {.offset_bytes = 0});
                page_id++;
                l1_write_addr += width_size;
            }
        }
        noc.async_read_barrier();
        dfb_in0.push_back(num_tiles_per_block);
    }
}
