// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of reader_unary_stick_layout_split_rows_singlecore.cpp.
//
// Reads row-major sticks from an interleaved input via TensorAccessor and pushes
// tilized blocks into the producer DFB.
//
// Bindings (named, from host KernelSpec):
//   dfb::input               — DFB endpoint (PRODUCER)
//   ta::input                — TensorAccessor (input, interleaved)
//   args::num_sticks
//   args::num_tiles_per_block
//   args::block_width_size
//   args::num_full_blocks_in_row
//   args::start_stick_id
//
// Dead-RTA-slot note (vs legacy positional layout):
//   Legacy pushed 9 RTAs incl. stick_size (slot 2), num_leftover_tiles (slot 6),
//   leftover_width_in_row (slot 7). Kernel only read slots 0,1,3,4,5,8. Dropped
//   in the port (5 RTAs total — matches what the kernel actually consumes).
//   Legacy also pushed CTA[0]=stick_size unused by the kernel — dropped.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t tile_height = 32;

    auto num_sticks = get_arg(args::num_sticks);
    auto num_tiles_per_block = get_arg(args::num_tiles_per_block);
    auto block_width_size = get_arg(args::block_width_size);
    auto num_full_blocks_in_row = get_arg(args::num_full_blocks_in_row);
    auto start_stick_id = get_arg(args::start_stick_id);

    const auto s = TensorAccessor(ta::input);

    Noc noc;
    DataflowBuffer cb_in0(dfb::input);

    uint32_t stick_ids[tile_height];
    uint32_t stick_offset = 0;

    auto read_tiles = [&](const uint32_t& num_tiles, const uint32_t& width_size) {
        cb_in0.reserve_back(num_tiles);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();
        for (uint32_t k = 0; k < tile_height; k++) {
            CoreLocalMem<uint32_t> dst(l1_write_addr);
            noc.async_read(
                s, dst, width_size, {.page_id = stick_ids[k], .offset_bytes = stick_offset}, {.offset_bytes = 0});
            l1_write_addr += width_size;
        }
        stick_offset += width_size;
        noc.async_read_barrier();
        cb_in0.push_back(num_tiles);
    };

    uint32_t stick_id = start_stick_id;
    for (uint32_t i = 0; i < num_sticks / tile_height; i++) {
        for (uint32_t j = 0; j < tile_height; j++) {
            stick_ids[j] = stick_id;
            stick_id++;
        }
        stick_offset = 0;

        for (uint32_t j = 0; j < num_full_blocks_in_row; j++) {
            read_tiles(num_tiles_per_block, block_width_size);
        }
    }
}
