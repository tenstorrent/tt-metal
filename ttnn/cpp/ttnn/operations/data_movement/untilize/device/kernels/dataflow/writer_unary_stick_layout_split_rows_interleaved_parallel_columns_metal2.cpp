// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp.
//
// Bindings:
//   dfb::out                       — DFB endpoint (CONSUMER)
//   ta::out                        — TensorAccessor (output, interleaved)
//   args::num_sticks               — RTA
//   args::num_tiles_per_core       — RTA
//   args::tile_width_size          — RTA
//   args::start_stick_id           — RTA
//   args::offset_within_stick      — RTA
//
// Dead-CTA-slot note: legacy CTA[0] (stick_size) was declared but never used in the
// kernel body. Dropped in the metal2 fork.

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
    auto num_tiles_per_core = get_arg(args::num_tiles_per_core);
    auto tile_width_size = get_arg(args::tile_width_size);
    auto start_stick_id = get_arg(args::start_stick_id);
    uint32_t offset_within_stick = get_arg(args::offset_within_stick);

    const auto s = TensorAccessor(ta::out);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);

    uint32_t curr_stick_offset = 0;
    uint32_t row_stick_ids[tile_height];

    auto write_tiles = [&](const uint32_t& num_tiles, const uint32_t& width_size) {
        cb_out.wait_front(num_tiles);
        uint32_t l1_read_addr = cb_out.get_read_ptr();
        for (uint32_t k = 0; k < tile_height; k++) {
            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(
                src,
                s,
                width_size,
                {.offset_bytes = 0},
                {.page_id = row_stick_ids[k], .offset_bytes = curr_stick_offset});
            l1_read_addr += width_size;
        }
        noc.async_write_barrier();
        cb_out.pop_front(num_tiles);
    };

    uint32_t stick_id = start_stick_id;

    uint32_t curr_offset = offset_within_stick;
    for (uint32_t i = 0; i < num_sticks / tile_height; i++) {
        for (uint32_t tile_id = 0; tile_id < num_tiles_per_core; tile_id++) {
            for (uint32_t j = 0; j < tile_height; j++) {
                row_stick_ids[j] = stick_id + j;
            }
            curr_stick_offset = curr_offset;
            write_tiles(1, tile_width_size);
            curr_offset += tile_width_size;
        }

        stick_id += tile_height;
    }
}
