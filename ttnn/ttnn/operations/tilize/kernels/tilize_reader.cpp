// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize reader (NCRISC / NoC0) — the `load_block` block operation (op_design.md).
//
// Custom block operation, not dataflow_kernel_lib::read_sticks_for_tilize: that
// helper derives BOTH the CB push quantum and the L1 stick stride from the valid
// row bytes, so narrowing a ragged last column block would push fewer than
// block_width pages and break the CB ring-wrap invariant
// (tilize_helpers_dataflow.inl:92-93, 117, 127; see op_design.md API Mapping).
//
// Per Tensix core: output-tile rectangle [row_start, row_start + core_row_tiles)
// x [col_start, col_start + core_col_tiles), cut into column blocks of
// block_width tiles. Per CB quantum of rows_per_quantum walk positions (tile-rows):
// rows_per_quantum * tile_h stick-segment reads in flight, one barrier, one push
// of the nominal rows_per_quantum * block_width pages (only the NoC transfer
// narrows on the ragged last column block; only the kernel's final quantum may
// hold fewer tile-rows). With read_ahead > 1 the next quantum's reads are issued
// before the previous quantum's (transaction-id) barrier.
//
// Split reader (CT `split_reader`): this RISC-V produces only the EVEN walk
// positions; the BRISC writer produces the odd ones into its own CB.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tilize_stick_reads.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_sticks = get_compile_time_arg_val(0);
    constexpr uint32_t block_width = get_compile_time_arg_val(1);       // tiles per column block (CB quantum)
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);            // sticks per tile-row
    constexpr uint32_t tile_col_bytes = get_compile_time_arg_val(3);    // bytes of one stick per tile-column
    constexpr uint32_t stick_page_bytes = get_compile_time_arg_val(4);  // aligned interleaved stick page
    constexpr bool split_reader = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t depth_in = get_compile_time_arg_val(6);          // CB slots (quanta)
    constexpr uint32_t read_ahead = get_compile_time_arg_val(7);        // quanta of reads in flight
    constexpr uint32_t rows_per_quantum = get_compile_time_arg_val(8);  // tile-rows per CB quantum
    constexpr auto input_args = TensorAccessorArgs<9>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start = get_arg_val<uint32_t>(3);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(4);
    const uint32_t traversal_rotation = get_arg_val<uint32_t>(5);

    const auto input_accessor = TensorAccessor(input_args, src_addr, stick_page_bytes);

    tilize_dataflow::Walker<block_width> walk(row_start, core_row_tiles, col_start, core_col_tiles, traversal_rotation);
    tilize_dataflow::
        StickProducer<cb_input_sticks, block_width, depth_in, read_ahead, tile_h, tile_col_bytes, rows_per_quantum>
            producer(traversal_rotation);

    const uint32_t num_positions = walk.num_positions();
    for (uint32_t seq = 0; seq < num_positions; ++seq, walk.advance()) {
        if (!split_reader || (seq & 1) == 0) {
            producer.issue_row(input_accessor, walk.row(), walk.first_col(), walk.valid_width());
        }
    }
    producer.complete_all();
}
