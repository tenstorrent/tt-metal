// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tilize writer (BRISC / NoC1) — the `store_block` block operation (op_design.md),
// plus, under the split reader, the odd half of `load_block`.
//
// No kernel_lib dataflow helper writes TILE pages (write_sticks_after_untilize
// writes ROW_MAJOR sticks), so this is a custom block operation.
//
// store_block, per CB quantum of rows_per_quantum tile-rows: wait the quantum's
// pages -> valid_width tile-page writes per tile-row in flight -> one flush (L1
// source reads done) -> pop the quantum. The flush, rather than a full write
// barrier, is enough to release the CB slots; one write barrier at kernel end
// guarantees the data has landed. Only the kernel's final quantum may be partial.
//
// Split reader (CT `split_reader`): this RISC-V also produces the ODD walk
// positions into cb_input_sticks_odd (its own CB, single producer). At odd
// position q it issues q's reads, and while they are in flight it stores every
// position whose inputs are already published (all positions <= the last odd
// position it pushed). Deadlock-free: each store waits on a position whose even
// input NCRISC produces independently and whose odd input this RISC-V has
// already pushed; the output CB (depth_out tile-rows) is drained in order.
//
// Tile-rows are walked with the same per-core rotation as the reader
// (tilize_stick_reads.hpp), so the CB FIFO order agrees.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tilize_stick_reads.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width = get_compile_time_arg_val(1);     // tiles per column block (CB quantum)
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(2);  // one output TILE page
    constexpr bool split_reader = get_compile_time_arg_val(3) != 0;
    constexpr uint32_t cb_input_sticks_odd = get_compile_time_arg_val(4);
    constexpr uint32_t tile_h = get_compile_time_arg_val(5);
    constexpr uint32_t tile_col_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t stick_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t depth_in = get_compile_time_arg_val(8);
    constexpr uint32_t read_ahead = get_compile_time_arg_val(9);
    constexpr uint32_t rows_per_quantum = get_compile_time_arg_val(10);  // tile-rows per CB quantum
    constexpr auto output_args = TensorAccessorArgs<11>();
    constexpr auto input_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(2);
    const uint32_t col_start = get_arg_val<uint32_t>(3);
    const uint32_t core_col_tiles = get_arg_val<uint32_t>(4);
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(5);  // C: output tile-columns of the whole tensor
    const uint32_t traversal_rotation = get_arg_val<uint32_t>(6);
    const uint32_t src_addr = get_arg_val<uint32_t>(7);  // input stick buffer (split reader only)

    const auto output_accessor = TensorAccessor(output_args, dst_addr, out_tile_bytes);

    tilize_dataflow::Walker<block_width> store_walk(
        row_start, core_row_tiles, col_start, core_col_tiles, traversal_rotation);
    const uint32_t num_positions = store_walk.num_positions();

    // Split reader runs one tile-row per quantum (host-enforced), so this stores one position.
    auto store_next = [&]() {
        tilize_dataflow::store_rows<cb_output_tiles, block_width, out_tile_bytes>(
            output_accessor, store_walk, tiles_per_row, 1);
    };

    static_assert(!split_reader || rows_per_quantum == 1, "the split reader alternates CBs per tile-row");
    if constexpr (split_reader) {
        const auto input_accessor = TensorAccessor(input_args, src_addr, stick_page_bytes);
        tilize_dataflow::Walker<block_width> load_walk(
            row_start, core_row_tiles, col_start, core_col_tiles, traversal_rotation);
        tilize_dataflow::
            StickProducer<cb_input_sticks_odd, block_width, depth_in, read_ahead, tile_h, tile_col_bytes, 1>
                producer(traversal_rotation);

        uint32_t stored = 0;     // positions [0, stored) are written
        uint32_t published = 0;  // positions [0, published) have all their inputs pushed
        for (uint32_t seq = 0; seq < num_positions; ++seq, load_walk.advance()) {
            if ((seq & 1) == 0) {
                continue;
            }
            const uint32_t pushed_before = producer.rows_pushed();  // odd items pushed
            producer.issue_row(input_accessor, load_walk.row(), load_walk.first_col(), load_walk.valid_width());
            const uint32_t pushed_after = producer.rows_pushed();
            if (pushed_after != pushed_before) {
                // The k-th odd item (k = pushed_after - 1) is position 2k + 1: all positions <= it are publishable.
                published = 2 * pushed_after;
            }
            while (stored < published) {
                store_next();
                ++stored;
            }
        }
        producer.complete_all();
        while (stored < num_positions) {
            store_next();
            ++stored;
        }
    } else {
        for (uint32_t done = 0; done < num_positions; done += rows_per_quantum) {
            const uint32_t remaining = num_positions - done;
            tilize_dataflow::store_rows<cb_output_tiles, block_width, out_tile_bytes>(
                output_accessor,
                store_walk,
                tiles_per_row,
                remaining < rows_per_quantum ? remaining : rows_per_quantum);
        }
    }
    noc_async_write_barrier();
}
