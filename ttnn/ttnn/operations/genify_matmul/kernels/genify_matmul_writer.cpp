// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// genify_matmul writer (BRISC / NoC1) — block operation `store_block` of regime R1.
//
// Drains the output CB in row groups of sb_h x out_block_n tiles (TileRowMajor, matching the
// matmul_block helper's per-row-group push) and writes the valid tiles to the interleaved output
// (DRAM or L1) through TensorAccessor. Padded rows / columns of ragged edge blocks are skipped.
// One flush per row group (the CB slot may be reused once the writes have left L1), one barrier at
// the end.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
// Valid extent of a block starting at `start` inside a region of `valid` tiles (0 when the block lies
// wholly in the padded tail of a ragged edge region: the grid-uniform R1a walk visits such blocks).
FORCE_INLINE uint32_t block_extent(uint32_t valid, uint32_t start, uint32_t nominal) {
    if (valid <= start) {
        return 0;
    }
    return (valid - start) < nominal ? (valid - start) : nominal;
}
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_batches = get_compile_time_arg_val(1);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t out_block_n_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t sb_h = get_compile_time_arg_val(4);
    constexpr uint32_t o_tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t o_page_bytes = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();

    constexpr uint32_t row_group_tiles = sb_h * out_block_n_tiles;
    constexpr uint32_t num_row_groups = block_m_tiles / sb_h;

    uint32_t arg = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t b0 = get_arg_val<uint32_t>(arg++);
    const uint32_t batches_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t row0 = get_arg_val<uint32_t>(arg++);
    const uint32_t rows_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t col0 = get_arg_val<uint32_t>(arg++);
    const uint32_t cols_valid = get_arg_val<uint32_t>(arg++);
    const uint32_t nb_b = get_arg_val<uint32_t>(arg++);
    const uint32_t nb_m = get_arg_val<uint32_t>(arg++);
    const uint32_t nb_n = get_arg_val<uint32_t>(arg++);
    const uint32_t tensor_m_tiles = get_arg_val<uint32_t>(arg++);
    const uint32_t out_n_tiles = get_arg_val<uint32_t>(arg++);

    const auto out_acc = TensorAccessor(out_args, out_addr, o_page_bytes);

    for (uint32_t bb = 0; bb < nb_b; ++bb) {
        const uint32_t blk_b0 = bb * block_batches;
        const uint32_t blk_b_valid =
            (batches_valid - blk_b0) < block_batches ? (batches_valid - blk_b0) : block_batches;
        for (uint32_t mb = 0; mb < nb_m; ++mb) {
            const uint32_t blk_r0 = mb * block_m_tiles;
            const uint32_t blk_rows = block_extent(rows_valid, blk_r0, block_m_tiles);
            for (uint32_t nbk = 0; nbk < nb_n; ++nbk) {
                const uint32_t blk_c0 = nbk * out_block_n_tiles;
                const uint32_t blk_cols = block_extent(cols_valid, blk_c0, out_block_n_tiles);
                for (uint32_t bi = 0; bi < block_batches; ++bi) {
                    const bool batch_valid = bi < blk_b_valid;
                    const uint32_t batch = b0 + blk_b0 + bi;
                    for (uint32_t g = 0; g < num_row_groups; ++g) {
                        cb_wait_front(cb_output_tiles, row_group_tiles);
                        if (batch_valid) {
                            const uint32_t l1_base = get_read_ptr(cb_output_tiles);
                            for (uint32_t r = 0; r < sb_h; ++r) {
                                const uint32_t blk_row = g * sb_h + r;
                                if (blk_row >= blk_rows) {
                                    break;
                                }
                                const uint32_t page =
                                    (batch * tensor_m_tiles + row0 + blk_r0 + blk_row) * out_n_tiles + col0 + blk_c0;
                                uint32_t l1 = l1_base + r * out_block_n_tiles * o_tile_bytes;
                                for (uint32_t n = 0; n < blk_cols; ++n) {
                                    noc_async_write(l1, out_acc.get_noc_addr(page + n), o_tile_bytes);
                                    l1 += o_tile_bytes;
                                }
                            }
                            noc_async_writes_flushed();  // slot is free once the writes left L1
                        }
                        cb_pop_front(cb_output_tiles, row_group_tiles);
                    }
                }
            }
        }
    }
    noc_async_write_barrier();
}
