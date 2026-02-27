// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for layernorm with ROW_MAJOR output.
//
// The compute kernel uses pack_untilize_block<blk, blk> to convert completed tiles from
// cb_out (CB 16) into cb_out_rm (CB 28) block-by-block.
//
// pack_untilize_block<blk, blk> produces true row-major data in cb_out_rm:
//   For each block of blk tiles, the CB holds a (TILE_H x blk*TILE_W) row-major array:
//     row r starts at offset  r * blk * TILE_W * elem_size  from the CB base.
//   This matches the access pattern used by the standard untilize writer
//   (writer_unary_stick_layout_split_rows_multi_core.cpp).
//
// This kernel drains cb_out_rm and writes each row as a single NOC write:
//   one write per (tile-block row) pair  =  TILE_H writes per block.
//
// Compile-time args:
//   CTA[0]    = blk  (block size in tiles)
//   CTA[1..N] = TensorAccessorArgs for output (ROW_MAJOR, page_size = W * elem_size)
//   CTA[last] = elem_size_bytes
//
// Runtime args:
//   arg[0] = dst_addr          (output base address)
//   arg[1] = Wt                (width in tiles)
//   arg[2] = num_tile_rows     (number of tile-rows assigned to this core)
//   arg[3] = tile_row_offset   (starting tile-row index, in tiles)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"

namespace generic = norm::kernel_util::generic;

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t Wt = get_arg_val<uint32_t>(1);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t tile_row_offset = get_arg_val<uint32_t>(3);  // in tiles

    constexpr uint32_t blk = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t elem_size = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    constexpr uint32_t cb_id_out_rm = tt::CBIndex::c_28;

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

    // The output DRAM buffer is ROW_MAJOR with page_size = W * elem_size (one full row per page).
    // W = Wt * TILE_W, computed from runtime args.
    const uint32_t rm_page_size = Wt * TILE_W * elem_size;
    const auto dst_a = TensorAccessor(dst_args, dst_addr, rm_page_size);

    // Row stride inside a pack_untilize_block<blk, blk> output block.
    // Row r within a block starts at:  l1_base + r * block_row_stride_bytes
    // (This is the standard row-major stride: blk tiles wide x TILE_W elements per tile.)
    constexpr uint32_t block_row_stride_bytes = blk * TILE_W * elem_size;

    for (uint32_t ncht = 0; ncht < num_tile_rows; ncht++) {
        const uint32_t abs_row_base = (tile_row_offset + ncht) * TILE_H;

        for (auto block : generic::blocks(Wt, blk)) {
            // Compute produces blk tiles (full_block_size) in cb_out_rm; the last block
            // may have fewer valid tiles (block.size() <= blk), but blk slots are reserved.
            cb_wait_front(cb_id_out_rm, block.full_block_size());
            const uint32_t l1_base = get_read_ptr(cb_id_out_rm);

            // Column byte offset in the output row where this block starts.
            const uint32_t col_byte_offset = block.start() * TILE_W * elem_size;
            // Number of valid bytes to write per row (only block.size() valid tiles).
            const uint32_t valid_bytes = block.size() * TILE_W * elem_size;

            for (uint32_t r = 0; r < TILE_H; r++) {
                const uint32_t l1_src = l1_base + r * block_row_stride_bytes;
                const uint64_t noc_dst = get_noc_addr(abs_row_base + r, dst_a) + col_byte_offset;
                noc_async_write(l1_src, noc_dst, valid_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_id_out_rm, block.full_block_size());
        }
    }
}
