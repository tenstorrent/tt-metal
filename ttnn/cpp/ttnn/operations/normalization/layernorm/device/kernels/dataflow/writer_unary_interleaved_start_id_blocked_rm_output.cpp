// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for layernorm with ROW_MAJOR output.
//
// The compute kernel uses pack_untilize_block<block_size, block_size> to convert completed tiles from
// cb_out (CB 16) into cb_out_rm (CB 28) block-by-block.
//
// pack_untilize_block<block_size, block_size> produces true row-major data in cb_out_rm:
//   For each block of block_size tiles, the CB holds a (TILE_H x block_size*TILE_W) row-major array:
//     row r starts at offset  r * block_size * TILE_W * elem_size  from the CB base.
//   This matches the access pattern used by the standard untilize writer
//   (writer_unary_stick_layout_split_rows_multi_core.cpp).
//
// This kernel drains cb_out_rm and writes each row as a single NOC write:
//   one write per (tile-block row) pair  =  TILE_H writes per block.
//
// Compile-time args:
//   CTA[0]    = block_size  (block size in tiles)
//   CTA[1..N] = TensorAccessorArgs for output (ROW_MAJOR, page_size = W * elem_size_bytes)
//   CTA[last] = elem_size_bytes
//
// Runtime args:
//   arg[0] = dst_addr          (output base address)
//   arg[1] = Wt                (width in tiles)
//   arg[2] = num_tile_rows     (number of tile-rows assigned to this core)
//   arg[3] = start_tile_row    (starting tile-row index for this core, in tiles)
//   arg[4] = H_logical         (total valid rows — skip writes beyond this to avoid OOB DRAM writes)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t Wt = get_arg_val<uint32_t>(1);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(3);   // starting tile-row index for this core
    const uint32_t H_logical = get_arg_val<uint32_t>(4);        // total valid (non-padded) rows

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t elem_size_bytes = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    constexpr uint32_t cb_id_out_rm = tt::CBIndex::c_28;

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

    // The output DRAM buffer is ROW_MAJOR with page_size = W * elem_size_bytes (one full row per page).
    // W = Wt * TILE_W, computed from runtime args.
    const uint32_t rm_page_size = Wt * TILE_W * elem_size_bytes;
    const auto dst_a = TensorAccessor(dst_args, dst_addr, rm_page_size);

    // Row stride inside a pack_untilize_block<block_size, block_size> output block.
    // Row r within a block starts at:  l1_base + r * block_row_stride_bytes
    // (This is the standard row-major stride: block_size tiles wide x TILE_W elements per tile.)
    constexpr uint32_t block_row_stride_bytes = block_size * TILE_W * elem_size_bytes;
    constexpr uint32_t tile_width_bytes = TILE_W * elem_size_bytes;

    for (uint32_t ncht = 0; ncht < num_tile_rows; ncht++) {
        const uint32_t abs_row_base = (start_tile_row + ncht) * TILE_H;

        // Clamp writes to valid rows only — rows >= H_logical are padding and must not
        // be written to DRAM (doing so corrupts adjacent allocations).
        uint32_t num_valid_rows = TILE_H;
        if (abs_row_base >= H_logical) {
            num_valid_rows = 0;
        } else if (H_logical - abs_row_base < TILE_H) {
            num_valid_rows = H_logical - abs_row_base;
        }

        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::write_row_major_block_from_cb<decltype(dst_a), decltype(block), TILE_W, TILE_H>(
                cb_id_out_rm, dst_a, abs_row_base, num_valid_rows, tile_width_bytes, block_row_stride_bytes, block);
        }
    }
}
