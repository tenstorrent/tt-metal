// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t Wt = get_arg_val<uint32_t>(1);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(3);
    const uint32_t H_logical = get_arg_val<uint32_t>(4);

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr uint32_t elem_off = dst_args.next_compile_time_args_offset();
    constexpr uint32_t elem_size_bytes = get_compile_time_arg_val(elem_off);

    constexpr uint32_t cb_id_out_rm = get_named_compile_time_arg_val("cb_out_rm");

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

    const auto dst_a = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    CircularBuffer cb_out_rm(cb_id_out_rm);

    constexpr uint32_t block_row_stride_bytes = block_size * TILE_W * elem_size_bytes;
    constexpr uint32_t tile_width_bytes = TILE_W * elem_size_bytes;

#ifdef OUTPUT_RESIDUAL_SUM
    // Same writer also drains the pre-add sum (cb_x_out), which is always TILE — written tile-by-tile to
    // the 2nd output. arg[3] here is a tile-row index, so the tile page_id base is start_tile_row * Wt.
    const uint32_t x_dst_addr = get_arg_val<uint32_t>(5);
    constexpr auto x_dst_args = TensorAccessorArgs<elem_off + 1>();
    constexpr uint32_t cb_id_x_out = get_named_compile_time_arg_val("cb_x_out");
    const uint32_t x_tile_bytes = get_tile_size(cb_id_x_out);
    CircularBuffer cb_x_out(cb_id_x_out);
    const auto s_x = TensorAccessor(x_dst_args, x_dst_addr);
    uint32_t x_tile_id = start_tile_row * Wt;
#endif

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
                noc, cb_out_rm, dst_a, abs_row_base, num_valid_rows, tile_width_bytes, block_row_stride_bytes, block);
#ifdef OUTPUT_RESIDUAL_SUM
            cb_x_out.wait_front(block.full_block_size());
            uint32_t idx = 0;
            for (auto i : block.local()) {
                noc.async_write(
                    cb_x_out, s_x, x_tile_bytes, {.offset_bytes = idx * x_tile_bytes}, {.page_id = x_tile_id});
                x_tile_id++;
                idx++;
            }
            noc.async_write_barrier();
            cb_x_out.pop_front(block.full_block_size());
#endif
        }
    }
}
