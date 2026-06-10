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
// Metal 2.0 bindings (see the program factory):
//   DFBs   : cb_out_rm (consumed)
//   Tensors: output (ROW_MAJOR, page_size = W * elem_size_bytes)
//   CTAs   : block_size, elem_size_bytes
//   RTAs   : Wt, num_tile_rows, start_tile_row, H_logical
//
// The legacy dst_addr buffer-address RTA is now a TensorBinding (TensorAccessor(ta::output)).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include <tt-metalium/constants.hpp>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t num_tile_rows = get_arg(args::num_tile_rows);
    const uint32_t start_tile_row = get_arg(args::start_tile_row);
    const uint32_t H_logical = get_arg(args::H_logical);

    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t elem_size_bytes = get_arg(args::elem_size_bytes);

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

    const auto dst_a = TensorAccessor(ta::output);

    Noc noc;
    DataflowBuffer cb_out_rm(dfb::cb_out_rm);

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
                noc, cb_out_rm, dst_a, abs_row_base, num_valid_rows, tile_width_bytes, block_row_stride_bytes, block);
        }
    }
}
