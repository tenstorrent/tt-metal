// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for layernorm with ROW_MAJOR output.
//
// The compute kernel uses pack_untilize_block<block_size, block_size> to convert completed tiles from
// cb_out into cb_out_rm block-by-block.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include <tt-metalium/constants.hpp>
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    auto Wt = get_arg(args::Wt);
    auto num_tile_rows = get_arg(args::num_tile_rows);
    auto start_tile_row = get_arg(args::writer_start);
    auto H_logical = get_arg(args::H_logical);

    constexpr auto block_size = get_arg(args::block_size);
    constexpr auto elem_size_bytes = get_arg(args::elem_size_bytes);

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

    const auto dst_a = TensorAccessor(ta::output);

    Noc noc;
    DataflowBuffer cb_out_rm(dfb::cb_out_rm);

    constexpr uint32_t block_row_stride_bytes = block_size * TILE_W * elem_size_bytes;
    constexpr uint32_t tile_width_bytes = TILE_W * elem_size_bytes;

    for (uint32_t ncht = 0; ncht < num_tile_rows; ncht++) {
        const uint32_t abs_row_base = (start_tile_row + ncht) * TILE_H;

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
