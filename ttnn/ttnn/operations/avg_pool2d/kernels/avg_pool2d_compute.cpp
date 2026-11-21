// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// avg_pool2d compute (block operation `reduce_column_block`, op_design.md).
//
// One reduce<SUM, REDUCE_COL> per block of block_ct_tiles channel tile columns: each column's Ht position
// tiles accumulate in DEST (the helper walks the block in DEST chunks of DEST_AUTO_LIMIT columns), scaled by
// the scaler pair the writer built: 1/(H*W) on every position lane, and on the last position tile row only
// the first last_tile_valid_rows lanes (with_partial(): tile-padding rows multiply by 0, rule R1).
// The result of each column is row 0 of one output tile, packed straight into `out`.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    const uint32_t core_ct_tiles = get_arg(args::core_ct_tiles);
    const uint32_t block_ct_tiles = get_arg(args::block_ct_tiles);
    const uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t dest_chunk_tiles = get_arg(args::dest_chunk_tiles);
    constexpr uint32_t C = get_arg(args::compute_threads);
    static_assert(C == 1, "avg_pool2d compute implements the (1, 1, 1) thread plan");
    // The reader emits tiles in chunks of dest_chunk_tiles columns; the helper consumes them in chunks of
    // DEST_AUTO_LIMIT. Both come from the same compute config on the host.
    static_assert(ckl::DEST_AUTO_LIMIT == dest_chunk_tiles, "reader chunk width must equal the helper's DEST chunk");

    compute_kernel_hw_startup(dfb::in, dfb::scaler, dfb::out);
    DataflowBuffer in(dfb::in);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer out(dfb::out);

    for (uint32_t block_start = 0; block_start < core_ct_tiles; block_start += block_ct_tiles) {
        const uint32_t block_cols =
            (core_ct_tiles - block_start < block_ct_tiles) ? core_ct_tiles - block_start : block_ct_tiles;
        ckl::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_COL,
            dfb::in,
            dfb::scaler,
            dfb::out,
            ckl::ReduceInputPolicy::WaitAndPopPerTile,
            ckl::ReduceDataFormatReconfigMode::NONE>(
            in,
            scaler,
            out,
            ckl::ReduceInputBlockShape::of(Ht, block_cols, 1),
            ckl::ReduceInputMemoryLayout::contiguous(),
            ckl::NoAccumulation{},
            ckl::NoOp{},
            ckl::ReducePartialScaler::with_partial());
    }

    // reduce() waits on the scaler pair but never pops it: release it once, after the last block.
    scaler.pop_front(ckl::ReducePartialScaler::with_partial().scaler_tile_count());
}
