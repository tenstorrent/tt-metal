// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilizer pool, compute RISC. Packs each tiled tile row the reader stages into TILE_HEIGHT row-major
// token rows for the writer.
//
// The tile row list is bounded rather than terminated by a sentinel, because this core's share is known
// before the launch: a compute kernel that waited on a CB for its stop signal would spin forever if
// the reader beside it ever failed to reach the end, on an op where a spin is a wedged board.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "../dataflow/dispatch_fabric2d_untilize_args.hpp"

using args = dspf2d::UntilizeCtArgs;

void kernel_main() {
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(args::kTileCb);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(args::kRowCb);
    constexpr uint32_t num_tile_rows = get_compile_time_arg_val(args::kNumTileRows);
    constexpr uint32_t pool_size = get_compile_time_arg_val(args::kPoolSize);
    constexpr uint32_t full_ct_dim = get_compile_time_arg_val(args::kTilesPerRow);
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(args::kBlockCtDim);
    constexpr uint32_t rows_per_tile_row = get_compile_time_arg_val(args::kRowsPerTileRow);
    constexpr uint32_t num_blocks = full_ct_dim / block_ct_dim;

    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_out(cb_out_id);

    const uint32_t first_tile_row = get_arg_val<uint32_t>(dspf2d::UntilizeRtArg::kFirstTileRow);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);
    pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in_id, cb_out_id);

    for (uint32_t s = first_tile_row; s < num_tile_rows; s += pool_size) {
        // The whole tile row is reserved before any of it is packed: the packer writes each column
        // block at its own offset into one contiguous run of rows_per_tile_row pages, so that run must
        // not wrap. The CB is a whole number of tile rows deep, which is what holds that.
        cb_out.reserve_back(rows_per_tile_row);
        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_in.wait_front(block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(cb_in_id, 1, cb_out_id, block);
            cb_in.pop_front(block_ct_dim);
        }
        cb_out.push_back(rows_per_tile_row);
    }
    pack_untilize_uninit(cb_out_id);
}
