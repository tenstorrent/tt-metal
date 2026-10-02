// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilizer pool, compute RISC. Untilizes each tile row from the reader into TILE_HEIGHT row-major token
// rows for the writer. The loop bound is this core's share of tile rows, known before launch, so no
// stop signal is needed.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "../dataflow/dispatch_fabric2d_untilize_args.hpp"

namespace {
constexpr dspf2d::UntilizeCtArgs args;
constexpr uint32_t num_blocks = args.num_blocks();
}  // namespace

void kernel_main() {
    CircularBuffer cb_in(args.tile_cb);
    CircularBuffer cb_out(args.row_cb);

    const uint32_t first_tile_row = get_arg_val<uint32_t>(dspf2d::UntilizeRtArg::kFirstTileRow);

    compute_kernel_hw_startup(args.tile_cb, args.row_cb);
    pack_untilize_init<args.block_ct_dim, args.tiles_per_row>(args.tile_cb, args.row_cb);

    for (uint32_t s = first_tile_row; s < args.num_tile_rows; s += args.pool_size) {
        // Reserve the whole tile row first: the packer writes each column block at its own offset into
        // one contiguous run of rows_per_tile_row pages. The CB is a whole number of tile rows deep, so
        // that run never wraps.
        cb_out.reserve_back(args.rows_per_tile_row);
        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_in.wait_front(args.block_ct_dim);
            pack_untilize_block<args.block_ct_dim, args.tiles_per_row>(args.tile_cb, 1, args.row_cb, block);
            cb_in.pop_front(args.block_ct_dim);
        }
        cb_out.push_back(args.rows_per_tile_row);
    }
    pack_untilize_uninit(args.row_cb);
}
