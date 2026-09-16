// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilizer pool, compute RISC. Packs each tiled stripe the reader stages into TILE_HEIGHT row-major
// token rows for the writer.
//
// The stripe list is bounded rather than terminated by a sentinel, because this core's share is known
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
    constexpr uint32_t num_stripes = get_compile_time_arg_val(args::kNumStripes);
    constexpr uint32_t pool_size = get_compile_time_arg_val(args::kPoolSize);
    constexpr uint32_t full_ct_dim = get_compile_time_arg_val(args::kTilesPerRow);
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(args::kBlockCtDim);
    constexpr uint32_t rows_per_stripe = get_compile_time_arg_val(args::kRowsPerStripe);
    constexpr uint32_t num_blocks = full_ct_dim / block_ct_dim;

    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_out(cb_out_id);

    const uint32_t first_stripe = get_arg_val<uint32_t>(dspf2d::UntilizeRtArg::kFirstStripe);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);
    pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in_id, cb_out_id);

    for (uint32_t s = first_stripe; s < num_stripes; s += pool_size) {
        // The whole stripe is reserved before any of it is packed: the packer writes each column
        // block at its own offset into one contiguous run of rows_per_stripe pages, so that run must
        // not wrap. The CB is a whole number of stripes deep, which is what holds that.
        cb_out.reserve_back(rows_per_stripe);
        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_in.wait_front(block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(cb_in_id, 1, cb_out_id, block);
            cb_in.pop_front(block_ct_dim);
        }
        cb_out.push_back(rows_per_stripe);
    }
    pack_untilize_uninit(cb_out_id);
}
