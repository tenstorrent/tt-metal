// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilizer pool, reader RISC. Streams this core's share of the TILE input's stripes into the CB the
// compute kernel packs from. Runs on a core no stream took, beside the stream cores and at the same
// time as their routing prologue.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "dispatch_fabric2d_untilize_args.hpp"

namespace {
using args = dspf2d::UntilizeCtArgs;

constexpr uint32_t cb_in = get_compile_time_arg_val(args::kTileCb);
constexpr uint32_t num_stripes = get_compile_time_arg_val(args::kNumStripes);
constexpr uint32_t pool_size = get_compile_time_arg_val(args::kPoolSize);
constexpr uint32_t tiles_per_row = get_compile_time_arg_val(args::kTilesPerRow);
constexpr uint32_t block_ct_dim = get_compile_time_arg_val(args::kBlockCtDim);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(args::kTileBytes);
constexpr uint32_t num_blocks = tiles_per_row / block_ct_dim;

// Chained after the scalars and the stream-core coordinates, the same way every other kernel of this
// op derives its accessor block, so adding a scalar cannot silently shift it.
constexpr uint32_t accessor_base =
    get_compile_time_arg_val(args::kStreamCoordsBase) + 2u * get_compile_time_arg_val(args::kStreamCount);
constexpr auto in_args = TensorAccessorArgs<accessor_base>();

}  // namespace

void kernel_main() {
    const uint32_t first_stripe = get_arg_val<uint32_t>(dspf2d::UntilizeRtArg::kFirstStripe);
    const auto in_acc = TensorAccessor(in_args, get_arg_val<uint32_t>(dspf2d::UntilizeRtArg::kBufferAddr), tile_bytes);

    // A TILE tensor is paged one tile per page, tile-row-major, so a stripe is tiles_per_row
    // consecutive pages. A ragged sequence's last stripe is read whole as well: its tile-padding rows
    // land in staging pages past the sequence, which the routing pass never reaches.
    for (uint32_t s = first_stripe; s < num_stripes; s += pool_size) {
        const uint32_t base_page = s * tiles_per_row;
        for (uint32_t blk = 0; blk < num_blocks; blk++) {
            // block_ct_dim tiles at a time, which is what pack_untilize consumes per call. The CB is
            // a whole number of blocks deep, so a block never straddles its wrap.
            cb_reserve_back(cb_in, block_ct_dim);
            const uint32_t write_ptr = get_write_ptr(cb_in);
            const uint32_t block_page = base_page + blk * block_ct_dim;
            for (uint32_t col = 0; col < block_ct_dim; col++) {
                noc_async_read_page(block_page + col, in_acc, write_ptr + col * tile_bytes);
            }
            noc_async_read_barrier();
            cb_push_back(cb_in, block_ct_dim);
        }
    }
}
