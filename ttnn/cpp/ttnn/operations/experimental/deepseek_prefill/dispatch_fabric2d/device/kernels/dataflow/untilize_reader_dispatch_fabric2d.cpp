// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilizer pool, reader RISC. Reads this core's share of the TILE input's tile rows into the CB the
// compute kernel untilizes from. Runs on a core no stream uses, while the stream cores build their
// routing index.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "dispatch_fabric2d_untilize_args.hpp"

namespace {
using args = dspf2d::UntilizeCtArgs;

constexpr uint32_t cb_in = get_compile_time_arg_val(args::kTileCb);
constexpr uint32_t num_tile_rows = get_compile_time_arg_val(args::kNumTileRows);
constexpr uint32_t pool_size = get_compile_time_arg_val(args::kPoolSize);
constexpr uint32_t tiles_per_row = get_compile_time_arg_val(args::kTilesPerRow);
constexpr uint32_t block_ct_dim = get_compile_time_arg_val(args::kBlockCtDim);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(args::kTileBytes);
constexpr uint32_t num_blocks = tiles_per_row / block_ct_dim;

// The accessor args follow the scalars and the stream-core coordinates.
constexpr uint32_t accessor_base =
    get_compile_time_arg_val(args::kStreamCoordsBase) + 2u * get_compile_time_arg_val(args::kStreamCount);
constexpr auto in_args = TensorAccessorArgs<accessor_base>();

}  // namespace

void kernel_main() {
    const uint32_t first_tile_row = get_arg_val<uint32_t>(dspf2d::UntilizeRtArg::kFirstTileRow);
    const auto in_acc = TensorAccessor(in_args, get_arg_val<uint32_t>(dspf2d::UntilizeRtArg::kBufferAddr), tile_bytes);

    // A TILE tensor has one tile per page in tile-row-major order, so a tile row is tiles_per_row
    // consecutive pages. When the sequence length is not a multiple of 32, the last tile row is still
    // read whole: its padding rows land in staging pages past the sequence, which the routing pass
    // never reads.
    for (uint32_t s = first_tile_row; s < num_tile_rows; s += pool_size) {
        const uint32_t base_page = s * tiles_per_row;
        for (uint32_t blk = 0; blk < num_blocks; blk++) {
            // block_ct_dim tiles at a time, as pack_untilize consumes them. The CB is a whole number of
            // blocks deep, so a block never wraps.
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
