// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/tilize.h"
#include "api/compute/untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "tt-metalium/constants.hpp"

// Compute kernel for per_token_cast_back.
// For each tile-row (32 sticks of H elements):
//   Phase 1: tilize_block(cb_in_fp8, H_tiles, cb_fp8_tile)
//       -- fp8 ROW_MAJOR -> fp8 TILE (no dtype conversion, just layout)
//   Phase 2: copy_tile + pack_tile  (per tile)
//       -- fp8 TILE -> DST (unpacker fp8 -> bf16/fp32 at L1 boundary)
//       -- DST -> cb_out_tile (bf16/fp32 TILE; packer just writes)
//   Phase 3: untilize_block(cb_out_tile, H_tiles, cb_out_rm)
//       -- bf16/fp32 TILE -> bf16/fp32 ROW_MAJOR
//
// v0 ignores the scale tensor (assumes scale == 1.0).

void kernel_main() {
    constexpr uint32_t cb_in_fp8 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_fp8_tile = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out_tile = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(3);
    constexpr uint32_t H_tiles = get_compile_time_arg_val(4);

    constexpr uint32_t TILE_HEIGHT = tt::constants::TILE_HEIGHT;

    uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in_fp8, cb_out_rm);

    for (uint32_t row = 0; row < num_tile_rows; ++row) {
        // Phase 1: fp8 RM -> fp8 TILE (no dtype change).
        tilize_init(cb_in_fp8, H_tiles, cb_fp8_tile);
        cb_wait_front(cb_in_fp8, TILE_HEIGHT);
        cb_reserve_back(cb_fp8_tile, H_tiles);
        tilize_block(cb_in_fp8, H_tiles, cb_fp8_tile);
        cb_push_back(cb_fp8_tile, H_tiles);
        cb_pop_front(cb_in_fp8, TILE_HEIGHT);
        tilize_uninit(cb_in_fp8, cb_fp8_tile);

        // Phase 2: fp8 TILE -> bf16/fp32 TILE via copy_tile + pack_tile.
        // Unpacker reads fp8 from cb_fp8_tile and converts to SrcA -> DST (cb_out_tile's dtype).
        // Packer writes DST -> cb_out_tile (same dtype, no conversion).
        copy_tile_init(cb_fp8_tile);
        cb_wait_front(cb_fp8_tile, H_tiles);
        cb_reserve_back(cb_out_tile, H_tiles);
        for (uint32_t t = 0; t < H_tiles; ++t) {
            acquire_dst();
            copy_tile(cb_fp8_tile, t, 0);
            pack_tile(0, cb_out_tile);
            release_dst();
        }
        cb_push_back(cb_out_tile, H_tiles);
        cb_pop_front(cb_fp8_tile, H_tiles);

        // Phase 3: bf16/fp32 TILE -> bf16/fp32 RM.
        untilize_init(cb_out_tile);
        cb_wait_front(cb_out_tile, H_tiles);
        cb_reserve_back(cb_out_rm, TILE_HEIGHT);
        untilize_block(cb_out_tile, H_tiles, cb_out_rm);
        cb_push_back(cb_out_rm, TILE_HEIGHT);
        cb_pop_front(cb_out_tile, H_tiles);
        untilize_uninit(cb_out_tile);
    }
}
