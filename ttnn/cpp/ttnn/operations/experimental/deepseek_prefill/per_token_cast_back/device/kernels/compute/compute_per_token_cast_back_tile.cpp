// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// per_token_cast_back, tile output: per unit (tile-row x G tiles) tilize the e4m3 rows (unpack_tilize -> bf16
// tiles), multiply each tile by its column-0 scale tile (column broadcast, FPU) and pack the result tiles straight
// to the TILE output CB. No untilize.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t cb_loop_count = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr uint32_t DST_TILES = 4;  // 32-bit DEST, half sync
    static_assert(G % DST_TILES == 0, "G must be a multiple of the DEST batch");

    CircularBuffer cb_tiles_obj(cb_tiles);
    CircularBuffer cb_scale_obj(cb_scale);
    CircularBuffer cb_out_obj(cb_out);
    CircularBuffer cb_loop_count_obj(cb_loop_count);

    compute_kernel_hw_startup(cb_in, cb_tiles);

    cb_loop_count_obj.wait_front(1);
    const uint32_t num_units = cb_loop_count_obj.read_tile_value(0, 0);
    cb_loop_count_obj.pop_front(1);

    for (uint32_t u = 0; u < num_units; ++u) {
        compute_kernel_lib::tilize<G, cb_in, cb_tiles>(1);

        reconfig_data_format(cb_tiles, cb_scale);
        pack_reconfig_data_format(cb_out);
        mul_bcast_cols_init(cb_tiles, cb_scale);
        cb_tiles_obj.wait_front(G);
        cb_scale_obj.wait_front(G);
        cb_out_obj.reserve_back(G);
        for (uint32_t j = 0; j < G; j += DST_TILES) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < DST_TILES; ++k) {
                mul_tiles_bcast_cols(cb_tiles, cb_scale, j + k, j + k, k);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < DST_TILES; ++k) {
                pack_tile(k, cb_out);
            }
            tile_regs_release();
        }
        cb_out_obj.push_back(G);
        cb_tiles_obj.pop_front(G);
        cb_scale_obj.pop_front(G);
    }
}
