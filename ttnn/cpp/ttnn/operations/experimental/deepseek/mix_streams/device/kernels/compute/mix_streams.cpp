// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

// One output tile per iteration:
//   out = comb^T @ streams + post (x) sublayer_out
// Both terms are single-tile matmuls accumulated into the same DST tile, so the whole
// "_mix" step costs two matmuls and one pack per output tile. The reader hands over a
// pre-transposed comb tile and a post tile whose padding is zeroed, which is what makes
// the second matmul equal the placement outer product (only column 0 of post and row 0
// of sublayer_out carry data).
void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_comb = get_compile_time_arg_val(0);
    constexpr uint32_t cb_post = get_compile_time_arg_val(1);
    constexpr uint32_t cb_streams = get_compile_time_arg_val(2);
    constexpr uint32_t cb_sub = get_compile_time_arg_val(3);
    constexpr uint32_t cb_out = get_compile_time_arg_val(4);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(5);

    constexpr uint32_t one_tile = 1;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_comb, cb_streams, cb_out);
    matmul_init(cb_comb, cb_streams);

    const uint32_t end_tile = start_tile + num_tiles;
    uint32_t tile = start_tile;
    while (tile < end_tile) {
        const uint32_t t = tile / n_tiles;
        const uint32_t group_end = (end_tile < (t + 1) * n_tiles) ? end_tile : (t + 1) * n_tiles;

        cb_wait_front(cb_comb, one_tile);
        cb_wait_front(cb_post, one_tile);

        for (uint32_t page = tile; page < group_end; ++page) {
            cb_wait_front(cb_streams, one_tile);
            cb_wait_front(cb_sub, one_tile);

            tile_regs_acquire();
            matmul_init(cb_comb, cb_streams);
            matmul_tiles(cb_comb, cb_streams, 0, 0, 0);
            matmul_init(cb_post, cb_sub);
            matmul_tiles(cb_post, cb_sub, 0, 0, 0);
            tile_regs_commit();

            cb_reserve_back(cb_out, one_tile);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, one_tile);

            cb_pop_front(cb_streams, one_tile);
            cb_pop_front(cb_sub, one_tile);
        }

        cb_pop_front(cb_comb, one_tile);
        cb_pop_front(cb_post, one_tile);
        tile = group_end;
    }
}
