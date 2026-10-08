// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// All-gather MoE local reduce with a tiled output, compute: as reduce_compute.cpp per token (sum of w * y_row pairs,
// packer L1 accumulation), but each token's row lands at row i % 32 of a 32-row row-major block (c_24, 32 x TILES
// pages), which is tilized into c_16 (32 TILES tiles: one tile row of the [T, H] partials) every 32 tokens.
// CT: 0 TILES (per row)   Common RT: 0 T, 1 tokens per core (multiples of 32), 2 grid x
#include <cstdint>
#include "api/compute/common.h"
#include "core_range.hpp"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t TILES = get_compile_time_arg_val(0);
    constexpr uint32_t DST = 4;  // fp32 DEST, half sync
    constexpr uint32_t cb_y = tt::CBIndex::c_0, cb_w = tt::CBIndex::c_1, cb_h = tt::CBIndex::c_2;
    constexpr uint32_t cb_rm = tt::CBIndex::c_24, cb_out = tt::CBIndex::c_16;
    const uint32_t n =
        core_range(get_common_arg_val<uint32_t>(0), get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2)).n;
    compute_kernel_hw_startup(cb_y, cb_w, cb_rm);
    for (uint32_t b = 0; b < n / 32; ++b) {
        mul_bcast_scalar_init(cb_y, cb_w);
        pack_reconfig_data_format(cb_rm);
        cb_reserve_back(cb_rm, 32 * TILES);
        for (uint32_t r = 0; r < 32; ++r) {
            cb_wait_front(cb_h, 1);
            const uint32_t cnt = read_tile_value(cb_h, 0, 0);
            for (uint32_t p = 0; p < cnt; ++p) {
                cb_wait_front(cb_y, TILES);
                cb_wait_front(cb_w, 1);
                pack_reconfig_l1_acc(p ? 1 : 0);
                for (uint32_t j0 = 0; j0 < TILES; j0 += DST) {  // DEST holds DST fp32 tiles
                    const uint32_t m = TILES - j0 < DST ? TILES - j0 : DST;
                    tile_regs_acquire();
                    for (uint32_t j = 0; j < m; ++j) {
                        mul_tiles_bcast<BroadcastType::SCALAR>(cb_y, cb_w, j0 + j, 0, j);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t j = 0; j < m; ++j) {
                        pack_tile<true>(j, cb_rm, r * TILES + j0 + j);
                    }
                    tile_regs_release();
                }
                cb_pop_front(cb_y, TILES);
                cb_pop_front(cb_w, 1);
            }
            pack_reconfig_l1_acc(0);
            cb_pop_front(cb_h, 1);
        }
        cb_push_back(cb_rm, 32 * TILES);
        compute_kernel_lib::tilize<32 * TILES, cb_rm, cb_out>(1);  // H / 32 tiles wide
    }
}
