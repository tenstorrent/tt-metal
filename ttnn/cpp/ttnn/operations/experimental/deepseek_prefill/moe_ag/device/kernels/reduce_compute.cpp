// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// All-gather MoE local reduce, compute: per token, out = sum over its pairs of w * y_row (row-major data treated as
// TILES tiles of 1024 elements: element-wise ops are layout agnostic), accumulated by the packer in L1 (the first pair
// overwrites). The pair count comes in a header page (reduce_reader.cpp).
// CT: 0 TILES
// Common RT: 0 tokens, 1 tokens per core, 2 grid x (this core's n: core_range)
#include <cstdint>
#include "api/compute/common.h"
#include "core_range.hpp"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"

void kernel_main() {
    constexpr uint32_t TILES = get_compile_time_arg_val(0);
    constexpr uint32_t DST = 4;  // fp32 DEST, half sync
    constexpr uint32_t cb_y = tt::CBIndex::c_0, cb_w = tt::CBIndex::c_1, cb_h = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t n =
        core_range(get_common_arg_val<uint32_t>(0), get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2)).n;
    compute_kernel_hw_startup(cb_y, cb_w, cb_out);
    mul_bcast_scalar_init(cb_y, cb_w);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_h, 1);
        const uint32_t cnt = read_tile_value(cb_h, 0, 0);
        cb_reserve_back(cb_out, TILES);
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
                    pack_tile<true>(j, cb_out, j0 + j);
                }
                tile_regs_release();
            }
            cb_pop_front(cb_y, TILES);
            cb_pop_front(cb_w, 1);
        }
        pack_reconfig_l1_acc(0);
        cb_push_back(cb_out, TILES);
        cb_pop_front(cb_h, 1);
    }
}
