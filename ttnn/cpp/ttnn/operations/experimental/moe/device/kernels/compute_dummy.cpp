// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    // Run-time arguments
    uint32_t argidx = 0;
    const auto core_id = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w0 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2c_mm0 = tt::CBIndex::c_2;
    constexpr auto cb_c2c_mm1 = tt::CBIndex::c_3;
    constexpr auto cb_c2w_elt = tt::CBIndex::c_4;
    constexpr auto cb_r2c_in2 = tt::CBIndex::c_5;
    constexpr auto cb_c2w_mm2 = tt::CBIndex::c_6;

    // CB Aliases
    constexpr auto cb_r2c_w1 = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;

    // Constants for MoE
    constexpr uint32_t num_w0_w1_tiles = 224;
    constexpr uint32_t num_w2_tiles_h = 64;
    const uint32_t num_w2_tiles_w = core_id < 32 ? 4 : 3;
    const uint32_t num_mm2_tiles = core_id < 32 ? 4 : 3;

    constexpr uint32_t num_elt_tiles = 1;

    constexpr uint32_t w0_w1_stride = 64;
    constexpr uint32_t w2_stride_w = 1;
    constexpr uint32_t w2_stride_h = 224;

    const uint32_t w0_tile_id_start = core_id;
    const uint32_t w1_tile_id_start = core_id;
    const uint32_t w2_tile_id_start = core_id < 32 ? 4 * core_id : 4 * 32 + 3 * (core_id - 32);

    // Read W0 and W1 from CB into registers
    for (uint32_t i = 0; i < num_w0_w1_tiles; ++i) {
        cb_wait_front(cb_r2c_w0, 1);
        cb_pop_front(cb_r2c_w0, 1);

        cb_wait_front(cb_r2c_w1, 1);
        cb_pop_front(cb_r2c_w1, 1);
    }

    // Write to cb_c2w_elt
    for (uint32_t i = 0; i < num_elt_tiles; ++i) {
        cb_reserve_back(cb_c2w_elt, 1);
        cb_push_back(cb_c2w_elt, 1);
    }

    // Read W2 from DRAM into CB
    for (uint32_t i = 0; i < num_w2_tiles_h; ++i) {
        cb_wait_front(cb_r2c_in2, 1);
        cb_pop_front(cb_r2c_in2, 1);

        for (uint32_t j = 0; j < num_w2_tiles_w; ++j) {
            cb_wait_front(cb_r2c_w2, 1);
            cb_pop_front(cb_r2c_w2, 1);
        }
    }

    // Write to cb_c2w_mm2
    for (uint32_t i = 0; i < num_mm2_tiles; ++i) {
        cb_reserve_back(cb_c2w_mm2, 1);
        cb_push_back(cb_c2w_mm2, 1);
    }
}
}  // namespace NAMESPACE
