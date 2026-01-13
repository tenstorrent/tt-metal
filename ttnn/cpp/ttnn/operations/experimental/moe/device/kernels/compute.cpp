// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");

    // Run-time arguments
    uint32_t argidx = 0;
    const auto core_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);

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
    constexpr uint32_t dst0 = 0;

    // Read W0 and W1 from CB into registers
    reconfig_data_format(cb_s2c_in, cb_r2c_w0);
    pack_reconfig_data_format(cb_c2c_mm0);
    mm_init(cb_s2c_in, cb_r2c_w0, cb_c2c_mm0);
    tile_regs_acquire();
    tile_regs_wait();
    for (uint32_t i = 0; i < num_w0_w1_tiles; ++i) {
        cb_wait_front(cb_r2c_w0, 1);
        matmul_tiles(cb_s2c_in, cb_r2c_w0, i, 0, dst0);
        cb_pop_front(cb_r2c_w0, 1);
    }
    silu_tile_init();
    silu_tile(dst0);
    tile_regs_commit();
    cb_reserve_back(cb_c2c_mm0, 1);
    pack_tile(dst0, cb_c2c_mm0);
    cb_push_back(cb_c2c_mm0, 1);
    tile_regs_release();

    reconfig_data_format(cb_s2c_in, cb_r2c_w1);
    pack_reconfig_data_format(cb_c2c_mm1);
    mm_init(cb_s2c_in, cb_r2c_w1, cb_c2c_mm1);

    tile_regs_acquire();
    tile_regs_wait();
    for (uint32_t i = 0; i < num_w0_w1_tiles; ++i) {
        cb_wait_front(cb_r2c_w1, 1);
        matmul_tiles(cb_s2c_in, cb_r2c_w1, i, 0, dst0);
        cb_pop_front(cb_r2c_w1, 1);
    }
    tile_regs_commit();
    cb_reserve_back(cb_c2c_mm1, 1);
    pack_tile(dst0, cb_c2c_mm1);
    cb_push_back(cb_c2c_mm1, 1);
    tile_regs_release();

    // Write to cb_c2w_elt
    binary_op_init_common(cb_c2c_mm0, cb_c2c_mm1, cb_c2w_elt);
    reconfig_data_format(cb_c2c_mm0, cb_c2c_mm1);
    pack_reconfig_data_format(cb_c2w_elt);
    mul_tiles_init(cb_c2c_mm0, cb_c2c_mm1);
    for (uint32_t i = 0; i < num_elt_tiles; ++i) {
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(cb_c2c_mm0, 1);
        cb_wait_front(cb_c2c_mm1, 1);
        mul_tiles(cb_c2c_mm0, cb_c2c_mm1, 0, 0, dst0);
        cb_pop_front(cb_c2c_mm0, 1);
        cb_pop_front(cb_c2c_mm1, 1);
        tile_regs_commit();
        cb_reserve_back(cb_c2w_elt, 1);
        pack_tile(dst0, cb_c2w_elt);
        tile_regs_release();
        cb_push_back(cb_c2w_elt, 1);
    }
    cb_wait_front(cb_c2w_elt, 1);
    cb_push_back(tt::CBIndex::c_4, 1);

    mm_init_short(cb_r2c_in2, cb_r2c_w2, cb_c2w_mm2);
    reconfig_data_format(cb_r2c_in2, cb_r2c_w2);
    pack_reconfig_data_format(cb_c2w_mm2);
    tile_regs_acquire();
    tile_regs_wait();
    for (uint32_t i = 0; i < num_w2_tiles_h; ++i) {
        cb_wait_front(cb_r2c_in2, 1);
        for (uint32_t j = 0; j < num_w2_tiles_w; ++j) {
            cb_wait_front(cb_r2c_w2, 1);
            matmul_tiles(cb_r2c_in2, cb_r2c_w2, 0, 0, j);
            cb_pop_front(cb_r2c_w2, 1);
        }
        cb_pop_front(cb_r2c_in2, 1);
    }

    tile_regs_commit();

    // Write to cb_c2w_mm2
    for (uint32_t i = 0; i < num_mm2_tiles; ++i) {
        cb_reserve_back(cb_c2w_mm2, 1);
        pack_tile(i, cb_c2w_mm2);
        cb_push_back(cb_c2w_mm2, 1);
    }
    tile_regs_release();
}
}  // namespace NAMESPACE
