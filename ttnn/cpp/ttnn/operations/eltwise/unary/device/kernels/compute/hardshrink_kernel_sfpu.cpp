// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_unary/comp.h"

namespace NAMESPACE {
void MAIN {
    const uint32_t packed_scalar = get_arg_val<uint32_t>(0);
    const auto lambd = reinterpret_cast<const float*>(&packed_scalar);
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;
    constexpr auto cb_tmp0 = tt::CBIndex::c_1;
    init_sfpu(cb_input, cb_output);

    // a⋅1(a+λ<0)+a⋅1(a−λ>0)
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(cb_input, 1);
            cb_reserve_back(cb_tmp0, 1);
            tile_regs_acquire();

            fill_tile(0, *lambd);
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1);
            ltz_tile(0);
            mul_binary_tile_init();
            mul_binary_tile(0, 1);

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, cb_tmp0);
            tile_regs_release();

            cb_push_back(cb_tmp0, 1);
            cb_wait_front(cb_tmp0, 1);
            tile_regs_acquire();

            fill_tile(1, *lambd);

            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);
            sub_binary_tile_init();
            sub_binary_tile(0, 1);
            gtz_tile(0);
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1);
            copy_tile_to_dst_init_short(cb_tmp0);
            copy_tile(cb_tmp0, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1);

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, cb_output);
            tile_regs_release();

            cb_pop_front(cb_input, 1);
            cb_pop_front(cb_tmp0, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
