// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/where.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;   // input
    constexpr auto cb_output = tt::CBIndex::c_2;  // output

    constexpr auto cb_tanh_lut = tt::CBIndex::c_1;   // lut tanh(x)
    constexpr auto cb_exp_2x = tt::CBIndex::c_3;     // exp(2x)
    constexpr auto cb_sub = tt::CBIndex::c_4;        // exp(2x) - 1
    constexpr auto cb_add = tt::CBIndex::c_5;        // recip ( exp(2x) + 1 )
    constexpr auto cb_tanh_exp = tt::CBIndex::c_6;   // tanh[x] = (exp[2x] - 1) / (exp[2x] + 1)

    constexpr uint32_t one = 0x3f800000u;    //  1.0f
    constexpr uint32_t two = 0x40000000u;    //  2.0f
    constexpr uint32_t limit = 0x40600000u;  //  3.5f

    init_sfpu(cb_input, cb_output);
    binop_with_scalar_tile_init();

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(cb_output, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_reserve_back(cb_tanh_lut, 1);
            cb_wait_front(cb_input, 1);
            tile_regs_acquire();

            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);
            tanh_tile_init();
            tanh_tile(0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tanh_lut);
            tile_regs_release();
            cb_push_back(cb_tanh_lut, 1);

            // exp(2x)
            cb_reserve_back(cb_exp_2x, 1);
            tile_regs_acquire();
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);

            mul_unary_tile(0, two);
            exp_tile_init();
            exp_tile(0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_exp_2x);
            tile_regs_release();

            cb_push_back(cb_exp_2x, 1);

            // exp(2x) - 1

            cb_reserve_back(cb_sub, 1);
            cb_wait_front(cb_exp_2x, 1);
            tile_regs_acquire();
            copy_tile_init(cb_exp_2x);
            copy_tile(cb_exp_2x, 0, 0);

            sub_unary_tile(0, one);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_sub);

            tile_regs_release();

            cb_push_back(cb_sub, 1);

            // recip ( exp(2x) + 1 )

            cb_reserve_back(cb_add, 1);
            tile_regs_acquire();
            copy_tile_init(cb_exp_2x);
            copy_tile(cb_exp_2x, 0, 0);

            add_unary_tile(0, one);
            recip_tile_init();
            recip_tile(0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_add);

            tile_regs_release();

            cb_push_back(cb_add, 1);
            cb_pop_front(cb_exp_2x, 1);

            // tanh[x] = (exp[2x] - 1) *  1 / (exp[2x] + 1)

            cb_reserve_back(cb_tanh_exp, 1);
            cb_wait_front(cb_sub, 1);
            cb_wait_front(cb_add, 1);
            tile_regs_acquire();

#ifdef TANH_BF16
            mul_tiles_init(cb_sub, cb_add);
            mul_tiles(cb_sub, cb_add, 0, 0, 0);
#endif

#ifdef TANH_FP32
            copy_tile_init(cb_sub);
            copy_tile(cb_sub, 0, 0);
            copy_tile_init(cb_add);
            copy_tile(cb_add, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
#endif

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tanh_exp);

            tile_regs_release();

            cb_push_back(cb_tanh_exp, 1);
            cb_pop_front(cb_sub, 1);
            cb_pop_front(cb_add, 1);

            // output = cb_tanh_lut if abs(x) > 3.5, otherwise cb_tanh_exp
            cb_wait_front(cb_tanh_lut, 1);
            cb_wait_front(cb_tanh_exp, 1);
            cb_wait_front(cb_input, 1);

            tile_regs_acquire();
            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, 0);
            abs_tile_init();
            abs_tile(0);
            unary_gt_tile_init();
            unary_gt_tile(0, limit);
            copy_tile_init(cb_tanh_lut);
            copy_tile(cb_tanh_lut, 0, 1);
            copy_tile_init(cb_tanh_exp);
            copy_tile(cb_tanh_exp, 0, 2);
            where_tile_init();
#ifdef TANH_FP32
            where_fp32_tile(0, 1, 2, 0);
#endif
#ifdef TANH_BF16
            where_tile(0, 1, 2, 0);
#endif
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);

            tile_regs_release();

            cb_pop_front(cb_tanh_exp, 1);
            cb_pop_front(cb_tanh_lut, 1);
            cb_pop_front(cb_input, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
