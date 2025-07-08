// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/comp.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    constexpr auto cb_input = tt::CBIndex::c_0;   // input
    constexpr auto cb_output = tt::CBIndex::c_2;  // output

    constexpr auto cb_tanh_lut = tt::CBIndex::c_1;   // lut tanh(x)
    constexpr auto cb_exp_2x = tt::CBIndex::c_3;     // exp(2x) and abs(x)
    constexpr auto cb_sub = tt::CBIndex::c_4;        // exp(2x) - 1
    constexpr auto cb_add = tt::CBIndex::c_5;        // recip ( exp(2x) + 1 )
    constexpr auto cb_tanh_exp = tt::CBIndex::c_6;   // tanh[x] = (exp[2x] - 1) / (exp[2x] + 1)
    constexpr auto cb_true_val = tt::CBIndex::c_7;   // output for x > 3.5
    constexpr auto cb_false_val = tt::CBIndex::c_8;  // output for x <= 3.5

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

            copy_tile_to_dst_init_short(cb_input);
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
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);

            mul_unary_tile(0, two);
            exp_tile_init<1u>();
            exp_tile<1u>(0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_exp_2x);
            tile_regs_release();

            cb_push_back(cb_exp_2x, 1);

            // exp(2x) - 1

            cb_reserve_back(cb_sub, 1);
            cb_wait_front(cb_exp_2x, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_exp_2x);
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
            copy_tile_to_dst_init_short(cb_exp_2x);
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

            mul_tiles_init(cb_sub, cb_add);
            mul_tiles(cb_sub, cb_add, 0, 0, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tanh_exp);

            tile_regs_release();

            cb_push_back(cb_tanh_exp, 1);
            cb_pop_front(cb_sub, 1);
            cb_pop_front(cb_add, 1);

            // abs(x) > 3.5f in c_3

            cb_reserve_back(cb_exp_2x, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_input);
            copy_tile(cb_input, 0, 0);

            abs_tile_init();
            abs_tile(0);
            unary_gt_tile_init();
            unary_gt_tile(0, limit);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_exp_2x);

            tile_regs_release();
            cb_push_back(cb_exp_2x, 1);

            cb_pop_front(cb_input, 1);

            // t2 output for x > 3.5

            cb_wait_front(cb_tanh_lut, 1);
            cb_wait_front(cb_exp_2x, 1);
            cb_reserve_back(cb_true_val, 1);

            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_exp_2x);
            copy_tile(cb_exp_2x, 0, 0);

            gtz_tile_init();
            gtz_tile(0);

            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_tanh_lut);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_tanh_lut, 0, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_true_val);

            tile_regs_release();

            cb_push_back(cb_true_val, 1);
            cb_pop_front(cb_tanh_lut, 1);

            // t1 output for x <= 3.5

            cb_wait_front(cb_tanh_exp, 1);
            cb_reserve_back(cb_false_val, 1);

            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_exp_2x);
            copy_tile(cb_exp_2x, 0, 0);

            lez_tile_init();
            lez_tile(0);

            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_tanh_exp);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_tanh_exp, 0, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_false_val);

            tile_regs_release();

            cb_push_back(cb_false_val, 1);
            cb_pop_front(cb_exp_2x, 1);
            cb_pop_front(cb_tanh_exp, 1);

            // out = t1 + t2
            cb_wait_front(cb_false_val, 1);
            cb_wait_front(cb_true_val, 1);

            tile_regs_acquire();

            add_tiles_init(cb_true_val, cb_false_val);
            add_tiles(cb_true_val, cb_false_val, 0, 0, 0);

            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_output);

            tile_regs_release();

            cb_pop_front(cb_true_val, 1);
            cb_pop_front(cb_false_val, 1);
        }
        cb_push_back(cb_output, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
