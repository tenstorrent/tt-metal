// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t step = get_arg_val<uint32_t>(0);
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb_param_in = tt::CBIndex::c_0;
    experimental::CircularBuffer cb_param_in_obj(cb_param_in);
    constexpr auto cb_grad_in = tt::CBIndex::c_1;
    experimental::CircularBuffer cb_grad_in_obj(cb_grad_in);
    constexpr auto cb_exp_avg_in = tt::CBIndex::c_2;
    experimental::CircularBuffer cb_exp_avg_in_obj(cb_exp_avg_in);
    constexpr auto cb_exp_avg_sq_in = tt::CBIndex::c_3;
    experimental::CircularBuffer cb_exp_avg_sq_in_obj(cb_exp_avg_sq_in);
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_in = tt::CBIndex::c_4;
    experimental::CircularBuffer cb_max_exp_avg_sq_in_obj(cb_max_exp_avg_sq_in);
#endif
    // lr, beta1, beta2, eps, weight_decay
    constexpr auto cb_scalar_args = tt::CBIndex::c_5;
    experimental::CircularBuffer cb_scalar_args_obj(cb_scalar_args);
    constexpr auto cb_one = tt::CBIndex::c_6;
    experimental::CircularBuffer cb_one_obj(cb_one);
    constexpr auto cb_param_out = tt::CBIndex::c_16;
    constexpr auto cb_exp_avg_out = tt::CBIndex::c_17;
    constexpr auto cb_exp_avg_sq_out = tt::CBIndex::c_18;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_out = tt::CBIndex::c_19;
#endif

    constexpr auto tmp_cb_param = tt::CBIndex::c_24;
    constexpr auto tmp_cb_exp_avg = tt::CBIndex::c_25;
    constexpr auto tmp_cb_exp_avg_sq = tt::CBIndex::c_26;
    experimental::CircularBuffer tmp_cb_exp_avg_sq_obj(tmp_cb_exp_avg_sq);
#ifdef AMSGRAD
    constexpr auto tmp_cb_max_exp_avg_sq = tt::CBIndex::c_27;
    experimental::CircularBuffer tmp_cb_max_exp_avg_sq_obj(tmp_cb_max_exp_avg_sq);
#endif
    constexpr auto cb_beta1_exponent = tt::CBIndex::c_28;
    experimental::CircularBuffer cb_beta1_exponent_obj(cb_beta1_exponent);
    constexpr auto cb_beta2_exponent = tt::CBIndex::c_29;
    experimental::CircularBuffer cb_beta2_exponent_obj(cb_beta2_exponent);
    constexpr auto cb_tmp1 = tt::CBIndex::c_30;
    experimental::CircularBuffer cb_tmp1_obj(cb_tmp1);
    constexpr auto cb_tmp2 = tt::CBIndex::c_31;
    experimental::CircularBuffer cb_tmp2_obj(cb_tmp2);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t beta1_tile = 1;
    constexpr uint32_t beta2_tile = 2;
    constexpr uint32_t eps_tile = 3;
    constexpr uint32_t weight_decay_tile = 4;
    constexpr uint32_t onetile = 1;

    cb_scalar_args_obj.wait_front(5);
    cb_one_obj.wait_front(onetile);
    cb_beta1_exponent_obj.wait_front(onetile);
    cb_beta2_exponent_obj.wait_front(onetile);

    binary_op_init_common(cb_param_in, cb_scalar_args, cb_param_out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb_param_in_obj.wait_front(onetile);
        cb_grad_in_obj.wait_front(onetile);
        cb_exp_avg_in_obj.wait_front(onetile);
        cb_exp_avg_sq_in_obj.wait_front(onetile);
#ifdef AMSGRAD
        cb_max_exp_avg_sq_in_obj.wait_front(onetile);
#endif
        // param = param - lr * weight_decay * param.
        // cb_tmp1 : weight_decay * cb_param_in
        mul_tiles_to_cb(cb_scalar_args, cb_param_in, cb_tmp1, weight_decay_tile, first_tile, /*pop0=*/0, /*pop1=*/0);

        // cb_tmp1 : lr * cb_tmp1
        mul_tiles_to_cb(cb_scalar_args, cb_tmp1, cb_tmp1, lr_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        // tmp_cb_param : cb_param_in - cb_tmp1
        sub_tiles_to_cb(cb_param_in, cb_tmp1, tmp_cb_param, first_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // cb_tmp1 = (1 - beta1)
        sub_tiles_to_cb(cb_one, cb_scalar_args, cb_tmp1, first_tile, beta1_tile, /*pop0=*/0, /*pop1=*/0);

        // cb_tmp1 = cb_grad_in * cb_tmp1
        mul_tiles_to_cb(cb_grad_in, cb_tmp1, cb_tmp1, first_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        // tmp_cb_exp_avg = cb_exp_avg_in * beta1
        mul_tiles_to_cb(cb_exp_avg_in, cb_scalar_args, tmp_cb_exp_avg, first_tile, beta1_tile, /*pop0=*/0, /*pop1=*/0);

        // tmp_cb_exp_avg = tmp_cb_exp_avg + cb_tmp1
        add_tiles_to_cb(tmp_cb_exp_avg, cb_tmp1, tmp_cb_exp_avg, first_tile, first_tile);

        // cb_exp_avg_out
        copy_tile_to_cb(tmp_cb_exp_avg, cb_exp_avg_out, first_tile, /*pop=*/0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        // cb_tmp1 = (1 - beta2)
        tile_regs_acquire();
        cb_tmp1_obj.reserve_back(onetile);
        sub_tiles_init_with_dt(cb_one, cb_scalar_args);
        sub_tiles(cb_one, cb_scalar_args, first_tile, beta2_tile, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_tmp1_obj.push_back(onetile);
        tile_regs_release();

        // cb_tmp2 = grad * grad
        mul_tiles_to_cb(cb_grad_in, cb_grad_in, cb_tmp2, first_tile, first_tile, /*pop0=*/0, /*pop1=*/0);

        // cb_tmp1 = cb_tmp1 * cb_tmp2
        mul_tiles_to_cb(cb_tmp1, cb_tmp2, cb_tmp1, first_tile, first_tile);

        // tmp_cb_exp_avg_sq = cb_exp_avg_sq_in * beta2
        mul_tiles_to_cb(
            cb_exp_avg_sq_in, cb_scalar_args, tmp_cb_exp_avg_sq, first_tile, beta2_tile, /*pop0=*/0, /*pop1=*/0);

        // tmp_cb_exp_avg_sq = tmp_cb_exp_avg_sq + cb_tmp1
        add_tiles_to_cb(tmp_cb_exp_avg_sq, cb_tmp1, tmp_cb_exp_avg_sq, first_tile, first_tile);

        // cb_exp_avg_sq_out
        copy_tile_to_cb(tmp_cb_exp_avg_sq, cb_exp_avg_sq_out, first_tile, /*pop=*/0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // cb_beta2_exponent = pow(beta2, step); Calculated from host

        // cb_tmp1 = 1 / (1 - cb_beta2_exponent);
        tile_regs_acquire();
        cb_tmp1_obj.reserve_back(onetile);
        sub_tiles_init_with_dt(cb_one, cb_beta2_exponent);
        sub_tiles(cb_one, cb_beta2_exponent, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_tmp1_obj.push_back(onetile);
        tile_regs_release();

#ifdef AMSGRAD
        // tmp_cb_max_exp_avg_sq = max(cb_max_exp_avg_sq_in, tmp_cb_exp_avg_sq);
        tile_regs_acquire();
        tmp_cb_max_exp_avg_sq_obj.reserve_back(onetile);
        copy_tile_init_with_dt(cb_max_exp_avg_sq_in);
        copy_tile(cb_max_exp_avg_sq_in, first_tile, dst0);
        copy_tile_init_with_dt(tmp_cb_exp_avg_sq);
        copy_tile(tmp_cb_exp_avg_sq, first_tile, dst1);
        binary_max_tile_init();
        binary_max_tile(dst0, dst1, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, tmp_cb_max_exp_avg_sq);
        tmp_cb_max_exp_avg_sq_obj.push_back(onetile);
        tile_regs_release();

        // cb_max_exp_avg_sq_out
        copy_tile_to_cb(tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out, first_tile, /*pop=*/0);
#endif

        // cb_tmp1 = sqrt(exp_avg_sq / cb_tmp1);
        tile_regs_acquire();
        cb_tmp1_obj.wait_front(onetile);
        cb_tmp1_obj.reserve_back(onetile);
#ifdef AMSGRAD
        mul_tiles_init_with_dt(tmp_cb_max_exp_avg_sq, cb_tmp1);
        mul_tiles(tmp_cb_max_exp_avg_sq, cb_tmp1, first_tile, first_tile, dst0);
#else
        mul_tiles_init_with_dt(tmp_cb_exp_avg_sq, cb_tmp1);
        mul_tiles(tmp_cb_exp_avg_sq, cb_tmp1, first_tile, first_tile, dst0);
#endif
        sqrt_tile_init();
        sqrt_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_tmp1_obj.pop_front(onetile);
        cb_tmp1_obj.push_back(onetile);
#ifdef AMSGRAD
        tmp_cb_max_exp_avg_sq_obj.pop_front(onetile);
#endif
        tmp_cb_exp_avg_sq_obj.pop_front(onetile);
        tile_regs_release();

        // cb_tmp1 = 1 / (cb_tmp1 + eps)
        tile_regs_acquire();
        cb_tmp1_obj.wait_front(onetile);
        cb_tmp1_obj.reserve_back(onetile);
        add_tiles_init_with_dt(cb_tmp1, cb_scalar_args);
        add_tiles(cb_tmp1, cb_scalar_args, first_tile, eps_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_tmp1_obj.pop_front(onetile);
        cb_tmp1_obj.push_back(onetile);
        tile_regs_release();

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_beta1_exponent = pow(beta1, step); Calculated from host

        // cb_tmp2 = 1 / (1 - cb_beta1_exponent);
        tile_regs_acquire();
        cb_tmp2_obj.reserve_back(onetile);
        sub_tiles_init_with_dt(cb_one, cb_beta1_exponent);
        sub_tiles(cb_one, cb_beta1_exponent, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp2);
        cb_tmp2_obj.push_back(onetile);
        tile_regs_release();

        // cb_tmp2 = lr * cb_tmp2;
        mul_tiles_to_cb(cb_scalar_args, cb_tmp2, cb_tmp2, lr_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        // cb_tmp2 = cb_tmp2 * tmp_cb_exp_avg;
        mul_tiles_to_cb(cb_tmp2, tmp_cb_exp_avg, cb_tmp2, first_tile, first_tile);

        // cb_tmp1 = cb_tmp1 * cb_tmp2;
        mul_tiles_to_cb(cb_tmp1, cb_tmp2, cb_tmp1, first_tile, first_tile);

        // param = tmp_cb_param - cb_tmp1;
        sub_tiles_to_cb(tmp_cb_param, cb_tmp1, cb_param_out, first_tile, first_tile);

        cb_param_in_obj.pop_front(onetile);
        cb_grad_in_obj.pop_front(onetile);
        cb_exp_avg_in_obj.pop_front(onetile);
        cb_exp_avg_sq_in_obj.pop_front(onetile);
#ifdef AMSGRAD
        cb_max_exp_avg_sq_in_obj.pop_front(onetile);
#endif
    }

    cb_scalar_args_obj.pop_front(5);
    cb_one_obj.pop_front(onetile);
    cb_beta1_exponent_obj.pop_front(onetile);
    cb_beta2_exponent_obj.pop_front(onetile);
}
