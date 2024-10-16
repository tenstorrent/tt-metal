// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    uint32_t step = get_arg_val<uint32_t>(0);
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb_param_in = tt::CB::c_in0;
    constexpr auto cb_grad_in = tt::CB::c_in1;
    constexpr auto cb_exp_avg_in = tt::CB::c_in2;
    constexpr auto cb_exp_avg_sq_in = tt::CB::c_in3;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_in = tt::CB::c_in4;
#endif
    // lr, beta1, beta2, eps, weight_decay
    constexpr auto cb_scalar_args = tt::CB::c_in5;
    constexpr auto cb_one = tt::CB::c_in6;
    constexpr auto cb_param_out = tt::CB::c_out0;
    constexpr auto cb_exp_avg_out = tt::CB::c_out1;
    constexpr auto cb_exp_avg_sq_out = tt::CB::c_out2;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_out = tt::CB::c_out3;
#endif

    constexpr auto tmp_cb_param = tt::CB::c_intermed0;
    constexpr auto tmp_cb_exp_avg = tt::CB::c_intermed1;
    constexpr auto tmp_cb_exp_avg_sq = tt::CB::c_intermed2;
#ifdef AMSGRAD
    constexpr auto tmp_cb_max_exp_avg_sq = tt::CB::c_intermed3;
#endif
    constexpr auto cb_tmp1 = tt::CB::c_intermed6;
    constexpr auto cb_tmp2 = tt::CB::c_intermed7;

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t beta1_tile = 1;
    constexpr uint32_t beta2_tile = 2;
    constexpr uint32_t eps_tile = 3;
    constexpr uint32_t weight_decay_tile = 4;
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_scalar_args, 5);
    cb_wait_front(cb_one, onetile);

    binary_op_init_common(cb_param_in, cb_scalar_args);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb_wait_front(cb_param_in, onetile);
        cb_wait_front(cb_grad_in, onetile);
        cb_wait_front(cb_exp_avg_in, onetile);
        cb_wait_front(cb_exp_avg_sq_in, onetile);
#ifdef AMSGRAD
        cb_wait_front(cb_max_exp_avg_sq_in, onetile);
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
        cb_reserve_back(cb_tmp1, onetile);
        sub_tiles_init_with_dt(cb_one, cb_scalar_args);
        sub_tiles(cb_one, cb_scalar_args, first_tile, beta2_tile, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_push_back(cb_tmp1, onetile);
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
        // cb_tmp1 = pow(beta2, step);
        tile_regs_acquire();
        cb_reserve_back(cb_tmp1, onetile);
        copy_tile_init_with_dt(cb_scalar_args);
        copy_tile(cb_scalar_args, beta2_tile, dst0);
        power_tile_init();
        power_tile(dst0, step);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_push_back(cb_tmp1, onetile);
        tile_regs_release();

        // cb_tmp1 = 1 / (1 - cb_tmp1);
        tile_regs_acquire();
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        sub_tiles_init_with_dt(cb_one, cb_tmp1);
        sub_tiles(cb_one, cb_tmp1, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_push_back(cb_tmp1, onetile);
        tile_regs_release();

#ifdef AMSGRAD
        // tmp_cb_max_exp_avg_sq = max(cb_max_exp_avg_sq_in, tmp_cb_exp_avg_sq);
        tile_regs_acquire();
        cb_reserve_back(tmp_cb_max_exp_avg_sq, onetile);
        copy_tile_init_with_dt(cb_max_exp_avg_sq_in);
        copy_tile(cb_max_exp_avg_sq_in, first_tile, dst0);
        copy_tile_init_with_dt(tmp_cb_exp_avg_sq);
        copy_tile(tmp_cb_exp_avg_sq, first_tile, dst1);
        max_tile_init();
        max_tile(dst0, dst1);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, tmp_cb_max_exp_avg_sq);
        cb_push_back(tmp_cb_max_exp_avg_sq, onetile);
        tile_regs_release();

        // cb_max_exp_avg_sq_out
        copy_tile_to_cb(tmp_cb_max_exp_avg_sq, cb_max_exp_avg_sq_out, first_tile, /*pop=*/0);
#endif

        // cb_tmp1 = sqrt(exp_avg_sq / cb_tmp1);
        tile_regs_acquire();
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_tmp1, onetile);
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
        cb_pop_front(cb_tmp1, onetile);
        cb_push_back(cb_tmp1, onetile);
#ifdef AMSGRAD
        cb_pop_front(tmp_cb_max_exp_avg_sq, onetile);
#endif
        cb_pop_front(tmp_cb_exp_avg_sq, onetile);
        tile_regs_release();

        // cb_tmp1 = 1 / (cb_tmp1 + eps)
        tile_regs_acquire();
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        add_tiles_init_with_dt(cb_tmp1, cb_scalar_args);
        add_tiles(cb_tmp1, cb_scalar_args, first_tile, eps_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_push_back(cb_tmp1, onetile);
        tile_regs_release();

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_tmp2 = pow(beta1, step);
        tile_regs_acquire();
        cb_reserve_back(cb_tmp2, onetile);
        copy_tile_init_with_dt(cb_scalar_args);
        copy_tile(cb_scalar_args, beta1_tile, dst0);
        power_tile_init();
        power_tile(dst0, step);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp2);
        cb_push_back(cb_tmp2, onetile);
        tile_regs_release();

        // cb_tmp2 = 1 / (1 - cb_tmp2);
        tile_regs_acquire();
        cb_wait_front(cb_tmp2, onetile);
        cb_reserve_back(cb_tmp2, onetile);
        sub_tiles_init_with_dt(cb_one, cb_tmp2);
        sub_tiles(cb_one, cb_tmp2, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp2);
        cb_pop_front(cb_tmp2, onetile);
        cb_push_back(cb_tmp2, onetile);
        tile_regs_release();

        // cb_tmp2 = lr * cb_tmp2;
        mul_tiles_to_cb(cb_scalar_args, cb_tmp2, cb_tmp2, lr_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        // cb_tmp2 = cb_tmp2 * tmp_cb_exp_avg;
        mul_tiles_to_cb(cb_tmp2, tmp_cb_exp_avg, cb_tmp2, first_tile, first_tile);

        // cb_tmp1 = cb_tmp1 * cb_tmp2;
        mul_tiles_to_cb(cb_tmp1, cb_tmp2, cb_tmp1, first_tile, first_tile);

        // param = tmp_cb_param - cb_tmp1;
        sub_tiles_to_cb(tmp_cb_param, cb_tmp1, cb_param_out, first_tile, first_tile);

        cb_pop_front(cb_param_in, onetile);
        cb_pop_front(cb_grad_in, onetile);
        cb_pop_front(cb_exp_avg_in, onetile);
        cb_pop_front(cb_exp_avg_sq_in, onetile);
#ifdef AMSGRAD
        cb_pop_front(cb_max_exp_avg_sq_in, onetile);
#endif
    }

    cb_pop_front(cb_scalar_args, 5);
    cb_pop_front(cb_one, onetile);
}  // void MAIN
}  // namespace NAMESPACE
