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
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t step = get_arg(args::step);
    constexpr auto per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    DataflowBuffer dfb_param_in_obj(dfb::param_in);
    DataflowBuffer dfb_grad_in_obj(dfb::grad);
    DataflowBuffer dfb_exp_avg_in_obj(dfb::exp_avg_in);
    DataflowBuffer dfb_exp_avg_sq_in_obj(dfb::exp_avg_sq_in);
#ifdef AMSGRAD
    DataflowBuffer dfb_max_exp_avg_sq_in_obj(dfb::max_exp_avg_sq_in);
#endif
    // lr, beta1, beta2, eps, weight_decay
    DataflowBuffer dfb_scalar_args_obj(dfb::scalar_args);
    DataflowBuffer dfb_one_obj(dfb::one);
    DataflowBuffer dfb_param_out_obj(dfb::param_out);
    DataflowBuffer dfb_exp_avg_out_obj(dfb::exp_avg_out);
    DataflowBuffer dfb_exp_avg_sq_out_obj(dfb::exp_avg_sq_out);
#ifdef AMSGRAD
    DataflowBuffer dfb_max_exp_avg_sq_out_obj(dfb::max_exp_avg_sq_out);
#endif

    DataflowBuffer tmp_dfb_param_obj(dfb::tmp_param);
    DataflowBuffer tmp_dfb_exp_avg_obj(dfb::tmp_exp_avg);
    DataflowBuffer tmp_dfb_exp_avg_sq_obj(dfb::tmp_exp_avg_sq);
#ifdef AMSGRAD
    DataflowBuffer tmp_dfb_max_exp_avg_sq_obj(dfb::tmp_max_exp_avg_sq);
#endif
    DataflowBuffer dfb_beta1_exponent_obj(dfb::beta1_exponent);
    DataflowBuffer dfb_beta2_exponent_obj(dfb::beta2_exponent);
    DataflowBuffer dfb_tmp1_obj(dfb::tmp1);
    DataflowBuffer dfb_tmp2_obj(dfb::tmp2);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t beta1_tile = 1;
    constexpr uint32_t beta2_tile = 2;
    constexpr uint32_t eps_tile = 3;
    constexpr uint32_t weight_decay_tile = 4;
    constexpr uint32_t onetile = 1;

    dfb_scalar_args_obj.wait_front(5);
    dfb_one_obj.wait_front(onetile);
    dfb_beta1_exponent_obj.wait_front(onetile);
    dfb_beta2_exponent_obj.wait_front(onetile);

    binary_op_init_common(dfb::param_in, dfb::scalar_args, dfb::param_out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        dfb_param_in_obj.wait_front(onetile);
        dfb_grad_in_obj.wait_front(onetile);
        dfb_exp_avg_in_obj.wait_front(onetile);
        dfb_exp_avg_sq_in_obj.wait_front(onetile);
#ifdef AMSGRAD
        dfb_max_exp_avg_sq_in_obj.wait_front(onetile);
#endif
        // param = param - lr * weight_decay * param.
        // tmp1 : weight_decay * param_in
        mul_tiles_to_cb(
            dfb_scalar_args_obj, dfb_param_in_obj, dfb_tmp1_obj, weight_decay_tile, first_tile, /*pop0=*/0, /*pop1=*/0);

        // tmp1 : lr * tmp1
        mul_tiles_to_cb(dfb_scalar_args_obj, dfb_tmp1_obj, dfb_tmp1_obj, lr_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        // tmp_param : param_in - tmp1
        sub_tiles_to_cb(
            dfb_param_in_obj, dfb_tmp1_obj, tmp_dfb_param_obj, first_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // tmp1 = (1 - beta1)
        sub_tiles_to_cb(dfb_one_obj, dfb_scalar_args_obj, dfb_tmp1_obj, first_tile, beta1_tile, /*pop0=*/0, /*pop1=*/0);

        // tmp1 = grad * tmp1
        mul_tiles_to_cb(dfb_grad_in_obj, dfb_tmp1_obj, dfb_tmp1_obj, first_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        // tmp_exp_avg = exp_avg_in * beta1
        mul_tiles_to_cb(
            dfb_exp_avg_in_obj,
            dfb_scalar_args_obj,
            tmp_dfb_exp_avg_obj,
            first_tile,
            beta1_tile,
            /*pop0=*/0,
            /*pop1=*/0);

        // tmp_exp_avg = tmp_exp_avg + tmp1
        add_tiles_to_cb(tmp_dfb_exp_avg_obj, dfb_tmp1_obj, tmp_dfb_exp_avg_obj, first_tile, first_tile);

        // exp_avg_out
        copy_tile_to_cb(tmp_dfb_exp_avg_obj, dfb_exp_avg_out_obj, first_tile, /*pop=*/0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        // tmp1 = (1 - beta2)
        tile_regs_acquire();
        dfb_tmp1_obj.reserve_back(onetile);
        sub_tiles_init_with_dt(dfb_one_obj, dfb_scalar_args_obj);
        sub_tiles(dfb::one, dfb::scalar_args, first_tile, beta2_tile, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        dfb_tmp1_obj.push_back(onetile);
        tile_regs_release();

        // tmp2 = grad * grad
        mul_tiles_to_cb(dfb_grad_in_obj, dfb_grad_in_obj, dfb_tmp2_obj, first_tile, first_tile, /*pop0=*/0, /*pop1=*/0);

        // tmp1 = tmp1 * tmp2
        mul_tiles_to_cb(dfb_tmp1_obj, dfb_tmp2_obj, dfb_tmp1_obj, first_tile, first_tile);

        // tmp_exp_avg_sq = exp_avg_sq_in * beta2
        mul_tiles_to_cb(
            dfb_exp_avg_sq_in_obj,
            dfb_scalar_args_obj,
            tmp_dfb_exp_avg_sq_obj,
            first_tile,
            beta2_tile,
            /*pop0=*/0,
            /*pop1=*/0);

        // tmp_exp_avg_sq = tmp_exp_avg_sq + tmp1
        add_tiles_to_cb(tmp_dfb_exp_avg_sq_obj, dfb_tmp1_obj, tmp_dfb_exp_avg_sq_obj, first_tile, first_tile);

        // exp_avg_sq_out
        copy_tile_to_cb(tmp_dfb_exp_avg_sq_obj, dfb_exp_avg_sq_out_obj, first_tile, /*pop=*/0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // beta2_exponent = pow(beta2, step); Calculated from host

        // tmp1 = 1 / (1 - beta2_exponent);
        tile_regs_acquire();
        dfb_tmp1_obj.reserve_back(onetile);
        sub_tiles_init_with_dt(dfb_one_obj, dfb_beta2_exponent_obj);
        sub_tiles(dfb::one, dfb::beta2_exponent, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        dfb_tmp1_obj.push_back(onetile);
        tile_regs_release();

#ifdef AMSGRAD
        // tmp_max_exp_avg_sq = max(max_exp_avg_sq_in, tmp_exp_avg_sq);
        tile_regs_acquire();
        tmp_dfb_max_exp_avg_sq_obj.reserve_back(onetile);
        copy_tile_init_with_dt(dfb_max_exp_avg_sq_in_obj);
        copy_tile(dfb::max_exp_avg_sq_in, first_tile, dst0);
        copy_tile_init_with_dt(tmp_dfb_exp_avg_sq_obj);
        copy_tile(dfb::tmp_exp_avg_sq, first_tile, dst1);
        binary_max_tile_init();
        binary_max_tile(dst0, dst1, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, tmp_dfb_max_exp_avg_sq_obj);
        tmp_dfb_max_exp_avg_sq_obj.push_back(onetile);
        tile_regs_release();

        // max_exp_avg_sq_out
        copy_tile_to_cb(tmp_dfb_max_exp_avg_sq_obj, dfb_max_exp_avg_sq_out_obj, first_tile, /*pop=*/0);
#endif

        // tmp1 = sqrt(exp_avg_sq / tmp1);
        tile_regs_acquire();
        dfb_tmp1_obj.wait_front(onetile);
        dfb_tmp1_obj.reserve_back(onetile);
#ifdef AMSGRAD
        mul_tiles_init_with_dt(tmp_dfb_max_exp_avg_sq_obj, dfb_tmp1_obj);
        mul_tiles(dfb::tmp_max_exp_avg_sq, dfb::tmp1, first_tile, first_tile, dst0);
#else
        mul_tiles_init_with_dt(tmp_dfb_exp_avg_sq_obj, dfb_tmp1_obj);
        mul_tiles(dfb::tmp_exp_avg_sq, dfb::tmp1, first_tile, first_tile, dst0);
#endif
        sqrt_tile_init();
        sqrt_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        dfb_tmp1_obj.pop_front(onetile);
        dfb_tmp1_obj.push_back(onetile);
#ifdef AMSGRAD
        tmp_dfb_max_exp_avg_sq_obj.pop_front(onetile);
#endif
        tmp_dfb_exp_avg_sq_obj.pop_front(onetile);
        tile_regs_release();

        // tmp1 = 1 / (tmp1 + eps)
        tile_regs_acquire();
        dfb_tmp1_obj.wait_front(onetile);
        dfb_tmp1_obj.reserve_back(onetile);
        add_tiles_init_with_dt(dfb_tmp1_obj, dfb_scalar_args_obj);
        add_tiles(dfb::tmp1, dfb::scalar_args, first_tile, eps_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        dfb_tmp1_obj.pop_front(onetile);
        dfb_tmp1_obj.push_back(onetile);
        tile_regs_release();

        // bias_correction1 = 1 - pow(beta1, step);
        // beta1_exponent = pow(beta1, step); Calculated from host

        // tmp2 = 1 / (1 - beta1_exponent);
        tile_regs_acquire();
        dfb_tmp2_obj.reserve_back(onetile);
        sub_tiles_init_with_dt(dfb_one_obj, dfb_beta1_exponent_obj);
        sub_tiles(dfb::one, dfb::beta1_exponent, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp2_obj);
        dfb_tmp2_obj.push_back(onetile);
        tile_regs_release();

        // tmp2 = lr * tmp2;
        mul_tiles_to_cb(dfb_scalar_args_obj, dfb_tmp2_obj, dfb_tmp2_obj, lr_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        // tmp2 = tmp2 * tmp_exp_avg;
        mul_tiles_to_cb(dfb_tmp2_obj, tmp_dfb_exp_avg_obj, dfb_tmp2_obj, first_tile, first_tile);

        // tmp1 = tmp1 * tmp2;
        mul_tiles_to_cb(dfb_tmp1_obj, dfb_tmp2_obj, dfb_tmp1_obj, first_tile, first_tile);

        // param = tmp_param - tmp1;
        sub_tiles_to_cb(tmp_dfb_param_obj, dfb_tmp1_obj, dfb_param_out_obj, first_tile, first_tile);

        dfb_param_in_obj.pop_front(onetile);
        dfb_grad_in_obj.pop_front(onetile);
        dfb_exp_avg_in_obj.pop_front(onetile);
        dfb_exp_avg_sq_in_obj.pop_front(onetile);
#ifdef AMSGRAD
        dfb_max_exp_avg_sq_in_obj.pop_front(onetile);
#endif
    }

    dfb_scalar_args_obj.pop_front(5);
    dfb_one_obj.pop_front(onetile);
    dfb_beta1_exponent_obj.pop_front(onetile);
    dfb_beta2_exponent_obj.pop_front(onetile);
}
