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

void kernel_main() {
    uint32_t step = get_arg_val<uint32_t>(0);
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb_param_in = tt::CBIndex::c_0;
    DataflowBuffer dfb_param_in_obj(cb_param_in);
    constexpr auto cb_grad_in = tt::CBIndex::c_1;
    DataflowBuffer dfb_grad_in_obj(cb_grad_in);
    constexpr auto cb_exp_avg_in = tt::CBIndex::c_2;
    DataflowBuffer dfb_exp_avg_in_obj(cb_exp_avg_in);
    constexpr auto cb_exp_avg_sq_in = tt::CBIndex::c_3;
    DataflowBuffer dfb_exp_avg_sq_in_obj(cb_exp_avg_sq_in);
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_in = tt::CBIndex::c_4;
    DataflowBuffer dfb_max_exp_avg_sq_in_obj(cb_max_exp_avg_sq_in);
#endif
    // lr, beta1, beta2, eps, weight_decay
    constexpr auto cb_scalar_args = tt::CBIndex::c_5;
    DataflowBuffer dfb_scalar_args_obj(cb_scalar_args);
    constexpr auto cb_one = tt::CBIndex::c_6;
    DataflowBuffer dfb_one_obj(cb_one);
    constexpr auto cb_param_out = tt::CBIndex::c_16;
    DataflowBuffer dfb_param_out_obj(cb_param_out);
    constexpr auto cb_exp_avg_out = tt::CBIndex::c_17;
    DataflowBuffer dfb_exp_avg_out_obj(cb_exp_avg_out);
    constexpr auto cb_exp_avg_sq_out = tt::CBIndex::c_18;
    DataflowBuffer dfb_exp_avg_sq_out_obj(cb_exp_avg_sq_out);
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_out = tt::CBIndex::c_19;
    DataflowBuffer dfb_max_exp_avg_sq_out_obj(cb_max_exp_avg_sq_out);
#endif

    constexpr auto tmp_cb_param = tt::CBIndex::c_24;
    DataflowBuffer tmp_dfb_param_obj(tmp_cb_param);
    constexpr auto tmp_cb_exp_avg = tt::CBIndex::c_25;
    DataflowBuffer tmp_dfb_exp_avg_obj(tmp_cb_exp_avg);
    constexpr auto tmp_cb_exp_avg_sq = tt::CBIndex::c_26;
    DataflowBuffer tmp_dfb_exp_avg_sq_obj(tmp_cb_exp_avg_sq);
#ifdef AMSGRAD
    constexpr auto tmp_cb_max_exp_avg_sq = tt::CBIndex::c_27;
    DataflowBuffer tmp_dfb_max_exp_avg_sq_obj(tmp_cb_max_exp_avg_sq);
#endif
    constexpr auto cb_beta1_exponent = tt::CBIndex::c_28;
    DataflowBuffer dfb_beta1_exponent_obj(cb_beta1_exponent);
    constexpr auto cb_beta2_exponent = tt::CBIndex::c_29;
    DataflowBuffer dfb_beta2_exponent_obj(cb_beta2_exponent);
    constexpr auto cb_tmp1 = tt::CBIndex::c_30;
    DataflowBuffer dfb_tmp1_obj(cb_tmp1);
    constexpr auto cb_tmp2 = tt::CBIndex::c_31;
    DataflowBuffer dfb_tmp2_obj(cb_tmp2);

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

    binary_op_init_common(cb_param_in, cb_scalar_args, cb_param_out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        dfb_param_in_obj.wait_front(onetile);
        dfb_grad_in_obj.wait_front(onetile);
        dfb_exp_avg_in_obj.wait_front(onetile);
        dfb_exp_avg_sq_in_obj.wait_front(onetile);
#ifdef AMSGRAD
        dfb_max_exp_avg_sq_in_obj.wait_front(onetile);
#endif
        // param = param - lr * weight_decay * param.
        // cb_tmp1 : weight_decay * cb_param_in
        mul_tiles_to_cb(
            dfb_scalar_args_obj, dfb_param_in_obj, dfb_tmp1_obj, weight_decay_tile, first_tile, /*pop0=*/0, /*pop1=*/0);

        // cb_tmp1 : lr * cb_tmp1
        mul_tiles_to_cb(dfb_scalar_args_obj, dfb_tmp1_obj, dfb_tmp1_obj, lr_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        // tmp_cb_param : cb_param_in - cb_tmp1
        sub_tiles_to_cb(
            dfb_param_in_obj, dfb_tmp1_obj, tmp_dfb_param_obj, first_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // cb_tmp1 = (1 - beta1)
        sub_tiles_to_cb(dfb_one_obj, dfb_scalar_args_obj, dfb_tmp1_obj, first_tile, beta1_tile, /*pop0=*/0, /*pop1=*/0);

        // cb_tmp1 = cb_grad_in * cb_tmp1
        mul_tiles_to_cb(dfb_grad_in_obj, dfb_tmp1_obj, dfb_tmp1_obj, first_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        // tmp_cb_exp_avg = cb_exp_avg_in * beta1
        mul_tiles_to_cb(
            dfb_exp_avg_in_obj,
            dfb_scalar_args_obj,
            tmp_dfb_exp_avg_obj,
            first_tile,
            beta1_tile,
            /*pop0=*/0,
            /*pop1=*/0);

        // tmp_cb_exp_avg = tmp_cb_exp_avg + cb_tmp1
        add_tiles_to_cb(tmp_dfb_exp_avg_obj, dfb_tmp1_obj, tmp_dfb_exp_avg_obj, first_tile, first_tile);

        // cb_exp_avg_out
        copy_tile_to_cb(tmp_dfb_exp_avg_obj, dfb_exp_avg_out_obj, first_tile, /*pop=*/0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        // cb_tmp1 = (1 - beta2)
        tile_regs_acquire();
        dfb_tmp1_obj.reserve_back(onetile);
        sub_tiles_init_with_dt(dfb_one_obj, dfb_scalar_args_obj);
        sub_tiles(cb_one, cb_scalar_args, first_tile, beta2_tile, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        dfb_tmp1_obj.push_back(onetile);
        tile_regs_release();

        // cb_tmp2 = grad * grad
        mul_tiles_to_cb(dfb_grad_in_obj, dfb_grad_in_obj, dfb_tmp2_obj, first_tile, first_tile, /*pop0=*/0, /*pop1=*/0);

        // cb_tmp1 = cb_tmp1 * cb_tmp2
        mul_tiles_to_cb(dfb_tmp1_obj, dfb_tmp2_obj, dfb_tmp1_obj, first_tile, first_tile);

        // tmp_cb_exp_avg_sq = cb_exp_avg_sq_in * beta2
        mul_tiles_to_cb(
            dfb_exp_avg_sq_in_obj,
            dfb_scalar_args_obj,
            tmp_dfb_exp_avg_sq_obj,
            first_tile,
            beta2_tile,
            /*pop0=*/0,
            /*pop1=*/0);

        // tmp_cb_exp_avg_sq = tmp_cb_exp_avg_sq + cb_tmp1
        add_tiles_to_cb(tmp_dfb_exp_avg_sq_obj, dfb_tmp1_obj, tmp_dfb_exp_avg_sq_obj, first_tile, first_tile);

        // cb_exp_avg_sq_out
        copy_tile_to_cb(tmp_dfb_exp_avg_sq_obj, dfb_exp_avg_sq_out_obj, first_tile, /*pop=*/0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // cb_beta2_exponent = pow(beta2, step); Calculated from host

        // cb_tmp1 = 1 / (1 - cb_beta2_exponent);
        tile_regs_acquire();
        dfb_tmp1_obj.reserve_back(onetile);
        sub_tiles_init_with_dt(dfb_one_obj, dfb_beta2_exponent_obj);
        sub_tiles(cb_one, cb_beta2_exponent, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        dfb_tmp1_obj.push_back(onetile);
        tile_regs_release();

#ifdef AMSGRAD
        // tmp_cb_max_exp_avg_sq = max(cb_max_exp_avg_sq_in, tmp_cb_exp_avg_sq);
        tile_regs_acquire();
        tmp_dfb_max_exp_avg_sq_obj.reserve_back(onetile);
        copy_tile_init_with_dt(dfb_max_exp_avg_sq_in_obj);
        copy_tile(cb_max_exp_avg_sq_in, first_tile, dst0);
        copy_tile_init_with_dt(tmp_dfb_exp_avg_sq_obj);
        copy_tile(tmp_cb_exp_avg_sq, first_tile, dst1);
        binary_max_tile_init();
        binary_max_tile(dst0, dst1, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, tmp_dfb_max_exp_avg_sq_obj);
        tmp_dfb_max_exp_avg_sq_obj.push_back(onetile);
        tile_regs_release();

        // cb_max_exp_avg_sq_out
        copy_tile_to_cb(tmp_dfb_max_exp_avg_sq_obj, dfb_max_exp_avg_sq_out_obj, first_tile, /*pop=*/0);
#endif

        // cb_tmp1 = sqrt(exp_avg_sq / cb_tmp1);
        tile_regs_acquire();
        dfb_tmp1_obj.wait_front(onetile);
        dfb_tmp1_obj.reserve_back(onetile);
#ifdef AMSGRAD
        mul_tiles_init_with_dt(tmp_dfb_max_exp_avg_sq_obj, dfb_tmp1_obj);
        mul_tiles(tmp_cb_max_exp_avg_sq, cb_tmp1, first_tile, first_tile, dst0);
#else
        mul_tiles_init_with_dt(tmp_dfb_exp_avg_sq_obj, dfb_tmp1_obj);
        mul_tiles(tmp_cb_exp_avg_sq, cb_tmp1, first_tile, first_tile, dst0);
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

        // cb_tmp1 = 1 / (cb_tmp1 + eps)
        tile_regs_acquire();
        dfb_tmp1_obj.wait_front(onetile);
        dfb_tmp1_obj.reserve_back(onetile);
        add_tiles_init_with_dt(dfb_tmp1_obj, dfb_scalar_args_obj);
        add_tiles(cb_tmp1, cb_scalar_args, first_tile, eps_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        dfb_tmp1_obj.pop_front(onetile);
        dfb_tmp1_obj.push_back(onetile);
        tile_regs_release();

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_beta1_exponent = pow(beta1, step); Calculated from host

        // cb_tmp2 = 1 / (1 - cb_beta1_exponent);
        tile_regs_acquire();
        dfb_tmp2_obj.reserve_back(onetile);
        sub_tiles_init_with_dt(dfb_one_obj, dfb_beta1_exponent_obj);
        sub_tiles(cb_one, cb_beta1_exponent, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp2_obj);
        dfb_tmp2_obj.push_back(onetile);
        tile_regs_release();

        // cb_tmp2 = lr * cb_tmp2;
        mul_tiles_to_cb(dfb_scalar_args_obj, dfb_tmp2_obj, dfb_tmp2_obj, lr_tile, first_tile, /*pop0=*/0, /*pop1=*/1);

        // cb_tmp2 = cb_tmp2 * tmp_cb_exp_avg;
        mul_tiles_to_cb(dfb_tmp2_obj, tmp_dfb_exp_avg_obj, dfb_tmp2_obj, first_tile, first_tile);

        // cb_tmp1 = cb_tmp1 * cb_tmp2;
        mul_tiles_to_cb(dfb_tmp1_obj, dfb_tmp2_obj, dfb_tmp1_obj, first_tile, first_tile);

        // param = tmp_cb_param - cb_tmp1;
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
