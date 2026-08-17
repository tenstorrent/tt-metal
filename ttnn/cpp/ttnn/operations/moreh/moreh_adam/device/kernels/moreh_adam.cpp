// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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

#ifdef FP32_DEST_ACC_EN
#define WITH_FP32_DEST_ACC(x) x
#else
#define WITH_FP32_DEST_ACC(x)
#endif

void kernel_main() {
    uint32_t step = get_arg_val<uint32_t>(0);
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb_param_in = tt::CBIndex::c_0;
    constexpr auto cb_grad_in = tt::CBIndex::c_1;
    constexpr auto cb_exp_avg_in = tt::CBIndex::c_2;
    constexpr auto cb_exp_avg_sq_in = tt::CBIndex::c_3;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_in = tt::CBIndex::c_4;
#endif
    // lr, beta1, beta2, eps, weight_decay
    constexpr auto cb_scalar_args = tt::CBIndex::c_5;
    constexpr auto cb_one = tt::CBIndex::c_6;
    constexpr auto cb_param_out = tt::CBIndex::c_16;
    constexpr auto cb_exp_avg_out = tt::CBIndex::c_17;
    constexpr auto cb_exp_avg_sq_out = tt::CBIndex::c_18;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_out = tt::CBIndex::c_19;
#endif

    constexpr auto tmp_cb_grad = tt::CBIndex::c_24;
    constexpr auto tmp_cb_exp_avg = tt::CBIndex::c_25;
    constexpr auto tmp_cb_exp_avg_sq = tt::CBIndex::c_26;
#ifdef AMSGRAD
    constexpr auto tmp_cb_max_exp_avg_sq = tt::CBIndex::c_27;
#endif
    constexpr auto cb_tmp1 = tt::CBIndex::c_30;
    constexpr auto cb_tmp2 = tt::CBIndex::c_31;

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t lr_tile = 0;
    constexpr uint32_t beta1_tile = 1;
    constexpr uint32_t beta2_tile = 2;
    constexpr uint32_t eps_tile = 3;
    constexpr uint32_t weight_decay_tile = 4;
    constexpr uint32_t onetile = 1;

    DataflowBuffer dfb_param_in_obj(cb_param_in);
    DataflowBuffer dfb_grad_in_obj(cb_grad_in);
    DataflowBuffer dfb_exp_avg_in_obj(cb_exp_avg_in);
    DataflowBuffer dfb_exp_avg_sq_in_obj(cb_exp_avg_sq_in);
#ifdef AMSGRAD
    DataflowBuffer dfb_max_exp_avg_sq_in_obj(cb_max_exp_avg_sq_in);
    DataflowBuffer tmp_dfb_max_exp_avg_sq_obj(tmp_cb_max_exp_avg_sq);
    DataflowBuffer dfb_max_exp_avg_sq_out_obj(cb_max_exp_avg_sq_out);
#endif
    DataflowBuffer dfb_scalar_args_obj(cb_scalar_args);
    DataflowBuffer dfb_one_obj(cb_one);
    DataflowBuffer dfb_param_out_obj(cb_param_out);
    DataflowBuffer dfb_exp_avg_out_obj(cb_exp_avg_out);
    DataflowBuffer dfb_exp_avg_sq_out_obj(cb_exp_avg_sq_out);
    DataflowBuffer tmp_dfb_grad_obj(tmp_cb_grad);
    DataflowBuffer tmp_dfb_exp_avg_obj(tmp_cb_exp_avg);
    DataflowBuffer dfb_tmp1_obj(cb_tmp1);
    DataflowBuffer dfb_tmp2_obj(cb_tmp2);
    DataflowBuffer tmp_dfb_exp_avg_sq_obj(tmp_cb_exp_avg_sq);

    dfb_scalar_args_obj.wait_front(5);
    dfb_one_obj.wait_front(onetile);

    compute_kernel_hw_startup(cb_param_in, cb_scalar_args, cb_param_out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // grad += grad + param * weight_decay;
        // cb_tmp1 : param * weight_decay;
        dfb_param_in_obj.wait_front(onetile);
        dfb_grad_in_obj.wait_front(onetile);
        dfb_exp_avg_in_obj.wait_front(onetile);
        dfb_exp_avg_sq_in_obj.wait_front(onetile);
#ifdef AMSGRAD
        dfb_max_exp_avg_sq_in_obj.wait_front(onetile);
#endif
        // cb_tmp1 : param * weight_decay;
        mul_tiles_to_cb(dfb_param_in_obj, dfb_scalar_args_obj, dfb_tmp1_obj, first_tile, weight_decay_tile, 0, 0);

        // tmp_cb_grad : cb_grad_in + cb_tmp1;
        add_tiles_to_cb(dfb_grad_in_obj, dfb_tmp1_obj, tmp_dfb_grad_obj, first_tile, first_tile, 0, 1);

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // cb_tmp1 = (1 - beta1)
        sub_tiles_to_cb(dfb_one_obj, dfb_scalar_args_obj, dfb_tmp1_obj, first_tile, beta1_tile, 0, 0);
        mul_tiles_to_cb(tmp_dfb_grad_obj, dfb_tmp1_obj, dfb_tmp1_obj, first_tile, first_tile, 0, 1);

        // tmp_cb_exp_avg = cb_exp_avg_in * beta1
        mul_tiles_to_cb(dfb_exp_avg_in_obj, dfb_scalar_args_obj, tmp_dfb_exp_avg_obj, first_tile, beta1_tile, 0, 0);

        // tmp_cb_exp_avg = tmp_cb_exp_avg + cb_tmp1
        add_tiles_to_cb(tmp_dfb_exp_avg_obj, dfb_tmp1_obj, tmp_dfb_exp_avg_obj, first_tile, first_tile, 1, 1);

        // cb_exp_avg_out
        copy_tile_to_cb(tmp_dfb_exp_avg_obj, dfb_exp_avg_out_obj, first_tile, 0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        sub_tiles_to_cb(dfb_one_obj, dfb_scalar_args_obj, dfb_tmp1_obj, first_tile, beta2_tile, 0, 0);

        // cb_tmp2 = grad * grad
        mul_tiles_to_cb(tmp_dfb_grad_obj, tmp_dfb_grad_obj, dfb_tmp2_obj, first_tile, first_tile, 1, 0);

        // cb_tmp1 = cb_tmp1 * cb_tmp2
        mul_tiles_to_cb(dfb_tmp1_obj, dfb_tmp2_obj, dfb_tmp1_obj, first_tile, first_tile, 1, 1);

        // tmp_cb_exp_avg_sq = cb_exp_avg_sq_in * beta2
        mul_tiles_to_cb(
            dfb_exp_avg_sq_in_obj, dfb_scalar_args_obj, tmp_dfb_exp_avg_sq_obj, first_tile, beta2_tile, 0, 0);

        // tmp_cb_exp_avg_sq = tmp_cb_exp_avg_sq + cb_tmp1
        add_tiles_to_cb(tmp_dfb_exp_avg_sq_obj, dfb_tmp1_obj, tmp_dfb_exp_avg_sq_obj, first_tile, first_tile, 1, 1);

        // cb_exp_avg_sq_out
        copy_tile_to_cb(tmp_dfb_exp_avg_sq_obj, dfb_exp_avg_sq_out_obj, first_tile, 0);
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // cb_tmp1 = pow(beta2, step);
        tile_regs_acquire();
        copy_tile_init_with_dt(dfb_scalar_args_obj);
        copy_tile(cb_scalar_args, beta2_tile, dst0);
        power_tile_init();
        power_tile(dst0, step);
        tile_regs_commit();

        tile_regs_wait();
        dfb_tmp1_obj.reserve_back(onetile);
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        dfb_tmp1_obj.push_back(onetile);
        tile_regs_release();

        // cb_tmp1 = 1 / (1 - cb_tmp1);
        tile_regs_acquire();
        dfb_tmp1_obj.wait_front(onetile);
        dfb_tmp1_obj.reserve_back(onetile);
        WITH_FP32_DEST_ACC(reconfig_data_format(cb_one, cb_tmp1));
        sub_init(cb_one, cb_tmp1);
        sub_tiles(cb_one, cb_tmp1, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        dfb_tmp1_obj.pop_front(onetile);
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
        tile_regs_acquire();
        tmp_dfb_max_exp_avg_sq_obj.wait_front(onetile);
        dfb_max_exp_avg_sq_out_obj.reserve_back(onetile);
        copy_tile_init_with_dt(tmp_dfb_max_exp_avg_sq_obj);
        copy_tile(tmp_cb_max_exp_avg_sq, first_tile, dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_max_exp_avg_sq_out_obj);
        dfb_max_exp_avg_sq_out_obj.push_back(onetile);
        tile_regs_release();
#endif

        // cb_tmp1 = sqrt(exp_avg_sq / cb_tmp1);
        tile_regs_acquire();
        dfb_tmp1_obj.wait_front(onetile);
        dfb_tmp1_obj.reserve_back(onetile);

#ifdef AMSGRAD
        mul_init(tmp_cb_max_exp_avg_sq, cb_tmp1);
        WITH_FP32_DEST_ACC(reconfig_data_format(tmp_cb_max_exp_avg_sq, cb_tmp1));
        mul_tiles(tmp_cb_max_exp_avg_sq, cb_tmp1, first_tile, first_tile, dst0);
#else
        mul_init(tmp_cb_exp_avg_sq, cb_tmp1);
        WITH_FP32_DEST_ACC(reconfig_data_format(tmp_cb_exp_avg_sq, cb_tmp1));
        mul_tiles(tmp_cb_exp_avg_sq, cb_tmp1, first_tile, first_tile, dst0);
#endif
        sqrt_tile_init();
        sqrt_tile(dst0);
        pack_tile_with_dt(dst0, dfb_tmp1_obj);
        tile_regs_commit();

        tile_regs_wait();
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
        WITH_FP32_DEST_ACC(reconfig_data_format(cb_tmp1, cb_scalar_args));
        add_init(cb_tmp1, cb_scalar_args);
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
        // cb_tmp2 = pow(beta1, step);
        tile_regs_acquire();
        dfb_tmp2_obj.reserve_back(onetile);
        copy_tile_init_with_dt(dfb_scalar_args_obj);
        copy_tile(cb_scalar_args, beta1_tile, dst0);
        power_tile_init();
        power_tile(dst0, step);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp2_obj);
        dfb_tmp2_obj.push_back(onetile);
        tile_regs_release();

        // cb_tmp2 = 1 / (1 - cb_tmp2);
        tile_regs_acquire();
        dfb_tmp2_obj.wait_front(onetile);
        WITH_FP32_DEST_ACC(reconfig_data_format(cb_one, cb_tmp2));
        sub_init(cb_one, cb_tmp2);
        sub_tiles(cb_one, cb_tmp2, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        dfb_tmp2_obj.pop_front(onetile);
        tile_regs_commit();

        tile_regs_wait();
        dfb_tmp2_obj.reserve_back(onetile);
        pack_tile_with_dt(dst0, dfb_tmp2_obj);
        dfb_tmp2_obj.push_back(onetile);
        tile_regs_release();

        // cb_tmp2 = lr * cb_tmp2;
        mul_tiles_to_cb(dfb_scalar_args_obj, dfb_tmp2_obj, dfb_tmp2_obj, lr_tile, first_tile, 0, 1);

        // cb_tmp2 = cb_tmp2 * tmp_cb_exp_avg;
        mul_tiles_to_cb(dfb_tmp2_obj, tmp_dfb_exp_avg_obj, dfb_tmp2_obj, first_tile, first_tile, 1, 1);

        // cb_tmp1 = cb_tmp1 * cb_tmp2;
        mul_tiles_to_cb(dfb_tmp1_obj, dfb_tmp2_obj, dfb_tmp1_obj, first_tile, first_tile, 1, 1);

        // param = param - cb_tmp1;
        sub_tiles_to_cb(dfb_param_in_obj, dfb_tmp1_obj, dfb_param_out_obj, first_tile, first_tile, 0, 1);

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
}
