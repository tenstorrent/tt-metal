// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // compile-time args
    constexpr auto num_output_tiles = get_arg(args::num_output_tiles);
    constexpr bool wt_need_bcast = (get_arg(args::wt_need_bcast) == 1);
    constexpr bool ht_need_bcast = (get_arg(args::ht_need_bcast) == 1);

    // runtime args
    const auto num_input_tiles_per_core = get_arg(args::num_input_tiles_per_core);
    const auto p = get_arg(args::p);
    const bool p_is_negative = get_arg(args::p_is_negative) == 1;
    const auto p_minus_one = get_arg(args::p_minus_one);
    const bool p_minus_one_is_negative = get_arg(args::p_minus_one_is_negative) == 1;

    DataflowBuffer dfb_x_obj(dfb::x);              // input(==x), c_0
    DataflowBuffer dfb_y_obj(dfb::y);              // output(==y), c_1
    DataflowBuffer dfb_dy_obj(dfb::dy);            // output_grad(==dy), c_2
    DataflowBuffer dfb_decimal_obj(dfb::decimal);  // decimal, c_3

    DataflowBuffer dfb_dx_obj(dfb::dx);  // input_grad(==dx), c_16

    // Compute-only intermediates (c_24..c_31), each filled and drained by this kernel (self-loop).
    DataflowBuffer dfb_xpow_obj(dfb::xpow);
    DataflowBuffer dfb_logx_obj(dfb::logx);
    DataflowBuffer dfb_exp_lxmd_obj(dfb::exp_lxmd);
    DataflowBuffer dfb_correct_xpow_obj(dfb::correct_xpow);
    DataflowBuffer dfb_tmp4_obj(dfb::tmp4);
    DataflowBuffer dfb_tmp5_obj(dfb::tmp5);
    DataflowBuffer dfb_recip_ypow_obj(dfb::recip_ypow);
    DataflowBuffer dfb_sign_obj(dfb::sign);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::dx);
    dfb_decimal_obj.wait_front(onetile);  // comes from the reader

    for (uint32_t idx = 0; idx < num_input_tiles_per_core; ++idx) {
        dfb_x_obj.wait_front(onetile);   // comes from the reader
        dfb_y_obj.wait_front(onetile);   // comes from the reader
        dfb_dy_obj.wait_front(onetile);  // comes from the reader

        sign_tile_to_cb(dfb_x_obj, dfb_sign_obj, 0, /*pop=*/0);

        // x^(p - 1)
        power_tile_with_abs_x_to_cb(
            dfb_x_obj,
            dfb_xpow_obj,
            dfb_logx_obj,
            dfb_decimal_obj,
            dfb_exp_lxmd_obj,
            dfb_correct_xpow_obj,
            p_minus_one,
            p_minus_one_is_negative);

        // x^(p - 1) * y -> cb_tmp4
        dfb_correct_xpow_obj.wait_front(onetile);
        dfb_tmp4_obj.reserve_back(onetile);

        tile_regs_acquire();
        if (ht_need_bcast && wt_need_bcast) {
            mul_bcast_scalar_init_with_dt(dfb_correct_xpow_obj, dfb_y_obj);
            mul_tiles_bcast_scalar(dfb::correct_xpow, dfb::y, 0, 0, dst0);
        } else if (ht_need_bcast) {
            mul_bcast_rows_init_with_dt(dfb_correct_xpow_obj, dfb_y_obj);
            mul_tiles_bcast_rows(dfb::correct_xpow, dfb::y, 0, 0, dst0);
        } else if (wt_need_bcast) {
            mul_bcast_cols_init_with_dt(dfb_correct_xpow_obj, dfb_y_obj);
            mul_tiles_bcast_cols(dfb::correct_xpow, dfb::y, 0, 0, dst0);
        } else {
            mul_tiles_init_with_dt(dfb_correct_xpow_obj, dfb_y_obj);
            mul_tiles(dfb::correct_xpow, dfb::y, 0, 0, dst0);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp4_obj);
        tile_regs_release();

        dfb_correct_xpow_obj.pop_front(onetile);
        dfb_tmp4_obj.push_back(onetile);

        // x^(p - 1) * y * dy -> cb_tmp5
        dfb_tmp4_obj.wait_front(onetile);
        dfb_tmp5_obj.reserve_back(onetile);

        tile_regs_acquire();
        if (ht_need_bcast && wt_need_bcast) {
            mul_bcast_scalar_init_with_dt(dfb_tmp4_obj, dfb_dy_obj);
            mul_tiles_bcast_scalar(dfb::tmp4, dfb::dy, 0, 0, dst0);
        } else if (ht_need_bcast) {
            mul_bcast_rows_init_with_dt(dfb_tmp4_obj, dfb_dy_obj);
            mul_tiles_bcast_rows(dfb::tmp4, dfb::dy, 0, 0, dst0);
        } else if (wt_need_bcast) {
            mul_bcast_cols_init_with_dt(dfb_tmp4_obj, dfb_dy_obj);
            mul_tiles_bcast_cols(dfb::tmp4, dfb::dy, 0, 0, dst0);
        } else {
            mul_tiles_init_with_dt(dfb_tmp4_obj, dfb_dy_obj);
            mul_tiles(dfb::tmp4, dfb::dy, 0, 0, dst0);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp5_obj);
        tile_regs_release();

        dfb_tmp4_obj.pop_front(onetile);
        dfb_tmp5_obj.push_back(onetile);

        // 1 / y^p
        power_and_recip_tile_to_cb(
            dfb_y_obj,
            dfb_xpow_obj,
            dfb_logx_obj,
            dfb_decimal_obj,
            dfb_exp_lxmd_obj,
            dfb_recip_ypow_obj,
            p,
            p_is_negative);

        // (x^(p - 1) * y * dy) / y^p -> cb_dx
        dfb_tmp5_obj.wait_front(onetile);
        dfb_recip_ypow_obj.wait_front(onetile);
        dfb_tmp4_obj.reserve_back(onetile);

        tile_regs_acquire();
        if (ht_need_bcast && wt_need_bcast) {
            mul_bcast_scalar_init_with_dt(dfb_tmp5_obj, dfb_recip_ypow_obj);
            mul_tiles_bcast_scalar(dfb::tmp5, dfb::recip_ypow, 0, 0, dst0);
        } else if (ht_need_bcast) {
            mul_bcast_rows_init_with_dt(dfb_tmp5_obj, dfb_recip_ypow_obj);
            mul_tiles_bcast_rows(dfb::tmp5, dfb::recip_ypow, 0, 0, dst0);
        } else if (wt_need_bcast) {
            mul_bcast_cols_init_with_dt(dfb_tmp5_obj, dfb_recip_ypow_obj);
            mul_tiles_bcast_cols(dfb::tmp5, dfb::recip_ypow, 0, 0, dst0);
        } else {
            mul_tiles_init_with_dt(dfb_tmp5_obj, dfb_recip_ypow_obj);
            mul_tiles(dfb::tmp5, dfb::recip_ypow, 0, 0, dst0);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, dfb_tmp4_obj);
        tile_regs_release();

        dfb_tmp5_obj.pop_front(onetile);
        dfb_recip_ypow_obj.pop_front(onetile);
        dfb_tmp4_obj.push_back(onetile);

        dfb_dy_obj.pop_front(onetile);

        // multiply abs sign
        mul_tiles_to_cb(dfb_sign_obj, dfb_tmp4_obj, dfb_dx_obj, 0, 0);
    }

    dfb_decimal_obj.pop_front(onetile);
}
