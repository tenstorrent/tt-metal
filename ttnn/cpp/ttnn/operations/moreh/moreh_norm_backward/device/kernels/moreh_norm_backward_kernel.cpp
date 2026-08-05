// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr bool wt_need_bcast = (get_compile_time_arg_val(1) == 1);
    constexpr bool ht_need_bcast = (get_compile_time_arg_val(2) == 1);

    // runtime args
    int i{0};
    const auto num_input_tiles_per_core = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto p_minus_one = get_arg_val<uint32_t>(i++);
    const bool p_minus_one_is_negative = get_arg_val<uint32_t>(i++) == 1;

    std::uint8_t input_id{tt::CBIndex::c_0};
    const auto cb_x = input_id++;
    DataflowBuffer dfb_x_obj(cb_x);  // input(==x)
    const auto cb_y = input_id++;
    DataflowBuffer dfb_y_obj(cb_y);  // output(==y)
    const auto cb_dy = input_id++;
    DataflowBuffer dfb_dy_obj(cb_dy);  // output_grad(==dy)
    const auto cb_decimal = input_id++;
    DataflowBuffer dfb_decimal_obj(cb_decimal);  // decimal

    std::uint8_t output_id{tt::CBIndex::c_16};
    const auto cb_dx = output_id++;  // input_grad(==dx)
    DataflowBuffer dfb_dx_obj(cb_dx);

    std::uint8_t intermed_id{tt::CBIndex::c_24};
    const auto cb_tmp0 = intermed_id++;
    const auto cb_tmp1 = intermed_id++;
    const auto cb_tmp2 = intermed_id++;
    const auto cb_tmp3 = intermed_id++;
    const auto cb_tmp4 = intermed_id++;
    DataflowBuffer dfb_tmp4_obj(cb_tmp4);
    const auto cb_tmp5 = intermed_id++;
    DataflowBuffer dfb_tmp5_obj(cb_tmp5);
    const auto cb_tmp6 = intermed_id++;
    const auto cb_tmp7 = intermed_id++;

    const auto cb_xpow = cb_tmp0;
    DataflowBuffer dfb_xpow_obj(cb_xpow);
    const auto cb_logx = cb_tmp1;
    DataflowBuffer dfb_logx_obj(cb_logx);
    const auto cb_exp_lxmd = cb_tmp2;
    DataflowBuffer dfb_exp_lxmd_obj(cb_exp_lxmd);
    const auto cb_correct_xpow = cb_tmp3;
    DataflowBuffer dfb_correct_xpow_obj(cb_correct_xpow);
    const auto cb_recip_ypow = cb_tmp6;
    DataflowBuffer dfb_recip_ypow_obj(cb_recip_ypow);
    const auto cb_sign = cb_tmp7;
    DataflowBuffer dfb_sign_obj(cb_sign);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);
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
            mul_tiles_bcast_scalar_init_short_with_dt(dfb_correct_xpow_obj, dfb_y_obj);
            mul_tiles_bcast_scalar(cb_correct_xpow, cb_y, 0, 0, dst0);
        } else if (ht_need_bcast) {
            mul_bcast_rows_init_short_with_dt(dfb_correct_xpow_obj, dfb_y_obj);
            mul_tiles_bcast_rows(cb_correct_xpow, cb_y, 0, 0, dst0);
        } else if (wt_need_bcast) {
            mul_bcast_cols_init_short_with_dt(dfb_correct_xpow_obj, dfb_y_obj);
            mul_tiles_bcast_cols(cb_correct_xpow, cb_y, 0, 0, dst0);
        } else {
            mul_tiles_init_with_dt(dfb_correct_xpow_obj, dfb_y_obj);
            mul_tiles(cb_correct_xpow, cb_y, 0, 0, dst0);
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
            mul_tiles_bcast_scalar_init_short_with_dt(dfb_tmp4_obj, dfb_dy_obj);
            mul_tiles_bcast_scalar(cb_tmp4, cb_dy, 0, 0, dst0);
        } else if (ht_need_bcast) {
            mul_bcast_rows_init_short_with_dt(dfb_tmp4_obj, dfb_dy_obj);
            mul_tiles_bcast_rows(cb_tmp4, cb_dy, 0, 0, dst0);
        } else if (wt_need_bcast) {
            mul_bcast_cols_init_short_with_dt(dfb_tmp4_obj, dfb_dy_obj);
            mul_tiles_bcast_cols(cb_tmp4, cb_dy, 0, 0, dst0);
        } else {
            mul_tiles_init_with_dt(dfb_tmp4_obj, dfb_dy_obj);
            mul_tiles(cb_tmp4, cb_dy, 0, 0, dst0);
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
            mul_tiles_bcast_scalar_init_short_with_dt(dfb_tmp5_obj, dfb_recip_ypow_obj);
            mul_tiles_bcast_scalar(cb_tmp5, cb_recip_ypow, 0, 0, dst0);
        } else if (ht_need_bcast) {
            mul_bcast_rows_init_short_with_dt(dfb_tmp5_obj, dfb_recip_ypow_obj);
            mul_tiles_bcast_rows(cb_tmp5, cb_recip_ypow, 0, 0, dst0);
        } else if (wt_need_bcast) {
            mul_bcast_cols_init_short_with_dt(dfb_tmp5_obj, dfb_recip_ypow_obj);
            mul_tiles_bcast_cols(cb_tmp5, cb_recip_ypow, 0, 0, dst0);
        } else {
            mul_tiles_init_with_dt(dfb_tmp5_obj, dfb_recip_ypow_obj);
            mul_tiles(cb_tmp5, cb_recip_ypow, 0, 0, dst0);
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
