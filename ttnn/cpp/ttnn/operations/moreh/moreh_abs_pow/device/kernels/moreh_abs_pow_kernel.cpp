// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "api/debug/dprint.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_rows_per_core = get_arg_val<uint32_t>(i++);
    const auto Wt = get_arg_val<uint32_t>(i++);
    const auto origin_w = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    std::uint8_t input_id{tt::CB::c_in0};
    const auto cb_x = input_id++;        // input
    const auto cb_one = input_id++;      // one
    const auto cb_decimal = input_id++;  // decimal
    const auto cb_mask_w = input_id++;   // mask_w

    std::uint8_t output_id{tt::CB::c_out0};
    const auto cb_y = output_id++;  // output

    std::uint8_t intermed_id{tt::CB::c_intermed0};
    const auto cb_tmp0 = intermed_id++;
    const auto cb_tmp1 = intermed_id++;
    const auto cb_tmp2 = intermed_id++;
    const auto cb_tmp3 = intermed_id++;

    const auto cb_xabs = cb_tmp0;      // |x|
    const auto cb_xpow = cb_tmp1;      // |x|^p
    const auto cb_logx = cb_tmp2;      // log(|x|)
    const auto cb_exp_lxmd = cb_tmp3;  // exp(log(|x|) * decimal)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0, tt::CB::c_out0);

    DataflowBuffer dfb_x_obj(cb_x);
    DataflowBuffer dfb_one_obj(cb_one);
    DataflowBuffer dfb_decimal_obj(cb_decimal);
    DataflowBuffer dfb_mask_w_obj(cb_mask_w);
    DataflowBuffer dfb_y_obj(cb_y);
    DataflowBuffer dfb_xabs_obj(cb_xabs);
    DataflowBuffer dfb_xpow_obj(cb_xpow);
    DataflowBuffer dfb_logx_obj(cb_logx);
    DataflowBuffer dfb_exp_lxmd_obj(cb_exp_lxmd);

    dfb_one_obj.wait_front(onetile);
    dfb_decimal_obj.wait_front(onetile);

    constexpr uint32_t TILE_W = 32;
    const bool do_mask_w = (origin_w % TILE_W) != 0;
    const auto mask_w = do_mask_w ? (origin_w % TILE_W) : TILE_W;

    if (do_mask_w) {
        dfb_mask_w_obj.wait_front(onetile);
    }
    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            tile_regs_acquire();
            dfb_x_obj.wait_front(onetile);
            dfb_xabs_obj.reserve_back(onetile);

            copy_tile_init_with_dt(dfb_x_obj);
            copy_tile(cb_x, 0, dst0);

            if (do_mask_w && (col_idx == Wt - 1)) {
                copy_tile_init_with_dt(dfb_mask_w_obj);
                copy_tile(cb_mask_w, 0, dst1);

                mask_tile_init();
                mask_tile(dst0, dst1);
            }

            abs_tile_init();
            abs_tile(dst0);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, dfb_xabs_obj);
            tile_regs_release();

            dfb_x_obj.pop_front(onetile);
            dfb_xabs_obj.push_back(onetile);

            power_tile_to_cb(
                dfb_xabs_obj,
                dfb_xpow_obj,
                dfb_logx_obj,
                dfb_decimal_obj,
                dfb_exp_lxmd_obj,
                dfb_y_obj,
                p,
                p_is_negative);
        }
    }

    dfb_one_obj.pop_front(onetile);
    dfb_decimal_obj.pop_front(onetile);
    if (do_mask_w) {
        dfb_mask_w_obj.pop_front(onetile);
    }
}
