// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "api/debug/dprint.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const auto num_rows_per_core = get_arg(args::num_rows_per_core);
    const auto Wt = get_arg(args::Wt);
    const auto origin_w = get_arg(args::origin_w);
    const auto p = get_arg(args::p);
    const bool p_is_negative = get_arg(args::p_is_negative) == 1;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;

    // Input/output roles map to the c_in0 / c_out0 DFBs (input == x, output == y).
    compute_kernel_hw_startup(dfb::x, dfb::x, dfb::y);

    DataflowBuffer dfb_x_obj(dfb::x);                // input
    DataflowBuffer dfb_one_obj(dfb::one);            // one
    DataflowBuffer dfb_decimal_obj(dfb::decimal);    // decimal
    DataflowBuffer dfb_mask_w_obj(dfb::mask_w);      // mask_w
    DataflowBuffer dfb_y_obj(dfb::y);                // output
    DataflowBuffer dfb_xabs_obj(dfb::xabs);          // |x|
    DataflowBuffer dfb_xpow_obj(dfb::xpow);          // |x|^p
    DataflowBuffer dfb_logx_obj(dfb::logx);          // log(|x|)
    DataflowBuffer dfb_exp_lxmd_obj(dfb::exp_lxmd);  // exp(log(|x|) * decimal)

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
            copy_tile(dfb::x, 0, dst0);

            if (do_mask_w && (col_idx == Wt - 1)) {
                copy_tile_init_with_dt(dfb_mask_w_obj);
                copy_tile(dfb::mask_w, 0, dst1);

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
