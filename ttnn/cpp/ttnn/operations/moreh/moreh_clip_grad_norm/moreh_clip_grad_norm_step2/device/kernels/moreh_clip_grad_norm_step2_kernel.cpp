// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    std::uint8_t input_id{0};
    const auto cb_input = input_id++;
    DataflowBuffer dfb_input_obj(cb_input);  // input(==tmp_pow_sum)
    const auto cb_decimal = input_id++;
    DataflowBuffer dfb_decimal_obj(cb_decimal);  // decimal

    std::uint8_t output_id{16};
    // x^p * exp(log(x) * decimal)
    const auto cb_y = output_id++;  // output(==total_norm)
    DataflowBuffer dfb_y_obj(cb_y);

    std::uint8_t intermed_id{24};
    const auto cb_x = intermed_id++;
    DataflowBuffer dfb_x_obj(cb_x);          // Sum[tmp_pow_sum](==x)
    const auto cb_xpow = intermed_id++;      // x^p
    DataflowBuffer dfb_xpow_obj(cb_xpow);
    const auto cb_logx = intermed_id++;      // log(x)
    DataflowBuffer dfb_logx_obj(cb_logx);
    const auto cb_exp_lxmd = intermed_id++;  // exp(log(x) * decimal)
    DataflowBuffer dfb_exp_lxmd_obj(cb_exp_lxmd);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    if (num_tiles > 1) {
        compute_kernel_hw_startup(cb_input, cb_x, cb_y);
    } else {
        compute_kernel_hw_startup(cb_logx, cb_decimal, cb_y);
    }

    dfb_decimal_obj.wait_front(onetile);  // comes from the reader

    // Compute cb_x
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        if (tile_idx == 0) {
            tile_regs_acquire();
            dfb_input_obj.wait_front(onetile);  // comes from the reader
            dfb_x_obj.reserve_back(onetile);

            copy_tile_init(cb_input);
            copy_tile(cb_input, 0, dst0);
            dfb_input_obj.pop_front(onetile);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(dst0, cb_x);
            dfb_x_obj.push_back(onetile);
            tile_regs_release();
        } else {
            tile_regs_acquire();
            dfb_input_obj.wait_front(onetile);  // comes from the reader
            dfb_x_obj.wait_front(onetile);
            dfb_x_obj.reserve_back(onetile);

            add_init(cb_input, cb_x);
            add_tiles(cb_input, cb_x, 0, 0, dst0);
            dfb_x_obj.pop_front(onetile);
            dfb_input_obj.pop_front(onetile);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(dst0, cb_x);
            dfb_x_obj.push_back(onetile);
            tile_regs_release();
        }
    }
    // x^p
    power_tile_to_cb(
        dfb_x_obj, dfb_xpow_obj, dfb_logx_obj, dfb_decimal_obj, dfb_exp_lxmd_obj, dfb_y_obj, p, p_is_negative);
}
