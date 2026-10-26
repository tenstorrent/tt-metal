// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_input = 0;  // input(==tmp_pow_sum)
    DataflowBuffer dfb_input_obj(cb_input);
    constexpr uint32_t cb_decimal = 1;
    DataflowBuffer dfb_decimal_obj(cb_decimal);

    // x^p * exp(log(x) * decimal)
    constexpr uint32_t cb_y = 16;  // output(==total_norm)

    constexpr uint32_t cb_x = 24;         // Sum[tmp_pow_sum](==x)
    constexpr uint32_t cb_xpow = 25;      // x^p
    constexpr uint32_t cb_logx = 26;      // log(x)
    constexpr uint32_t cb_exp_lxmd = 27;  // exp(log(x) * decimal)
    DataflowBuffer dfb_y_obj(cb_y);
    DataflowBuffer dfb_x_obj(cb_x);
    DataflowBuffer dfb_xpow_obj(cb_xpow);
    DataflowBuffer dfb_logx_obj(cb_logx);
    DataflowBuffer dfb_exp_lxmd_obj(cb_exp_lxmd);

    constexpr uint32_t onetile = 1;

    if (num_tiles > 1) {
        compute_kernel_hw_startup(cb_input, cb_x, cb_y);
    } else {
        compute_kernel_hw_startup(cb_logx, cb_decimal, cb_y);
    }

    dfb_decimal_obj.wait_front(onetile);  // comes from the reader

    // Compute cb_x
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        if (tile_idx == 0) {
            copy_tile_to_dfb<cb_input, cb_x>(dfb_input_obj, dfb_x_obj);
        } else {
            add_tiles_to_dfb<cb_input, cb_x, cb_x>(dfb_input_obj, dfb_x_obj, dfb_x_obj);
        }
    }
    // x^p
    power_tile_to_dfb<cb_x, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_y>(
        dfb_x_obj, dfb_xpow_obj, dfb_logx_obj, dfb_decimal_obj, dfb_exp_lxmd_obj, dfb_y_obj, p, p_is_negative);
}
