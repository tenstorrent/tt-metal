// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // PowerIterative, Recip, Log, Exp
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);
    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;

    constexpr uint32_t cb_input = 0;
    constexpr uint32_t cb_decimal = 1;
    DataflowBuffer dfb_decimal_obj(cb_decimal);

    // x^p * exp(log(x) * decimal)
    constexpr uint32_t cb_y = 16;

    constexpr uint32_t cb_x = 24;
    constexpr uint32_t cb_xpow = 25;
    constexpr uint32_t cb_logx = 26;
    constexpr uint32_t cb_exp_lxmd = 27;

    constexpr uint32_t onetile = 1;

    if (num_tiles > 1) {
        compute_kernel_hw_startup(cb_input, cb_x, cb_y);
    } else {
        compute_kernel_hw_startup(cb_logx, cb_decimal, cb_y);
    }

    dfb_decimal_obj.wait_front(onetile);

    // Compute cb_x
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        if (tile_idx == 0) {
            copy_tile_to_cb<cb_input, cb_x>();
        } else {
            add_tiles_to_cb<cb_input, cb_x, cb_x>();
        }
    }
    // x^p
    power_tile_to_cb<cb_x, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_y>(p, p_is_negative);
}
