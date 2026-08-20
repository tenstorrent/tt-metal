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

    constexpr uint32_t dfb_input_id = 0;  // input(==tmp_pow_sum)
    constexpr uint32_t dfb_decimal_id = 1;
    DataflowBuffer dfb_decimal_obj(dfb_decimal_id);

    // x^p * exp(log(x) * decimal)
    constexpr uint32_t dfb_y_id = 16;  // output(==total_norm)

    constexpr uint32_t dfb_x_id = 24;         // Sum[tmp_pow_sum](==x)
    constexpr uint32_t dfb_xpow_id = 25;      // x^p
    constexpr uint32_t dfb_logx_id = 26;      // log(x)
    constexpr uint32_t dfb_exp_lxmd_id = 27;  // exp(log(x) * decimal)

    constexpr uint32_t onetile = 1;

    if (num_tiles > 1) {
        compute_kernel_hw_startup(dfb_input_id, dfb_x_id, dfb_y_id);
    } else {
        compute_kernel_hw_startup(dfb_logx_id, dfb_decimal_id, dfb_y_id);
    }

    dfb_decimal_obj.wait_front(onetile);  // comes from the reader

    // Compute dfb_x_id
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        if (tile_idx == 0) {
            copy_tile_to_dfb<dfb_input_id, dfb_x_id>();
        } else {
            add_tiles_to_dfb<dfb_input_id, dfb_x_id, dfb_x_id>();
        }
    }
    // x^p
    power_tile_to_dfb<dfb_x_id, dfb_xpow_id, dfb_logx_id, dfb_decimal_id, dfb_exp_lxmd_id, dfb_y_id>(p, p_is_negative);
}
